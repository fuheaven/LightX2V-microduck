//! Optional external action source for the existing robotd controller.
//! No motor IO is performed here. The normal Safety::apply path still owns
//! limits, torque and fall handling. Enable explicitly with a private Unix
//! socket in MICRODUCK_EXTERNAL_POLICY_SOCKET before starting robotd.
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::{Arc, Mutex, OnceLock};
use std::sync::atomic::{AtomicU64, Ordering};
use std::collections::{BTreeSet, VecDeque};
use std::time::{Duration, Instant};
use serde_json::{Value, json};

const MAX_AGE_TICKS: u64 = 10;
const MAX_AGE: Duration = Duration::from_millis(200);
const MAX_LINE: u64 = 64 * 1024;
static HUB: OnceLock<Option<Arc<Hub>>> = OnceLock::new();
static MISSED_RECEIPT_TICKS: AtomicU64 = AtomicU64::new(0);
static CONTROL_TICKS: AtomicU64 = AtomicU64::new(0);
const RECEIPT_CAPACITY: usize = 4096;
const LEGACY_MODE: &str = "source_tick_age_compensated";
const PREFIX_MODE: &str = "prefix_playback_v1";
const SUPPORTED_EXECUTION_MODES: [&str; 2] = [LEGACY_MODE, PREFIX_MODE];
const PREDICTION_HORIZON: u64 = 50;
const EXECUTION_PREFIX: u64 = 5;

struct Chunk {
    source_capture: Option<duck_control::io::RemoteCapture>,
    source_tick: u64,
    execution_mode: &'static str,
    admission_tick: u64,
    playback_start_tick: u64,
    execution_prefix: Option<u64>,
    prediction_horizon: Option<u64>,
    consumed_rows: usize,
    valid_until: u64,
    rows: Vec<[f32; 14]>,
    received: Instant,
}

struct State {
    retired_episodes: BTreeSet<String>,
    observation_sources: VecDeque<(u64, Option<duck_control::io::RemoteCapture>)>,
    receipts: VecDeque<Arc<Value>>,
    receipt_sequence: u64,
    last_consumed: Option<Value>,
    last_control_ns: Option<u64>,
    source_capture: Option<duck_control::io::RemoteCapture>,
    permitted: bool,
    episode: Option<String>,
    execution_mode: &'static str,
    tick: u64,
    observation: Vec<f32>,
    observed: Instant,
    chunk: Option<Chunk>,
    last_source: Option<u64>,
    accepted: u64,
    rejected: u64,
    external_ticks: u64,
    fallback_ticks: u64,
    reason: String,
}

impl Default for State {
    fn default() -> Self {
        Self { retired_episodes: BTreeSet::new(), observation_sources: VecDeque::new(), receipts: VecDeque::new(), receipt_sequence: 0,
            last_consumed: None, last_control_ns: None,
            source_capture: None, permitted: false, episode: None, execution_mode: LEGACY_MODE, tick: 0, observation: vec![], observed: Instant::now(), chunk: None,
            last_source: None, accepted: 0, rejected: 0, external_ticks: 0, fallback_ticks: 0,
            reason: "inactive".into() }
    }
}

struct ReadSnapshot {
    observation: Value,
    observed: Instant,
    receipts: VecDeque<Arc<Value>>,
    receipt_sequence: u64,
}

impl ReadSnapshot {
    fn from_state(state: &State) -> Self {
        Self { observation: observation_reply(state), observed: state.observed,
            receipts: state.receipts.clone(), receipt_sequence: state.receipt_sequence }
    }
}

struct Hub { state: Mutex<State>, published: arc_swap::ArcSwap<ReadSnapshot> }

impl Hub {
    fn new() -> Self {
        let state = State::default();
        let published = arc_swap::ArcSwap::from_pointee(ReadSnapshot::from_state(&state));
        Self { state: Mutex::new(state), published }
    }
    fn publish(&self, state: &State) { self.published.store(Arc::new(ReadSnapshot::from_state(state))); }
    fn read(&self, method: &str, params: &Value) -> Option<Result<Value, String>> {
        if !matches!(method, "policy.external.observation" | "policy.external.status" | "policy.external.receipts") { return None; }
        let snapshot = self.published.load_full();
        Some(if method != "policy.external.receipts" {
            let mut value = snapshot.observation.clone();
            value["observation_age_s"] = json!(snapshot.observed.elapsed().as_secs_f64());
            value["unrecorded_control_ticks"] = json!(MISSED_RECEIPT_TICKS.load(Ordering::Relaxed));
            Ok(value)
        } else {
            receipts_reply(&snapshot.receipts, snapshot.receipt_sequence, params)
        })
    }
}

fn receipt(state: &mut State, kind: &str, detail: Value) -> Value {
    state.receipt_sequence += 1;
    let mut value = json!({"sequence": state.receipt_sequence, "kind": kind,
        "monotonic_ns": duck_ipc_proto::clock::monotonic_ns(),
        "daemon_tick": state.tick, "episode_id": state.episode, "execution_mode": state.execution_mode,
        "control_capture": state.source_capture,
        "consume_boundary": "controller_external_buffer_before_safety_and_io"});
    if let Some(fields) = detail.as_object() {
        value.as_object_mut().unwrap().extend(fields.clone());
    }
    state.receipts.push_back(Arc::new(value.clone()));
    while state.receipts.len() > RECEIPT_CAPACITY { state.receipts.pop_front(); }
    value
}

thread_local! {
    static SOURCE_CAPTURE: std::cell::RefCell<Option<duck_control::io::RemoteCapture>> = const { std::cell::RefCell::new(None) };
}

/// Called by the control loop after its sensor read, before this tick's step.
/// Coast over a failed read retains the original capture timestamp and body tick.
pub fn set_source_capture(capture: Option<duck_control::io::RemoteCapture>) {
    // Setter and sample run synchronously in the same controller.step call path
    // with no await between them. No RPC mutex is taken on the control thread.
    SOURCE_CAPTURE.with(|slot| *slot.borrow_mut() = capture);
}

pub fn initialize() {
    HUB.get_or_init(|| {
        let path = std::env::var("MICRODUCK_EXTERNAL_POLICY_SOCKET").ok()?;
        let listener = UnixListener::bind(&path).expect("external-policy socket must be new and writable");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o660)).expect("external socket permissions");
        let hub = Arc::new(Hub::new());
        let worker = hub.clone();
        std::thread::Builder::new().name("external-policy-ipc".into()).spawn(move || {
            // One short-lived RPC connection at a time. Control never waits on
            // this lock, and a stalled client is bounded by the read timeout.
            for stream in listener.incoming().flatten() {
                if let Err(error) = serve(stream, &worker) {
                    tracing::debug!(%error, "external policy connection closed");
                }
            }
        }).expect("external policy IPC thread");
        Some(hub)
    });
}

pub fn revoke(reason: &str) {
    if let Some(Some(hub)) = HUB.get() {
        if let Ok(mut state) = hub.state.lock() {
            if state.episode.is_some() { receipt(&mut state, "revoke", json!({"reason": reason})); }
            if let Some(episode) = state.episode.clone() { state.retired_episodes.insert(episode); }
            state.episode = None;
            state.chunk = None;
            state.last_source = None;
            state.reason = reason.into();
            hub.publish(&state);
        }
    }
}

/// Publish a single-tick snapshot and, if admissible, consume exactly one row.
/// A contended RPC mutex simply leaves the official policy in control this tick.
pub fn sample(observation: &[f32], permitted: bool) -> Option<[f32; 14]> {
    let Some(Some(hub)) = HUB.get() else { return None };
    // Failed lock acquisition is still a real control tick. Never extend a
    // packet's tick lifetime or shift its action indices after contention.
    let tick = CONTROL_TICKS.fetch_add(1, Ordering::Relaxed) + 1;
    let Ok(mut state) = hub.state.try_lock() else {
        MISSED_RECEIPT_TICKS.fetch_add(1, Ordering::Relaxed);
        return None;
    };
    let result = sample_state_at_tick(&mut state, observation, permitted, SOURCE_CAPTURE.with(|slot| slot.borrow().clone()), tick);
    hub.publish(&state);
    result
}

#[cfg(test)]
fn sample_state(state: &mut State, observation: &[f32], permitted: bool, capture: Option<duck_control::io::RemoteCapture>) -> Option<[f32; 14]> {
    sample_state_at_tick(state, observation, permitted, capture, state.tick + 1)
}

fn sample_state_at_tick(state: &mut State, observation: &[f32], permitted: bool, capture: Option<duck_control::io::RemoteCapture>, tick: u64) -> Option<[f32; 14]> {
    let control_ns = duck_ipc_proto::clock::monotonic_ns();
    let period_ns = state.last_control_ns.map(|last| control_ns.saturating_sub(last));
    state.last_control_ns = Some(control_ns);
    state.tick = tick;
    state.observation.clear();
    state.observation.extend_from_slice(observation);
    state.source_capture = capture;
    state.observation_sources.push_back((state.tick, state.source_capture.clone()));
    while state.observation_sources.len() > 64 { state.observation_sources.pop_front(); }
    state.observed = Instant::now();
    state.permitted = permitted;
    if !permitted {
        if state.episode.is_some() { receipt(state, "revoke", json!({"reason": "local_control_preempted", "control_period_ns": period_ns})); }
        if let Some(episode) = state.episode.clone() { state.retired_episodes.insert(episode); }
        state.episode = None;
        state.chunk = None;
        state.reason = "local_control_preempted".into();
        return None;
    }
    if state.episode.is_none() { return None; }
    let tick = state.tick;
    let (selected, details) = match state.chunk.as_ref() {
        None => (None, json!({"reason": "no_chunk"})),
        Some(chunk) => {
            let index = tick.saturating_sub(chunk.playback_start_tick) as usize;
            let reason = if tick >= chunk.valid_until { "expired_tick" }
                else if chunk.source_capture.as_ref().is_some_and(|capture| Some(capture.episode_id.as_str()) != state.episode.as_deref()) { "source_capture_episode_mismatch" }
                else if chunk.source_capture.as_ref().is_some_and(|capture| control_ns < capture.capture_monotonic_ns
                    || control_ns - capture.capture_monotonic_ns >= MAX_AGE.as_nanos() as u64) { "stale_capture" }
                else if chunk.received.elapsed() >= MAX_AGE { "expired_wall" }
                else if tick < chunk.playback_start_tick { "playback_not_started" }
                else if index >= chunk.rows.len() { "chunk_exhausted" }
                else { "external" };
            let selected = if reason == "external" { chunk.rows.get(index).copied() } else { None };
            (selected, json!({"reason": reason, "source_tick": chunk.source_tick, "action_index": index,
                "execution_mode": chunk.execution_mode, "admission_tick": chunk.admission_tick,
                "playback_start_tick": chunk.playback_start_tick, "execution_prefix": chunk.execution_prefix,
                "prediction_horizon": chunk.prediction_horizon,
                "valid_until_tick": chunk.valid_until, "source_capture": chunk.source_capture,
                "received_age_ns": chunk.received.elapsed().as_nanos() as u64, "raw_action14": selected}))
        }
    };
    let mut details = details;
    // Timestamp the actual buffer decision, not the later JSON allocation.
    details["monotonic_ns"] = json!(control_ns);
    details["control_period_ns"] = json!(period_ns);
    details["execution_source"] = json!(if selected.is_some() { "external" } else { "official_fallback" });
    let event = receipt(state, "consume", details);
    if selected.is_some() {
        if let Some(chunk) = state.chunk.as_mut() { chunk.consumed_rows += 1; }
        state.last_consumed = Some(event);
        state.external_ticks += 1;
        state.reason = "external".into();
    } else {
        state.chunk = None;
        state.fallback_ticks += 1;
        state.reason = "official_fallback".into();
    }
    selected
}

fn request(state: &mut State, method: &str, params: &Value) -> Result<Value, String> {
    match method {
        "policy.external.open" => {
            let episode = params["episode_id"].as_str().filter(|s| !s.is_empty() && s.len() <= 128).ok_or("invalid episode_id")?;
            let execution_mode = match params.get("execution_mode") {
                None => LEGACY_MODE,
                Some(value) if value.as_str() == Some(LEGACY_MODE) => LEGACY_MODE,
                Some(value) if value.as_str() == Some(PREFIX_MODE) => PREFIX_MODE,
                _ => return Err("unsupported execution_mode".into()),
            };
            if state.episode.is_some() { return Err("external session already active; close it first".into()); }
            if state.retired_episodes.contains(episode) { return Err("retired_episode; use a new episode_id".into()); }
            if state.observation.len() != 61 || state.observed.elapsed() >= MAX_AGE { return Err("no fresh controller observation; enable the official policy first".into()); }
            state.episode = Some(episode.into());
            state.execution_mode = execution_mode;
            state.chunk = None;
            state.last_source = None;
            state.accepted = 0;
            state.rejected = 0;
            state.external_ticks = 0;
            state.fallback_ticks = 0;
            state.last_consumed = None;
            receipt(state, "open", json!({}));
            Ok(json!({"episode_id": episode, "tick": state.tick, "control_hz": 50, "protocol": 1,
                "execution_mode": state.execution_mode, "supported_execution_modes": SUPPORTED_EXECUTION_MODES,
                "execution_receipt_protocol": 1, "receipt_sequence": state.receipt_sequence,
                "unrecorded_control_ticks": MISSED_RECEIPT_TICKS.load(Ordering::Relaxed)}))
        }
        "policy.external.close" => {
            if params["episode_id"].as_str() != state.episode.as_deref() { return Err("wrong_episode".into()); }
            receipt(state, "close", json!({"reason": "session_closed", "buffer_cleared": true}));
            if let Some(episode) = state.episode.clone() { state.retired_episodes.insert(episode); }
            state.episode = None;
            state.chunk = None;
            state.last_source = None;
            state.reason = "closed".into();
            Ok(json!({"closed": true}))
        }
        "policy.external.observation" | "policy.external.status" => Ok(observation_reply(state)),
        "policy.external.receipts" => receipts_reply(&state.receipts, state.receipt_sequence, params),
        "policy.external.push_chunk" => {
            let result = parse_chunk(state, params);
            match result {
                Ok(chunk) => {
                    let source = chunk.source_tick;
                    receipt(state, "submit", json!({"result": "accepted", "source_tick": source, "packet_episode_id": params["episode_id"],
                        "execution_mode": chunk.execution_mode, "admission_tick": chunk.admission_tick,
                        "playback_start_tick": chunk.playback_start_tick, "execution_prefix": chunk.execution_prefix,
                        "prediction_horizon": chunk.prediction_horizon,
                        "valid_until_tick": chunk.valid_until, "rows": chunk.rows.len(), "source_capture": chunk.source_capture}));
                    let reply = json!({"accepted": true, "source_tick": source, "tick": state.tick,
                        "execution_mode": chunk.execution_mode, "admission_tick": chunk.admission_tick,
                        "playback_start_tick": chunk.playback_start_tick, "execution_prefix": chunk.execution_prefix,
                        "prediction_horizon": chunk.prediction_horizon});
                    state.last_source = Some(source);
                    state.chunk = Some(chunk);
                    state.accepted += 1;
                    Ok(reply)
                }
                Err(error) => {
                    receipt(state, "submit", json!({"result": "rejected", "reason": error,
                        "packet_episode_id": params["episode_id"].as_str().filter(|value| value.len() <= 128),
                        "source_tick": params["source_tick"].as_u64()}));
                    state.rejected += 1; Err(error)
                }
            }
        }
        _ => Err("unknown external policy method".into()),
    }
}


fn observation_reply(state: &State) -> Value {
    json!({
            "protocol": 1, "episode_id": state.episode, "tick": state.tick,
            "execution_mode": state.execution_mode, "supported_execution_modes": SUPPORTED_EXECUTION_MODES,
            "source_capture_protocol": 1, "source_capture": state.source_capture,
            "external_control_permitted": state.permitted,
            "execution_receipt_protocol": 1, "receipt_sequence": state.receipt_sequence,
            "last_consumed": state.last_consumed, "unrecorded_control_ticks": MISSED_RECEIPT_TICKS.load(Ordering::Relaxed),
            "state": state.observation, "observation_age_s": state.observed.elapsed().as_secs_f64(),
            "control_hz": 50, "accepted": state.accepted, "rejected": state.rejected,
            "external_ticks": state.external_ticks, "fallback_ticks": state.fallback_ticks,
            "reason": state.reason, "hardware_validated": false,
        })
}

fn receipts_reply(receipts: &VecDeque<Arc<Value>>, sequence: u64, params: &Value) -> Result<Value, String> {
    let after = params["after_sequence"].as_u64().ok_or("invalid receipt cursor")?;
            let limit = params["limit"].as_u64().unwrap_or(16).min(16) as usize;
            if limit == 0 || after > sequence { return Err("invalid receipt cursor/limit".into()); }
            let oldest = receipts.front().and_then(|row| row["sequence"].as_u64()).unwrap_or(sequence + 1);
            let events: Vec<_> = receipts.iter().filter(|row| row["sequence"].as_u64().unwrap() > after).take(limit).map(|row| row.as_ref().clone()).collect();
            let cursor = events.last().and_then(|row| row["sequence"].as_u64()).unwrap_or(after);
            Ok(json!({"protocol": 1, "events": events, "next_sequence": cursor, "latest_sequence": sequence,
                "oldest_sequence": oldest, "missing_events": oldest.saturating_sub(after + 1),
                "unrecorded_control_ticks": MISSED_RECEIPT_TICKS.load(Ordering::Relaxed)}))
}

fn parse_chunk(state: &State, params: &Value) -> Result<Chunk, String> {
    if state.episode.is_none() || params["episode_id"].as_str() != state.episode.as_deref() { return Err("wrong_episode".into()); }
    let prefix_mode = state.execution_mode == PREFIX_MODE;
    if (prefix_mode || params.get("execution_mode").is_some()) && params["execution_mode"].as_str() != Some(state.execution_mode) {
        return Err("execution_mode mismatch".into());
    }
    if prefix_mode && (params["prediction_horizon"].as_u64() != Some(PREDICTION_HORIZON)
        || params["execution_prefix"].as_u64() != Some(EXECUTION_PREFIX)) {
        return Err("prefix_playback_v1 requires prediction_horizon=50 and execution_prefix=5".into());
    }
    if state.observed.elapsed() >= MAX_AGE { return Err("stale_controller".into()); }
    if params["control_dt_s"].as_f64().is_none_or(|dt| (dt - 0.02).abs() > 1e-9) { return Err("control_dt_s must be .02".into()); }
    let source = params["source_tick"].as_u64().ok_or("invalid source_tick")?;
    let admission_tick = CONTROL_TICKS.load(Ordering::Relaxed).max(state.tick);
    let current_tick = if prefix_mode { admission_tick } else { state.tick };
    if source > state.tick { return Err("future_observation".into()); }
    if state.last_source.is_some_and(|last| source <= last) { return Err("out_of_order".into()); }
    if current_tick - source >= MAX_AGE_TICKS { return Err("stale_observation".into()); }
    let values = params["actions"].as_array().filter(|v| !v.is_empty() && v.len() <= 256).ok_or("actions must be [1..256,14]")?;
    if prefix_mode && values.len() != EXECUTION_PREFIX as usize { return Err("prefix_playback_v1 requires exactly five action rows".into()); }
    let mut rows = Vec::with_capacity(values.len());
    for value in values {
        let row = value.as_array().filter(|v| v.len() == 14).ok_or("action row must have 14 joints")?;
        let mut action = [0.0; 14];
        for (index, entry) in row.iter().enumerate() {
            let number = entry.as_f64().filter(|x| x.is_finite() && x.abs() <= 10.).ok_or("invalid joint residual")?;
            action[index] = number as f32;
        }
        rows.push(action);
    }
    // Prediction freshness remains anchored to the source observation. The
    // playback prefix is a separate bound and never renews this lifetime.
    let source_expiry = source.checked_add(MAX_AGE_TICKS).ok_or("source tick overflow")?;
    let horizon_expiry = if prefix_mode { source_expiry } else {
        source.checked_add(rows.len() as u64).ok_or("source tick overflow")?.min(source_expiry)
    };
    let valid_until = params["valid_until_tick"].as_u64().unwrap_or(horizon_expiry).min(horizon_expiry);
    if valid_until <= current_tick { return Err("expired".into()); }
    let playback_start_tick = if prefix_mode { admission_tick.checked_add(1).ok_or("playback tick overflow")? } else { source };
    let source_capture = state.observation_sources.iter().find(|(tick, _)| *tick == source).and_then(|(_, capture)| capture.clone());
    if state.source_capture.is_some() && source_capture.is_none() { return Err("unknown_source_capture".into()); }
    if source_capture.as_ref().is_some_and(|capture| Some(capture.episode_id.as_str()) != state.episode.as_deref()) {
        return Err("source_capture_episode_mismatch".into());
    }
    if source_capture.as_ref().is_some_and(|capture| {
        let now = duck_ipc_proto::clock::monotonic_ns();
        now < capture.capture_monotonic_ns || now - capture.capture_monotonic_ns >= MAX_AGE.as_nanos() as u64
    }) { return Err("stale_capture".into()); }
    if prefix_mode {
        if let Some(previous) = state.chunk.as_ref() {
            let now = duck_ipc_proto::clock::monotonic_ns();
            let expired = current_tick >= previous.valid_until
                || current_tick.saturating_sub(previous.playback_start_tick) >= previous.rows.len() as u64
                || previous.received.elapsed() >= MAX_AGE
                || previous.source_capture.as_ref().is_some_and(|capture| now < capture.capture_monotonic_ns
                    || now - capture.capture_monotonic_ns >= MAX_AGE.as_nanos() as u64);
            if previous.consumed_rows < previous.rows.len() && !expired { return Err("prefix_in_progress".into()); }
        }
    }
    Ok(Chunk { source_capture, source_tick: source, execution_mode: state.execution_mode,
        admission_tick, playback_start_tick, execution_prefix: prefix_mode.then_some(EXECUTION_PREFIX),
        prediction_horizon: prefix_mode.then_some(PREDICTION_HORIZON), consumed_rows: 0,
        valid_until, rows, received: Instant::now() })
}

fn serve(mut stream: UnixStream, hub: &Hub) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(2)))?;
    stream.set_write_timeout(Some(Duration::from_secs(2)))?;
    let mut line = Vec::new();
    BufReader::new(stream.try_clone()?).take(MAX_LINE + 1).read_until(b'\n', &mut line)?;
    if line.len() as u64 > MAX_LINE { return Ok(()); }
    let input: Value = match serde_json::from_slice(&line) { Ok(value) => value, Err(_) => return Ok(()) };
    let id = input.get("id").cloned().unwrap_or(Value::Null);
    let method = input["method"].as_str().unwrap_or("");
    // Read RPCs clone and serialize immutable snapshots outside the control
    // mutex. Mutations retain the existing authority and validation path.
    let result = if let Some(result) = hub.read(method, &input["params"]) { result } else {
        let mut state = hub.state.lock().map_err(|_| std::io::Error::other("external state poisoned"))?;
        let result = request(&mut state, method, &input["params"]);
        hub.publish(&state);
        result
    };
    let reply = match result {
        Ok(value) => json!({"jsonrpc": "2.0", "id": id, "result": value}),
        Err(error) => json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32602, "message": error}}),
    };
    writeln!(stream, "{reply}")
}

#[cfg(test)]
mod tests {
    use super::*;
    fn active() -> State {
        State { episode: Some("episode".into()), tick: 5, observation: vec![0.;61], ..State::default() }
    }
    #[test]
    fn read_rpc_uses_published_snapshot_while_control_mutex_is_held() {
        let hub = Hub::new();
        let mut guard = hub.state.lock().unwrap();
        *guard = active();
        receipt(&mut guard, "test", json!({"marker": 1}));
        hub.publish(&guard);
        let old = hub.published.load_full();
        guard.tick = 99;
        let reply = hub.read("policy.external.observation", &json!({})).unwrap().unwrap();
        assert_eq!(reply["tick"], 5);
        let page = hub.read("policy.external.receipts", &json!({"after_sequence":0})).unwrap().unwrap();
        assert_eq!(page["events"][0]["marker"], 1);
        assert!(hub.read("policy.external.close", &json!({})).is_none());
        hub.publish(&guard);
        assert_eq!(hub.read("policy.external.status", &json!({})).unwrap().unwrap()["tick"], 99);
        assert_eq!(old.observation["tick"], 5);
    }
    #[test]
    fn stopped_publisher_does_not_freeze_observation_age() {
        let hub = Hub::new();
        let mut state = hub.state.lock().unwrap();
        state.observed = Instant::now() - Duration::from_millis(250);
        hub.publish(&state);
        let result = hub.read("policy.external.observation", &json!({})).unwrap().unwrap();
        assert!(result["observation_age_s"].as_f64().unwrap() >= 0.25);
    }
    #[test]
    fn skipped_control_ticks_do_not_extend_action_lifetime() {
        let mut state = active();
        request(&mut state, "policy.external.push_chunk", &packet()).unwrap();
        assert_eq!(sample_state_at_tick(&mut state, &[0.;61], true, None, 12), Some([0.1;14]));
        assert_eq!(state.last_consumed.as_ref().unwrap()["action_index"], 9);
        assert_eq!(sample_state_at_tick(&mut state, &[0.;61], true, None, 13), None);
        assert_eq!(state.receipts.back().unwrap()["reason"], "expired_tick");
    }
    fn packet() -> Value {
        json!({"episode_id":"episode", "source_tick":3, "control_dt_s":0.02, "actions":vec![vec![0.1;14];50]})
    }
    fn prefix_packet(source_tick: u64) -> Value {
        let actions: Vec<_> = (0..5).map(|index| vec![index as f32 / 10.; 14]).collect();
        json!({"episode_id":"episode", "source_tick":source_tick, "control_dt_s":0.02,
            "execution_mode":PREFIX_MODE, "prediction_horizon":50, "execution_prefix":5, "actions":actions})
    }
    fn prefix_active() -> State { State { execution_mode: PREFIX_MODE, ..active() } }
    #[test]
    fn prefix_mode_is_explicit_and_advertised_without_changing_legacy_default() {
        let mut state = State { observation: vec![0.;61], ..State::default() };
        let supported = json!([LEGACY_MODE, PREFIX_MODE]);
        assert_eq!(observation_reply(&state)["supported_execution_modes"], supported);
        for mode in [json!("unknown"), json!(true), Value::Null] {
            assert_eq!(request(&mut state, "policy.external.open", &json!({"episode_id":"episode", "execution_mode":mode})).unwrap_err(), "unsupported execution_mode");
            assert!(state.episode.is_none());
        }
        let opened = request(&mut state, "policy.external.open", &json!({"episode_id":"episode", "execution_mode":PREFIX_MODE})).unwrap();
        assert_eq!(opened["execution_mode"], PREFIX_MODE);
        assert_eq!(opened["supported_execution_modes"], supported);
        request(&mut state, "policy.external.close", &json!({"episode_id":"episode"})).unwrap();
        let reopened = request(&mut state, "policy.external.open", &json!({"episode_id":"legacy"})).unwrap();
        assert_eq!(reopened["execution_mode"], LEGACY_MODE);
        assert_eq!(state.execution_mode, LEGACY_MODE);
    }
    #[test]
    fn prefix_rows_zero_through_four_are_consumed_once_then_exhausted() {
        let mut state = prefix_active();
        let result = request(&mut state, "policy.external.push_chunk", &prefix_packet(3)).unwrap();
        assert_eq!(result["admission_tick"], 5);
        assert_eq!(result["playback_start_tick"], 6);
        let accepted = state.receipts.back().unwrap();
        assert_eq!(accepted["execution_mode"], PREFIX_MODE);
        assert_eq!(accepted["source_tick"], 3);
        assert_eq!(accepted["valid_until_tick"], 13);
        assert_eq!(accepted["prediction_horizon"], 50);
        assert_eq!(accepted["execution_prefix"], 5);
        assert_eq!(accepted["rows"], 5);
        for index in 0..5 {
            assert_eq!(sample_state(&mut state, &[0.;61], true, None), Some([index as f32 / 10.;14]));
            let row = state.last_consumed.as_ref().unwrap();
            assert_eq!(row["action_index"], index);
            assert_eq!(row["daemon_tick"], 6 + index);
            assert_eq!(row["source_tick"], 3);
            assert_eq!(row["playback_start_tick"], 6);
            assert_eq!(row["execution_mode"], PREFIX_MODE);
            assert_eq!(row["execution_prefix"], 5);
            assert_eq!(row["prediction_horizon"], 50);
        }
        assert!(sample_state(&mut state, &[0.;61], true, None).is_none());
        assert_eq!(state.receipts.back().unwrap()["reason"], "chunk_exhausted");
        assert_eq!(state.external_ticks, 5);
        assert_eq!(state.fallback_ticks, 1);
    }
    #[test]
    fn late_prefix_start_does_not_renew_source_tick_or_wall_expiry() {
        let mut state = State { tick: 11, ..prefix_active() };
        request(&mut state, "policy.external.push_chunk", &prefix_packet(3)).unwrap();
        assert_eq!(state.chunk.as_ref().unwrap().valid_until, 13);
        assert_eq!(state.chunk.as_ref().unwrap().playback_start_tick, 12);
        assert_eq!(sample_state(&mut state, &[0.;61], true, None), Some([0.;14]));
        assert!(sample_state(&mut state, &[0.;61], true, None).is_none());
        assert_eq!(state.receipts.back().unwrap()["reason"], "expired_tick");
        assert_eq!(request(&mut state, "policy.external.push_chunk", &prefix_packet(3)).unwrap_err(), "out_of_order");
        let mut stale = State { tick: 13, ..prefix_active() };
        assert_eq!(request(&mut stale, "policy.external.push_chunk", &prefix_packet(3)).unwrap_err(), "stale_observation");
        let mut wall = prefix_active();
        request(&mut wall, "policy.external.push_chunk", &prefix_packet(3)).unwrap();
        wall.chunk.as_mut().unwrap().received = Instant::now() - MAX_AGE;
        assert!(sample_state(&mut wall, &[0.;61], true, None).is_none());
        assert_eq!(wall.receipts.back().unwrap()["reason"], "expired_wall");
    }
    #[test]
    fn prefix_honors_explicit_source_deadline() {
        let mut state = prefix_active();
        let mut value = prefix_packet(3);
        value["valid_until_tick"] = json!(7);
        request(&mut state, "policy.external.push_chunk", &value).unwrap();
        assert_eq!(sample_state(&mut state, &[0.;61], true, None), Some([0.;14]));
        assert!(sample_state(&mut state, &[0.;61], true, None).is_none());
        assert_eq!(state.receipts.back().unwrap()["reason"], "expired_tick");
    }
    #[test]
    fn prefix_replacement_requires_consumption_and_preserves_the_good_chunk_on_rejection() {
        let mut state = prefix_active();
        request(&mut state, "policy.external.push_chunk", &prefix_packet(3)).unwrap();
        assert_eq!(request(&mut state, "policy.external.push_chunk", &prefix_packet(4)).unwrap_err(), "prefix_in_progress");
        assert_eq!(state.last_source, Some(3));
        assert_eq!(state.chunk.as_ref().unwrap().source_tick, 3);
        for _ in 0..5 { assert!(sample_state(&mut state, &[0.;61], true, None).is_some()); }
        let reply = request(&mut state, "policy.external.push_chunk", &prefix_packet(10)).unwrap();
        assert_eq!(reply["playback_start_tick"], 11);
        assert_eq!(sample_state(&mut state, &[0.;61], true, None), Some([0.;14]));
        assert_eq!(state.fallback_ticks, 0);
    }
    #[test]
    fn skipped_prefix_ticks_do_not_replay_rows_or_count_as_full_consumption() {
        let mut state = prefix_active();
        request(&mut state, "policy.external.push_chunk", &prefix_packet(3)).unwrap();
        assert_eq!(sample_state_at_tick(&mut state, &[0.;61], true, None, 6), Some([0.;14]));
        assert_eq!(sample_state_at_tick(&mut state, &[0.;61], true, None, 9), Some([0.3;14]));
        assert_eq!(sample_state_at_tick(&mut state, &[0.;61], true, None, 10), Some([0.4;14]));
        assert_eq!(state.last_consumed.as_ref().unwrap()["playback_start_tick"], 6);
        assert_eq!(state.last_consumed.as_ref().unwrap()["action_index"], 4);
        assert_eq!(request(&mut state, "policy.external.push_chunk", &prefix_packet(10)).unwrap_err(), "prefix_in_progress");
        assert!(sample_state(&mut state, &[0.;61], true, None).is_none());
        assert_eq!(state.receipts.back().unwrap()["reason"], "chunk_exhausted");
        request(&mut state, "policy.external.push_chunk", &prefix_packet(11)).unwrap();
        assert_eq!(state.fallback_ticks, 1);
    }
    #[test]
    fn expired_prefix_can_be_replaced_without_clearing_recorded_fallback() {
        let mut state = prefix_active();
        request(&mut state, "policy.external.push_chunk", &prefix_packet(3)).unwrap();
        assert!(sample_state_at_tick(&mut state, &[0.;61], true, None, 13).is_none());
        assert_eq!(state.receipts.back().unwrap()["reason"], "expired_tick");
        request(&mut state, "policy.external.push_chunk", &prefix_packet(13)).unwrap();
        assert_eq!(sample_state(&mut state, &[0.;61], true, None), Some([0.;14]));
        assert_eq!(state.fallback_ticks, 1);
    }
    #[test]
    fn malformed_prefix_metadata_or_unnegotiated_mode_is_rejected() {
        let state = prefix_active();
        for field in ["prediction_horizon", "execution_prefix", "execution_mode"] {
            let mut missing = prefix_packet(3);
            missing.as_object_mut().unwrap().remove(field);
            assert!(parse_chunk(&state, &missing).is_err());
            for invalid in [Value::Null, json!(true), json!("5"), json!(0), json!(5.0)] {
                let mut packet = prefix_packet(3);
                packet[field] = invalid;
                assert!(parse_chunk(&state, &packet).is_err());
            }
        }
        let mut mismatch = prefix_packet(3);
        mismatch["execution_mode"] = json!(LEGACY_MODE);
        assert!(parse_chunk(&state, &mismatch).is_err());
        assert!(parse_chunk(&active(), &prefix_packet(3)).is_err());
        for count in [1, 4, 6, 50] {
            let mut wrong_rows = prefix_packet(3);
            wrong_rows["actions"] = json!(vec![vec![0.;14];count]);
            assert!(parse_chunk(&state, &wrong_rows).is_err());
        }
    }
    #[test]
    fn prefix_capture_identity_and_age_remain_bound_to_original_observation() {
        let mut state = prefix_active();
        let origin = capture(100);
        state.source_capture = Some(origin.clone());
        assert_eq!(request(&mut state, "policy.external.push_chunk", &prefix_packet(3)).unwrap_err(), "unknown_source_capture");
        let mut wrong = origin.clone(); wrong.episode_id = "other-episode".into();
        state.observation_sources.push_back((3, Some(wrong)));
        assert_eq!(request(&mut state, "policy.external.push_chunk", &prefix_packet(3)).unwrap_err(), "source_capture_episode_mismatch");
        state.observation_sources.clear();
        state.observation_sources.push_back((3, Some(origin.clone())));
        request(&mut state, "policy.external.push_chunk", &prefix_packet(3)).unwrap();
        assert_eq!(sample_state(&mut state, &[0.;61], true, Some(capture(104))), Some([0.;14]));
        let row = state.last_consumed.as_ref().unwrap();
        assert_eq!(row["source_tick"], 3);
        assert_eq!(row["action_index"], 0);
        assert_eq!(row["source_capture"]["body_tick"], 100);
        assert_eq!(row["source_capture"]["capture_monotonic_ns"], origin.capture_monotonic_ns);
        assert_eq!(row["control_capture"]["body_tick"], 104);
        state.chunk.as_mut().unwrap().source_capture.as_mut().unwrap().capture_monotonic_ns = 0;
        assert!(sample_state(&mut state, &[0.;61], true, Some(capture(105))).is_none());
        assert_eq!(state.receipts.back().unwrap()["reason"], "stale_capture");
    }
    #[test]
    fn prefix_session_startup_without_a_chunk_still_counts_fallback() {
        let mut state = prefix_active();
        assert!(sample_state(&mut state, &[0.;61], true, None).is_none());
        assert_eq!(state.fallback_ticks, 1);
        assert_eq!(state.receipts.back().unwrap()["reason"], "no_chunk");
        assert_eq!(state.receipts.back().unwrap()["execution_mode"], PREFIX_MODE);
    }
    #[test]
    fn delayed_actions_remain_anchored_to_observation_and_ttl() {
        let state = active();
        let chunk = parse_chunk(&state, &packet()).unwrap();
        assert_eq!(chunk.source_tick, 3);
        assert_eq!(chunk.valid_until, 13);
    }
    #[test]
    fn old_episodes_and_out_of_order_requests_are_refused() {
        let mut state = active();
        let mut p = packet(); p["episode_id"] = json!("old");
        assert!(parse_chunk(&state, &p).is_err());
        state.last_source = Some(3);
        assert!(parse_chunk(&state, &packet()).is_err());
    }
    #[test]
    fn nonfinite_wrong_shape_and_stale_packets_are_refused() {
        let mut state = active();
        let mut p = packet(); p["actions"][0][0] = Value::Null;
        assert!(parse_chunk(&state, &p).is_err());
        state.tick = 14;
        assert!(parse_chunk(&state, &packet()).is_err());
    }
    #[test]
    fn close_drops_buffer_and_requires_explicit_reopen() {
        let mut state = active();
        request(&mut state,"policy.external.push_chunk",&packet()).unwrap();
        request(&mut state,"policy.external.close",&json!({"episode_id":"episode"})).unwrap();
        assert!(state.chunk.is_none());
        assert!(parse_chunk(&state,&packet()).is_err());
    }
    #[test]
    fn snapshot_exposes_explicit_sensor_identity_and_control_permission() {
        let mut state = active();
        state.permitted = true;
        state.source_capture = Some(duck_control::io::RemoteCapture {
            protocol: 1, body_tick: 100, task_tick: 3, task_epoch: 2,
            episode_id: "episode".into(), capture_monotonic_ns: 123456,
            command13: [0.; 13], command_program_sha256: Some("program".into()),
        });
        let result = request(&mut state, "policy.external.observation", &json!({})).unwrap();
        assert_eq!(result["tick"], 5);
        assert_eq!(result["source_capture"]["body_tick"], 100);
        assert_eq!(result["source_capture"]["task_tick"], 3);
        assert_eq!(result["external_control_permitted"], true);
    }
    fn capture(body_tick: u64) -> duck_control::io::RemoteCapture {
        duck_control::io::RemoteCapture { protocol: 1, body_tick, task_tick: body_tick,
            task_epoch: 1, episode_id: "episode".into(), capture_monotonic_ns: duck_ipc_proto::clock::monotonic_ns(),
            command13: [0.;13], command_program_sha256: Some("program".into()) }
    }
    #[test]
    fn receipt_binds_consumption_to_origin_capture_not_newest_capture() {
        let mut state = active();
        let origin = capture(100);
        state.source_capture = Some(origin.clone());
        state.observation_sources.push_back((3, Some(origin.clone())));
        request(&mut state, "policy.external.push_chunk", &packet()).unwrap();
        let selected = sample_state(&mut state, &[0.;61], true, Some(capture(104)));
        assert_eq!(selected, Some([0.1;14]));
        let row = state.last_consumed.as_ref().unwrap();
        assert_eq!(row["source_tick"], 3);
        assert_eq!(row["daemon_tick"], 6);
        assert_eq!(row["action_index"], 3);
        assert_eq!(row["source_capture"]["body_tick"], 100);
        assert_eq!(row["source_capture"]["capture_monotonic_ns"], origin.capture_monotonic_ns);
        assert_eq!(row["control_capture"]["body_tick"], 104);
        assert!(row["monotonic_ns"].as_u64().unwrap() >= origin.capture_monotonic_ns);
    }
    #[test]
    fn receipt_records_expiry_and_cancel_rejects_late_packets_and_id_reuse() {
        let mut state = active();
        let mut value = packet(); value["valid_until_tick"] = json!(6);
        request(&mut state, "policy.external.push_chunk", &value).unwrap();
        assert!(sample_state(&mut state, &[0.;61], true, None).is_none());
        assert_eq!(state.receipts.back().unwrap()["reason"], "expired_tick");
        assert_eq!(state.receipts.back().unwrap()["execution_source"], "official_fallback");
        request(&mut state, "policy.external.close", &json!({"episode_id":"episode"})).unwrap();
        assert_eq!(request(&mut state, "policy.external.push_chunk", &packet()).unwrap_err(), "wrong_episode");
        assert_eq!(state.receipts.back().unwrap()["result"], "rejected");
        assert!(request(&mut state, "policy.external.open", &json!({"episode_id":"episode"})).unwrap_err().contains("retired_episode"));
        request(&mut state, "policy.external.open", &json!({"episode_id":"new"})).unwrap();
        assert_eq!(request(&mut state, "policy.external.push_chunk", &packet()).unwrap_err(), "wrong_episode");
        assert!(state.chunk.is_none());
    }
    #[test]
    fn stale_capture_is_checked_at_consumption_and_missing_identity_rejected() {
        let mut state = active();
        state.source_capture = Some(capture(100));
        assert_eq!(request(&mut state, "policy.external.push_chunk", &packet()).unwrap_err(), "unknown_source_capture");
        state.observation_sources.push_back((3, Some(capture(100))));
        request(&mut state, "policy.external.push_chunk", &packet()).unwrap();
        state.chunk.as_mut().unwrap().source_capture.as_mut().unwrap().capture_monotonic_ns = 0;
        assert!(sample_state(&mut state, &[0.;61], true, Some(capture(101))).is_none());
        assert_eq!(state.receipts.back().unwrap()["reason"], "stale_capture");
    }
    #[test]
    fn bounded_receipt_pages_report_overflow_instead_of_claiming_complete_history() {
        let mut state = active();
        for _ in 0..RECEIPT_CAPACITY + 3 { receipt(&mut state, "test", json!({})); }
        let page = request(&mut state, "policy.external.receipts", &json!({"after_sequence":0,"limit":1000})).unwrap();
        assert_eq!(page["missing_events"], 3);
        assert_eq!(page["oldest_sequence"], 4);
        assert_eq!(page["events"].as_array().unwrap().len(), 16);
        let next = request(&mut state, "policy.external.receipts", &json!({"after_sequence":page["next_sequence"]})).unwrap();
        assert_eq!(next["missing_events"], 0);
    }
}
