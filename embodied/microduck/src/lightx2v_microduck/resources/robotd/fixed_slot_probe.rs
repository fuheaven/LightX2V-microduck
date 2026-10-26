//! Isolated CPU transport diagnostic. This is not robotd's controller or a VLA.
use std::{error::Error, fs, thread, time::{Duration, Instant}};
use duck_control::{io::{RemoteExecution, RemoteSlot}, safety::{Safety, SafetyConfig, Limit}, sim::RemoteIo};
use serde::Deserialize;
use serde_json::json;

const MODE: &str = "fixed_body_slot_probe_v1";

#[derive(Deserialize)]
struct Input {
    address: String,
    case: String,
    targets: Vec<[f64; 15]>,
    raw_actions: Vec<[f32; 14]>,
    running_gain: u16,
}

fn ns() -> u64 {
    let mut value = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    let result = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut value) };
    assert_eq!(result, 0);
    value.tv_sec as u64 * 1_000_000_000 + value.tv_nsec as u64
}

fn run(input: Input, output: &str) -> Result<(), Box<dyn Error>> {
    assert!(matches!(input.case.as_str(), "nominal" | "clamp" | "nonfinite" | "cancel" | "late"));
    let count = if input.case == "nominal" {15} else {1};
    assert_eq!(input.targets.len(), count);
    assert_eq!(input.raw_actions.len(), count);
    let mut safety = Safety::new(RemoteIo::at(input.address), SafetyConfig::default());
    let mut rows=Vec::new();
    let started=Instant::now();
    let result = (|| -> Result<(), Box<dyn Error>> {
        safety.set_torque(true)?;
        let mut capture=None;
        for sequence in 0..count {
            let sensors=safety.read()?;
            if sequence % 5 == 0 { capture=safety.source_capture(); }
            let source=capture.clone().ok_or("body did not publish a source capture")?;
            let status=safety.fixed_slot_status()?;
            assert_eq!(status.protocol, MODE);
            assert_eq!(status.period_ns, 20_000_000);
            let slot=RemoteSlot { protocol:MODE.into(), generation:status.generation,
                intended_body_tick:status.next_body_tick, valid_until_body_tick:source.body_tick+10 };
            let execution=RemoteExecution { protocol:1, daemon_tick:sequence as u64+1,
                episode_id:source.episode_id.clone(), execution_mode:MODE.into(), execution_source:"external".into(),
                source_tick:source.body_tick, action_index:(sequence%5) as u64, raw_action14:input.raw_actions[sequence],
                selected_monotonic_ns:ns(), source_capture:Some(source.clone()) };
            let mut targets=input.targets[sequence];
            if input.case == "clamp" { targets[0]=100.; }
            if input.case == "nonfinite" { targets[0]=f64::NAN; }
            if input.case == "late" {
                while ns() <= status.integration_deadline_ns { thread::sleep(Duration::from_micros(50)); }
            }
            let before=ns();
            let applied=safety.apply_with_slot(targets,sensors.positions,input.running_gain,Some(execution.clone()),Some(slot.clone()));
            let after=ns();
            match applied {
                Ok(applied) => {
                    if input.case == "late" { return Err("late target was accepted".into()); }
                    rows.push(json!({"sequence":sequence+1,"source":source,"slot":slot,
                        "execution":execution,"hold_targets":sensors.positions,
                        "range_limited":applied.limited_by(Limit::Range),
                        "nonfinite_limited":applied.limited_by(Limit::NotFinite),
                        "apply_start_ns":before,"apply_ack_ns":after}));
                    if input.case == "cancel" { safety.invalidate_fixed_slots(status.generation)?; }
                    loop {
                        let current=safety.fixed_slot_status()?;
                        if current.body_tick >= status.next_body_tick { break; }
                        if started.elapsed() > Duration::from_secs(3) { return Err("probe deadline exceeded".into()); }
                        thread::sleep(Duration::from_micros(250));
                    }
                }
                Err(error) if input.case == "late" => {
                    rows.push(json!({"sequence":sequence+1,"expected_rejection":error.to_string(),
                                    "source":source,"slot":slot,"apply_start_ns":before,"apply_ack_ns":after}));
                }
                Err(error) => return Err(error.into()),
            }
        }
        let final_status=safety.fixed_slot_status()?;
        safety.invalidate_fixed_slots(final_status.generation)?;
        Ok(())
    })();
    // A failed diagnostic also retires pending writes where communication works.
    if result.is_err() {
        if let Ok(status)=safety.fixed_slot_status() { let _=safety.invalidate_fixed_slots(status.generation); }
    }
    fs::write(output, serde_json::to_vec_pretty(&json!({"protocol":"rust_safety_fixed_slot_probe_v1",
        "case":input.case,"status":if result.is_ok(){"completed"}else{"failed"},
        "error":result.as_ref().err().map(|e|e.to_string()),"rows":rows,
        "elapsed_seconds":started.elapsed().as_secs_f64(),"robotd_controller_used":false,
        "model_used":false,"gpu_used":false,"counts_as_skill_success":false}))?)?;
    result
}

fn main() -> Result<(), Box<dyn Error>> {
    let args:Vec<_>=std::env::args().collect();
    if args.len()!=3 { return Err("usage: fixed_slot_probe input.json output.json".into()); }
    run(serde_json::from_slice(&fs::read(&args[1])?)?, &args[2])
}
