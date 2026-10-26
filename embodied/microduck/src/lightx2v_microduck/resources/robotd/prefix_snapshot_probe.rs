//! Actual Safety/RemoteIo post-prefix reads. This is not robotd's policy loop.
use std::{error::Error, fs, thread, time::{Duration, Instant}};
use duck_control::{io::{RemoteCapture, RemoteExecution, RemoteOwner, RemoteSlot},
    safety::{Safety, SafetyConfig, Limit}, sim::RemoteIo};
use serde::Deserialize;
use serde_json::{json, Value};

const MODE: &str = "fixed_body_slot_probe_v1";
#[derive(Deserialize)]
struct Input { address: String, case: String, targets: Vec<[f64;15]>, raw_actions: Vec<[f32;14]> }

fn ns() -> u64 {
    let mut t=libc::timespec{tv_sec:0,tv_nsec:0};
    assert_eq!(unsafe {libc::clock_gettime(libc::CLOCK_MONOTONIC,&mut t)},0);
    t.tv_sec as u64*1_000_000_000+t.tv_nsec as u64
}
fn sleep_until(deadline:u64) {
    loop {
        let now=ns();if now>=deadline {break;}
        thread::sleep(Duration::from_nanos((deadline-now).min(250_000)));
    }
}
fn ready_source(reply:&Value, completed:u64, previous:&RemoteCapture, raw:&[f32;14], after:u64, require_rgb:bool) -> Result<RemoteCapture,Box<dyn Error>> {
    if reply["protocol"]!="leased_post_prefix_snapshot_v1" || reply["status"]!="ready" || reply["eligible"]!=true
        || reply["capture_phase"]!="post_integration" || reply["missing_snapshots"]!=0 || reply["require_rgb"].as_bool()!=Some(require_rgb)
        || reply["snapshot_sequence"].as_u64()!=Some(after+1)
        || reply["next_body_tick"].as_u64()!=Some(completed+1) {return Err("invalid post-prefix identity".into());}
    let source:RemoteCapture=serde_json::from_value(reply["source_capture"].clone())?;
    if require_rgb && (reply["render_status"]!="ready" || reply["rgb"]["encoding"]!="base64_uint8_rgb"
        || reply["rgb"]["shape"]!=json!([256,256,3])
        || reply["rgb"]["data"].as_str().map_or(true,|x|x.len()!=256*256*4)
        || reply["render_ready_monotonic_ns"].as_u64().map_or(true,|x|x<source.capture_monotonic_ns)) {
        return Err("missing rendered snapshot payload".into());
    }
    let state=reply["state61"].as_array().ok_or("missing state61")?;
    if source.protocol!=1 || source.body_tick!=completed || completed!=previous.body_tick+5
        || source.task_tick!=previous.task_tick+5 || source.episode_id!=previous.episode_id
        || source.task_epoch!=previous.task_epoch || source.command_program_sha256!=previous.command_program_sha256
        || source.capture_monotonic_ns<=previous.capture_monotonic_ns || state.len()!=61
        || state.iter().any(|x| x.as_f64().map_or(true,|x|!x.is_finite())) {return Err("invalid post-prefix state or epoch".into());}
    for i in 0..14 {if state[34+i].as_f64().unwrap() as f32!=raw[i] {return Err("history is not integrated row four".into());}}
    for i in 0..13 {if state[48+i].as_f64().unwrap() as f32!=source.command13[i] {return Err("command does not match capture".into());}}
    let integrations=reply["prefix_integrations"].as_array().ok_or("missing prefix evidence")?;
    if integrations.len()!=5 {return Err("missing physical rows".into());}
    let mut sequences=std::collections::HashSet::new();
    for (i,row) in integrations.iter().enumerate() {
        let end=row["integration_end_monotonic_ns"].as_u64().ok_or("missing integration timestamp")?;
        if row["body_tick"].as_u64()!=Some(completed-4+i as u64) || end>source.capture_monotonic_ns
            || !sequences.insert(row["io_write_sequence"].as_u64().ok_or("missing IO sequence")?) {return Err("invalid prefix order".into());}
    }
    let deadline=reply["next_integration_deadline_ns"].as_u64().ok_or("missing deadline")?;
    let now=ns();
    if now<source.capture_monotonic_ns || now>=deadline || now-source.capture_monotonic_ns>=200_000_000 {
        return Err("post-prefix observation arrived late".into());
    }
    Ok(source)
}

fn run(input:Input,output:&str)->Result<(),Box<dyn Error>> {
    if !matches!(input.case.as_str(),"state"|"cancel"|"future_cursor"|"rendered") {return Err("unknown case".into());}
    let count=if input.case=="state" {15}else{5};
    assert_eq!(input.targets.len(),count);assert_eq!(input.raw_actions.len(),count);
    let mut safety=Safety::new(RemoteIo::at(input.address),SafetyConfig::default());
    let mut owner:Option<RemoteOwner>=None;
    let mut rows=Vec::new();let mut observations=Vec::new();let mut preparations=Vec::new();let mut lifecycle=Vec::new();
    let started=Instant::now();
    let result=(||->Result<(),Box<dyn Error>>{
        safety.set_torque(true)?;
        if input.case=="rendered" {
            loop {
                let reply=safety.prepare_snapshot_renderer()?;
                if reply["observation_eligible"]!=false {return Err("preparation became an observation".into());}
                let ready=reply["status"]=="ready";
                if !ready && reply["status"]!="pending" && reply["status"]!="warming" {return Err("preparation failed".into());}
                preparations.push(json!({"reply":reply,"received_ns":ns()}));
                if ready {break;}
                if started.elapsed()>Duration::from_secs(5) {return Err("preparation timeout".into());}
                thread::sleep(Duration::from_millis(2));
            }
        }
        let initial=safety.fixed_slot_status()?;
        if initial.period_ns!=20_000_000 {return Err("incorrect slot period".into());}
        sleep_until(initial.integration_deadline_ns+4_000_000);
        let current=safety.open_fixed_slot_owner(&initial.episode_id,initial.task_epoch)?;
        lifecycle.push(json!({"op":"open","generation":current.generation,"ack_ns":ns()}));
        owner=Some(current.clone());
        safety.read()?;
        let mut capture=safety.source_capture().ok_or("missing initial source")?;
        let mut heartbeat=0;
        for seq in 0..count {
            if seq>0 && seq%3==0 {
                heartbeat+=1;let renewed=safety.heartbeat_fixed_slot_owner(&current,heartbeat)?;
                lifecycle.push(json!({"op":"heartbeat","sequence":heartbeat,"generation":renewed.generation,"ack_ns":ns()}));
            }
            let sensors=safety.read()?; // Safety hold state; never replaces the completed-prefix source.
            let status=safety.fixed_slot_status()?;
            let index=seq%5;
            if status.next_body_tick!=capture.body_tick+index as u64+1 {return Err("unexpected physical gap".into());}
            let slot=RemoteSlot{owner_token:Some(current.owner_token.clone()),protocol:MODE.into(),generation:current.generation,
                intended_body_tick:status.next_body_tick,valid_until_body_tick:capture.body_tick+10};
            let execution=RemoteExecution{protocol:1,daemon_tick:seq as u64+1,episode_id:capture.episode_id.clone(),
                execution_mode:MODE.into(),execution_source:"external".into(),source_tick:capture.body_tick,
                action_index:index as u64,raw_action14:input.raw_actions[seq],selected_monotonic_ns:ns(),source_capture:Some(capture.clone())};
            let before=ns();
            let applied=safety.apply_with_slot(input.targets[seq],sensors.positions,200,Some(execution.clone()),Some(slot))?;
            if applied.limited_by(Limit::Range)||applied.limited_by(Limit::NotFinite) {return Err("unexpected Safety limit".into());}
            rows.push(json!({"source":capture,"execution":execution,"intended_body_tick":status.next_body_tick,"apply_start_ns":before,"apply_ack_ns":ns()}));
            while safety.fixed_slot_status()?.body_tick<status.next_body_tick {
                if started.elapsed()>Duration::from_secs(7) {return Err("physics timeout".into());}
                thread::sleep(Duration::from_micros(250));
            }
            if index==4 {
                let after=(seq/5) as u64;
                if input.case=="cancel" {
                    safety.close_fixed_slot_owner(&current)?;owner=None;
                    let error=safety.post_prefix_snapshot(&current,after,false).err().ok_or("cancelled owner read accepted")?;
                    observations.push(json!({"expected_rejection":error.to_string(),"received_ns":ns()}));
                    break;
                }
                if input.case=="future_cursor" {
                    let error=safety.post_prefix_snapshot(&current,after+2,false).err().ok_or("future cursor accepted")?;
                    observations.push(json!({"expected_rejection":error.to_string(),"received_ns":ns()}));
                }
                let require_rgb=input.case=="rendered";
                loop {
                    let before=ns();
                    let response=safety.post_prefix_snapshot(&current,after,require_rgb);
                    let received=ns();
                    match response {
                        Ok(reply)=>{
                            let eligible=reply["eligible"]==true;
                            let source=if eligible {Some(ready_source(&reply,status.next_body_tick,&capture,&input.raw_actions[seq],after,require_rgb)?)}else{None};
                            let pending=reply["status"]=="image_pending"||reply["status"]=="waiting";
                            observations.push(json!({"reply":reply,"require_rgb":require_rgb,"start_ns":before,"received_ns":received}));
                            if let Some(source)=source {capture=source;break;}
                            if !require_rgb {return Err("state-only prefix was not ready".into());}
                            if !pending {break;}
                        }
                        Err(error) if require_rgb=>{observations.push(json!({"rejection":error.to_string(),"start_ns":before,"received_ns":received}));break;}
                        Err(error)=>return Err(error.into()),
                    }
                    if started.elapsed()>Duration::from_secs(7) {return Err("snapshot timeout".into());}
                    thread::sleep(Duration::from_micros(250));
                }
            }
        }
        if let Some(current)=owner.take() {
            match safety.close_fixed_slot_owner(&current) {
                Ok(_)=>lifecycle.push(json!({"op":"close","generation":current.generation,"ack_ns":ns()})),
                Err(error) if input.case=="rendered"=>lifecycle.push(json!({"op":"expired_close","error":error.to_string(),"ack_ns":ns()})),
                Err(error)=>return Err(error.into()),
            }
        }
        Ok(())
    })();
    if let Some(current)=owner {let _=safety.close_fixed_slot_owner(&current);}
    fs::write(output,serde_json::to_vec_pretty(&json!({"protocol":"rust_post_prefix_snapshot_probe_v1","case":input.case,
        "status":if result.is_ok(){"completed"}else{"failed"},"error":result.as_ref().err().map(|e|e.to_string()),
        "rows":rows,"observations":observations,"preparations":preparations,"lifecycle":lifecycle,
        "elapsed_seconds":started.elapsed().as_secs_f64(),"model_used":false,"gpu_used":false,
        "robotd_controller_used":false,"handoff_verified":false,"counts_as_skill_success":false}))?)?;
    result
}
fn main()->Result<(),Box<dyn Error>> {
    let args:Vec<_>=std::env::args().collect();
    if args.len()!=3 {return Err("usage: prefix_snapshot_probe input.json output.json".into());}
    run(serde_json::from_slice(&fs::read(&args[1])?)?,&args[2])
}
