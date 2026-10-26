//! CPU diagnostic of actual Safety/RemoteIo owner lease RPCs, not a policy controller.
use std::{error::Error, fs, thread, time::{Duration, Instant}};
use duck_control::{io::{RemoteExecution, RemoteOwner, RemoteSlot},
    safety::{Safety, SafetyConfig, Limit}, sim::RemoteIo};
use serde::Deserialize;
use serde_json::{json, Value};

const MODE: &str = "fixed_body_slot_probe_v1";
const PERIOD: u64 = 20_000_000;

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
    assert_eq!(unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut value) }, 0);
    value.tv_sec as u64 * 1_000_000_000 + value.tv_nsec as u64
}

fn sleep_until(deadline: u64) {
    loop {
        let now = ns();
        if now >= deadline { break; }
        thread::sleep(Duration::from_nanos((deadline-now).min(250_000)));
    }
}

fn public_owner(owner: &RemoteOwner) -> Value {
    json!({"protocol":owner.protocol,"generation":owner.generation,
        "lease_ns":owner.lease_ns,"lease_deadline_ns":owner.lease_deadline_ns})
}

fn public_slot(slot: &RemoteSlot) -> Value {
    json!({"protocol":slot.protocol,"generation":slot.generation,
        "intended_body_tick":slot.intended_body_tick,
        "valid_until_body_tick":slot.valid_until_body_tick,"owner_token_present":slot.owner_token.is_some()})
}

fn run(input: Input, output: &str) -> Result<(), Box<dyn Error>> {
    assert!(matches!(input.case.as_str(), "nominal" | "clamp" | "nonfinite" | "cancel" | "late" | "expiry" | "retired_owner"));
    let count = match input.case.as_str() { "nominal"=>15, "retired_owner"=>2, _=>1 };
    assert_eq!(input.targets.len(), count);
    assert_eq!(input.raw_actions.len(), count);
    let mut safety = Safety::new(RemoteIo::at(input.address), SafetyConfig::default());
    let mut rows=Vec::new();
    let mut lifecycle=Vec::new();
    let mut active_owner:Option<RemoteOwner>=None;
    let mut quiet_window=None;
    let started=Instant::now();
    let result = (|| -> Result<(), Box<dyn Error>> {
        safety.set_torque(true)?;
        let initial=safety.fixed_slot_status()?;
        assert_eq!(initial.period_ns,PERIOD);
        sleep_until(initial.integration_deadline_ns+5_000_000);
        let mut owner=safety.open_fixed_slot_owner(&initial.episode_id,initial.task_epoch)?;
        assert_eq!(owner.protocol,"fixed_slot_owner_lease_probe_v1");
        assert_eq!(owner.lease_ns,150_000_000);
        lifecycle.push(json!({"op":"open","owner":public_owner(&owner),"ack_ns":ns()}));
        active_owner=Some(owner.clone());
        let mut retired=None;
        let mut capture=None;
        let mut heartbeat_sequence=0;
        for sequence in 0..count {
            if sequence>0 && sequence%3==0 {
                heartbeat_sequence+=1;
                let renewed=safety.heartbeat_fixed_slot_owner(&owner,heartbeat_sequence)?;
                assert_eq!(renewed.generation,owner.generation);
                lifecycle.push(json!({"op":"heartbeat","sequence":heartbeat_sequence,
                    "generation":renewed.generation,"lease_deadline_ns":renewed.lease_deadline_ns,"ack_ns":ns()}));
            }
            if input.case=="expiry" {
                let status=safety.fixed_slot_status()?;
                let origin=status.integration_deadline_ns-(status.next_body_tick-1)*PERIOD;
                let target=(owner.lease_deadline_ns-origin+PERIOD-1)/PERIOD+1;
                loop {
                    if safety.fixed_slot_status()?.body_tick>=target-1 {break;}
                    if started.elapsed()>Duration::from_secs(3) {return Err("expiry wait timed out".into());}
                    thread::sleep(Duration::from_micros(250));
                }
            }
            let sensors=safety.read()?;
            if sequence%5==0 || input.case=="retired_owner" {capture=safety.source_capture();}
            let source=capture.clone().ok_or("body did not publish source capture")?;
            let status=safety.fixed_slot_status()?;
            assert_eq!(status.protocol,MODE);
            assert_eq!(status.generation,owner.generation);
            let slot=RemoteSlot {owner_token:Some(owner.owner_token.clone()),protocol:MODE.into(),
                generation:owner.generation,intended_body_tick:status.next_body_tick,
                valid_until_body_tick:source.body_tick+10};
            let index=if input.case=="retired_owner" {0} else {sequence%5};
            let execution=RemoteExecution {protocol:1,daemon_tick:sequence as u64+1,
                episode_id:source.episode_id.clone(),execution_mode:MODE.into(),execution_source:"external".into(),
                source_tick:source.body_tick,action_index:index as u64,raw_action14:input.raw_actions[sequence],
                selected_monotonic_ns:ns(),source_capture:Some(source.clone())};
            let mut targets=input.targets[sequence];
            if input.case=="clamp" {targets[0]=100.;}
            if input.case=="nonfinite" {targets[0]=f64::NAN;}
            if input.case=="late" {sleep_until(status.integration_deadline_ns+100_000);}
            let before=ns();
            let applied=safety.apply_with_slot(targets,sensors.positions,input.running_gain,
                Some(execution.clone()),Some(slot.clone()));
            let after=ns();
            match applied {
                Ok(applied)=>{
                    if input.case=="late" {return Err("late slot was accepted".into());}
                    rows.push(json!({"sequence":sequence+1,"source":source,"slot":public_slot(&slot),
                        "execution":execution,"hold_targets":sensors.positions,
                        "range_limited":applied.limited_by(Limit::Range),
                        "nonfinite_limited":applied.limited_by(Limit::NotFinite),
                        "apply_start_ns":before,"apply_ack_ns":after}));
                    if input.case=="retired_owner" && sequence==0 {
                        let generation=safety.close_fixed_slot_owner(&owner)?;
                        lifecycle.push(json!({"op":"close","generation":owner.generation,"next_generation":generation,"ack_ns":ns()}));
                        retired=Some((owner.clone(),execution,slot,targets,sensors.positions));
                        owner=safety.open_fixed_slot_owner(&initial.episode_id,initial.task_epoch)?;
                        lifecycle.push(json!({"op":"open","owner":public_owner(&owner),"ack_ns":ns()}));
                        active_owner=Some(owner.clone());
                        continue;
                    }
                    if let Some((old_owner,old_execution,old_slot,old_targets,old_hold))=retired.take() {
                        let rejected=safety.apply_with_slot(old_targets,old_hold,input.running_gain,
                            Some(old_execution),Some(old_slot)).err().ok_or("retired write accepted")?;
                        let close_rejected=safety.close_fixed_slot_owner(&old_owner).err().ok_or("retired close accepted")?;
                        lifecycle.push(json!({"op":"retired_rejections","write":rejected.to_string(),
                            "close":close_rejected.to_string(),"generation":old_owner.generation,"ack_ns":ns()}));
                    }
                    if input.case=="cancel" {
                        let generation=safety.close_fixed_slot_owner(&owner)?;
                        lifecycle.push(json!({"op":"close","generation":owner.generation,"next_generation":generation,"ack_ns":ns()}));
                        active_owner=None;
                    }
                    if input.case=="expiry" {
                        let start=ns();
                        sleep_until(status.integration_deadline_ns+PERIOD+5_000_000);
                        quiet_window=Some(json!({"start_ns":start,"end_ns":ns(),
                            "lease_deadline_ns":owner.lease_deadline_ns,"target_body_tick":slot.intended_body_tick}));
                    }
                    loop {
                        if safety.fixed_slot_status()?.body_tick>=status.next_body_tick {break;}
                        if started.elapsed()>Duration::from_secs(3) {return Err("probe deadline exceeded".into());}
                        thread::sleep(Duration::from_micros(250));
                    }
                }
                Err(error) if input.case=="late"=>{
                    rows.push(json!({"sequence":sequence+1,"expected_rejection":error.to_string(),
                        "source":source,"slot":public_slot(&slot),"apply_start_ns":before,"apply_ack_ns":after}));
                }
                Err(error)=>return Err(error.into()),
            }
        }
        if let Some(owner)=active_owner.take() {
            match safety.close_fixed_slot_owner(&owner) {
                Ok(generation)=>lifecycle.push(json!({"op":"close","generation":owner.generation,"next_generation":generation,"ack_ns":ns()})),
                Err(error) if input.case=="expiry" || input.case=="late"=>lifecycle.push(json!({"op":"expired_close_rejected","error":error.to_string(),"ack_ns":ns()})),
                Err(error)=>return Err(error.into()),
            }
        }
        Ok(())
    })();
    if let Some(owner)=active_owner {let _=safety.close_fixed_slot_owner(&owner);}
    fs::write(output,serde_json::to_vec_pretty(&json!({"protocol":"rust_safety_owner_lease_probe_v1",
        "case":input.case,"status":if result.is_ok(){"completed"}else{"failed"},
        "error":result.as_ref().err().map(|e|e.to_string()),"rows":rows,"lifecycle":lifecycle,
        "quiet_window":quiet_window,"elapsed_seconds":started.elapsed().as_secs_f64(),
        "robotd_controller_used":false,"model_used":false,"gpu_used":false,"counts_as_skill_success":false}))?)?;
    result
}

fn main()->Result<(),Box<dyn Error>> {
    let args:Vec<_>=std::env::args().collect();
    if args.len()!=3 {return Err("usage: lease_slot_probe input.json output.json".into());}
    run(serde_json::from_slice(&fs::read(&args[1])?)?,&args[2])
}
