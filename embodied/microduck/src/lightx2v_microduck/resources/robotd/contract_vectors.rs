//! Golden vectors call the pinned upstream Rust APIs; no policy math copied.
use duck_control::{ImuData, Observation, DEFAULT_POSITION};
use duck_control::obs::{BodyPose, Command};
use duck_control::policy::{Net, Policy, PolicyPaths};
use serde_json::json;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(std::env::args().nth(1).ok_or("policy directory required")?);
    let count: usize = std::env::args().nth(2).unwrap_or_else(|| "32".into()).parse()?;
    if count == 0 { return Err("vector count must be positive".into()); }
    let names = ["alpha_walking.onnx", "alpha_stand.onnx", "alpha_sitstand.onnx", "alpha_ground_pick.onnx",
                 "ball_kick_left.onnx", "ball_kick_right.onnx", "roulade.onnx", "roller.onnx", "roller_crouch.onnx"];
    for name in names {
        // Each file occupies the mandatory Walk slot so no scheduler fallback
        // can silently substitute another network for this conformance check.
        let paths = PolicyPaths { walk: root.join(name), ..Default::default() };
        let mut policy = Policy::load(&paths, 0.05)?;
        let mut previous = [0.0f32; 14];
        for index in 0..count {
            let t = index as f64 * 0.17;
            let pitch = 0.21 * t.sin();
            let imu = ImuData { gyro: [0.1*t.cos(), -0.2*t.sin(), 0.3*(t+0.4).sin()],
                gravity: [pitch.sin(), 0.0, -pitch.cos()], quat: [(pitch/2.0).cos(), 0.0, (pitch/2.0).sin(), 0.0] };
            let positions: [f64; 15] = std::array::from_fn(|j| DEFAULT_POSITION[j] + 0.08*(t+j as f64*0.31).sin());
            let velocities: [f64; 15] = std::array::from_fn(|j| 0.3*(t+j as f64*0.21).cos());
            let command = Command { twist: [0.2*t.sin(), 0.1*t.cos(), 0.5*(t+0.8).sin()],
                head: [0.04*t.sin(), 0.03*t.cos(), 0.06*(t+0.3).sin(), 0.01*t.cos()],
                body: BodyPose { z: 0.004*t.sin(), roll: 0.02*t.cos(), pitch: 0.03*t.sin() } };
            let observation = Observation::build(&imu, &positions, &velocities, &DEFAULT_POSITION, &previous, &command);
            let action = policy.infer(&observation, Net::Walk)?;
            let offsets = Observation::scatter_action(&action);
            let targets: [f64; 15] = std::array::from_fn(|j| DEFAULT_POSITION[j] + offsets[j]);
            println!("{}", json!({"policy":name,"index":index,"gyro":imu.gyro,"gravity":imu.gravity,
                "joint_positions":positions,"joint_velocities":velocities,"home_pose":DEFAULT_POSITION,
                "previous_raw_action":previous,"command13":&observation.as_slice()[48..],
                "observation61":observation.as_slice(),"raw_action14":action,"joint_targets15":targets}));
            previous = action;
        }
    }
    Ok(())
}
