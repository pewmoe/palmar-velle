mod camera;
mod gestures;
mod hand_model;
mod landmarks;
mod mouse;

use anyhow::{Context, Result};
use opencv::{core, imgproc, prelude::*};
use std::env;

use camera::WebcamCapture;
use gestures::{GestureEvent, GestureRecognizer};
use hand_model::HandModel;
use mouse::VirtualMouse;

fn main() -> Result<()> {
    env_logger::init();

    // Initialize ort runtime environment (required for v2.0)
    let _ = ort::init().commit();

    let args: Vec<String> = env::args().collect();

    // Usage:
    //
    //     gesture-control <model.onnx> [camera_index]
    //
    // Example:
    //
    //     gesture-control hand_landmark.onnx 0
    //
    let model_path = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("hand_landmark.onnx");

    let camera_index: u32 = args
        .get(2)
        .map(|s| s.parse())
        .transpose()
        .context("camera index must be a number")?
        .unwrap_or(0);

    println!("Gesture Control");
    println!("---------------");
    println!("Model : {model_path}");
    println!("Camera: {camera_index}");
    println!();
    println!("Starting camera...");

    let mut camera = WebcamCapture::open(camera_index)?;

    let (camera_w, camera_h) = camera.resolution();

    println!(
        "Camera resolution: {}x{}",
        camera_w, camera_h
    );

    println!("Loading hand model...");

    let mut model = HandModel::new(model_path)?;

    println!("Creating virtual mouse...");

    let mut mouse = VirtualMouse::new()?;

    println!("Starting gesture recognition.");
    println!();
    println!("Gestures:");
    println!("  Index finger          = move cursor");
    println!("  Thumb + index         = left click / drag");
    println!("  Thumb + middle        = right click");
    println!("  Thumb + index+middle  = scroll");
    println!("  Two quick pinches     = double click");
    println!();
    println!("Move your hand in front of the camera.");
    println!("Press Ctrl+C to stop.");
    println!();

    let mut recognizer = GestureRecognizer::new();

    loop {
        // ------------------------------------------------------------
        // 1. Capture RGB frame from webcam
        // ------------------------------------------------------------

        let rgb = camera
            .next_frame()
            .context("failed to capture camera frame")?;

        let width = rgb.width() as i32;
        let height = rgb.height() as i32;

        // ------------------------------------------------------------
        // 2. Convert RGB image to an OpenCV BGR Mat.
        // ------------------------------------------------------------

        let rgb_data = rgb.as_raw();

        let rgb_mat = core::Mat::from_slice(rgb_data)
            .context("failed to create OpenCV Mat from camera frame")?;

        let rgb_mat = rgb_mat
            .reshape(3, height)
            .context("failed to reshape camera frame")?;

        let mut bgr_mat = core::Mat::default();

        imgproc::cvt_color(
            &rgb_mat,
            &mut bgr_mat,
            imgproc::COLOR_RGB2BGR,
            0,
        )
        .context("failed to convert RGB frame to BGR")?;

        debug_assert_eq!(bgr_mat.cols(), width);
        debug_assert_eq!(bgr_mat.rows(), height);

        // ------------------------------------------------------------
        // 3. Run hand detection / landmark inference
        // ------------------------------------------------------------

        let hand = model
            .detect(&bgr_mat)
            .context("hand model inference failed")?;

        // ------------------------------------------------------------
        // 4. Turn landmarks into gesture events
        // ------------------------------------------------------------

        let events = recognizer.process(hand.as_ref());

        // ------------------------------------------------------------
        // 5. Translate gesture events into virtual mouse events
        // ------------------------------------------------------------

        for event in events {
            match event {
                GestureEvent::Move { x, y } => {
                    mouse.move_to(x, y)?;
                }
                GestureEvent::PrimaryDown => {
                    log::debug!("Primary down");
                    mouse.left_button(true)?;
                }
                GestureEvent::PrimaryUp => {
                    log::debug!("Primary up");
                    mouse.left_button(false)?;
                }
                GestureEvent::SecondaryDown => {
                    log::debug!("Secondary down");
                    mouse.right_button(true)?;
                }
                GestureEvent::SecondaryUp => {
                    log::debug!("Secondary up");
                    mouse.right_button(false)?;
                }
                GestureEvent::Scroll { dy } => {
                    mouse.scroll(dy)?;
                }
                GestureEvent::DoubleClick => {
                    log::debug!("Double click");
                    mouse.double_click()?;
                }
                GestureEvent::HandLost => {
                    log::debug!("Hand lost");
                }
            }
        }
    }
}