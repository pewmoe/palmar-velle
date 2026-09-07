# palmar-velle:BETA

A lightweight, high-performance gesture-controlled virtual mouse written in **Rust**, powered by computer vision and ONNX runtime (`ort`). `velle` transforms your standard webcam into a touchless cursor controller featuring smooth tracking, deadzone jitter reduction, and native Linux kernel-level input injection via `uinput`.

## Features

* **Real-Time Hand Tracking**: Uses ONNX-based hand landmark models equipped with an expanded Region of Interest (ROI) tracking mechanism to keep your hand securely locked in frame.

* **Kernel-Level Input Injection**: Uses `uinput` for seamless, native mouse movement and clicking without being tied to a specific desktop environment.

* **Jitter & Smoothing Filters**: Built-in Exponential Moving Average (EMA) smoothing and deadzone thresholding eliminate micro-jitter when holding the cursor still.

* **Comprehensive Gesture Control**: Support for cursor navigation, left/right clicks, dragging, scrolling, and double clicks entirely through intuitive hand gestures.

## Gestures & Controls

| Gesture | Action | 
 | ----- | ----- | 
| **Index Finger Tracking** | Move cursor across screen | 
| **Thumb + Index Pinch** | Left click / Drag | 
| **Thumb + Middle Pinch** | Right click | 
| **Thumb + Index + Middle** | Vertical Scroll | 
| **Two Quick Pinches** | Double Click | 

## Prerequisites

Before building and running `velle`, verify your Linux environment satisfies the following requirements:

1. **Rust toolchain** (latest stable release)

2. **OpenCV development libraries** (required by the `opencv` crate)

3. **A hand landmark ONNX model** (such as `hand_landmark.onnx` placed in your project directory)

4. **Device Permissions**: Ensure your user account has permission to access your webcam (`/dev/video*`) and create virtual input devices (`uinput`).

## Installation & Building

1. Clone or navigate to your repository workspace:

   ```
   cd velle
   
   ```

2. Compile the project in release mode for maximum tracking performance:

   ```
   cargo build --release
   
   ```

## Usage

Run the optimized binary, supplying the path to your ONNX model and your webcam device index (typically `0`):

```
cargo run --release -- hand_landmark.onnx 0

```

To stop the application at any time, press **Ctrl+C** in your terminal.

## Tech Stack

* [**Rust**](https://www.rust-lang.org/) — Core systems logic and rigorous memory safety

* [**ORT**](https://github.com/pykeio/ort) — ONNX Runtime bindings ensuring fast machine learning inference

* [**OpenCV**](https://opencv.org/) — High-speed image processing and color space conversion

* [**Nokhwa**](https://github.com/raymanfx/nokhwa) — Cross-platform webcam frame capture

* [**uinput**](https://github.com/pop-os/uinput) — Linux kernel input event generation



## this is my basically baby, I've been working on it for 4 months and you better believe it's gonna be the solution for gesture control.
