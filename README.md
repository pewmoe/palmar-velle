# palmar-velle:BETA

A lightweight, high-performance gesture-controlled virtual mouse written in **Rust**, powered by computer vision and ONNX runtime (`ort`). `velle` transforms your standard webcam into a touchless cursor controller featuring smooth tracking, scale-invariant pinch detection, and native Linux kernel-level input injection via `uinput`.

## Features

- **Real-Time Hand Tracking**: Uses an ONNX-based hand landmark model with an adaptive Region of Interest (ROI) tracker that follows your hand across the frame and smoothly resizes as you move closer to or farther from the camera.

- **Kernel-Level Input Injection**: Uses `uinput` with relative (`REL_X`/`REL_Y`) mouse motion events, the same signature a physical mouse uses — so it's correctly recognized as a pointer device by libinput and works out of the box under both X11 and Wayland, without being tied to a specific desktop environment.

- **Jitter & Smoothing Filters**: A One Euro Filter smooths cursor motion frame-to-frame, and pinch thresholds are normalized by hand scale (not raw frame distance) with a short debounce window — so accidental clicks from landmark noise or camera-distance drift are rejected without adding noticeable input lag.

- **Comprehensive Gesture Control**: Support for cursor navigation, left/right clicks, dragging, scrolling, and double clicks entirely through intuitive hand gestures.

## Gestures & Controls

| Gesture                          | Action                     |
| --------------------------------- | -------------------------- |
| **Palm Center Tracking**          | Move cursor across screen  |
| **Thumb + Index Pinch**           | Left click / Drag          |
| **Thumb + Middle Pinch**          | Right click                |
| **Thumb + Index + Middle Pinch**  | Vertical Scroll            |
| **Two Quick Index Pinches**       | Double Click               |

Cursor position tracks the centroid of your wrist and two knuckle landmarks (not a single fingertip), which stays stable even while your fingers are pinching — so clicking and dragging doesn't cause the cursor to jump.

Pinch thresholds scale with your hand's size in frame, so the same physical pinch gesture triggers consistently whether your hand is close to or farther from the webcam.

## Prerequisites

Before building and running `velle`, verify your Linux environment satisfies the following requirements:

1. **Rust toolchain** (latest stable release)

2. **OpenCV development libraries** (required by the `opencv` crate) — including a matching `libstdc++-dev` package for your installed GCC version, which the `opencv` crate's bindgen step needs to parse OpenCV's C++ headers.

3. **A hand landmark ONNX model** (such as `hand_landmark.onnx` placed in your project directory)

4. **Device Permissions**: Ensure your user account has permission to access your webcam (`/dev/video*`) and create virtual input devices (`uinput`). If `/dev/uinput` isn't writable by your user, add yourself to the relevant group (e.g. `input`) and log out/in.

## Installation & Building

1. Clone or navigate to your repository workspace:

   ```
   cd palmar-velle
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

### Verifying the virtual mouse device

While `velle` is running, you can confirm the virtual pointer device is correctly recognized by your system:

```
sudo libinput list-devices | grep -A5 "Gesture Control"
```

You should see `Capabilities: pointer` listed for the "Gesture Control Virtual Mouse" device.

## Tech Stack

- [**Rust**](https://www.rust-lang.org/) — Core systems logic and rigorous memory safety

- [**ORT**](https://github.com/pykeio/ort) — ONNX Runtime bindings ensuring fast machine learning inference

- [**OpenCV**](https://opencv.org/) — High-speed image cropping, resizing, and color space handling

- [**Nokhwa**](https://github.com/raymanfx/nokhwa) — Cross-platform webcam frame capture

- [**uinput**](https://github.com/pop-os/uinput) — Linux kernel input event generation

## this is my basically baby, I've been working on it for 4 months and you better believe it's gonna be the solution for gesture control.
