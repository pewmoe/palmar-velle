//! Optional preview/control window (egui via eframe), on its own thread.
//!
//! The pipeline thread never waits on this module:
//!   * frames travel over `sync_channel(1)` with `try_send` (drop if full);
//!   * `frame_pending` lets the pipeline skip the RGBA conversion entirely
//!     while the UI hasn't consumed the previous frame;
//!   * `preview_enabled` is a single relaxed atomic read per frame; when
//!     false, no conversion and no channel send happens;
//!   * status events use `try_send` too.
//!
//! Landmark skeletons are drawn here as vector overlays from 21 normalized
//! points, so the pipeline never burns pixels into the frame.

use std::collections::VecDeque;
use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use eframe::egui::{self, pos2, vec2, Color32, ColorImage, Rect, Sense, Stroke, TextureHandle, TextureOptions};
use image::RgbImage;
use winit::platform::wayland::EventLoopBuilderExtWayland;
use winit::platform::x11::EventLoopBuilderExtX11;

use crate::config::{valid_name, ConfigRequest};
use crate::landmarks::Hand;

pub const PREVIEW_W: usize = 320;
pub const PREVIEW_H: usize = 240;

// ---------------------------------------------------------------------------
// Messages shared with the pipeline
// ---------------------------------------------------------------------------

pub struct PreviewFrame {
    /// PREVIEW_W x PREVIEW_H, RGBA8.
    pub rgba: Vec<u8>,
    /// Source frame width/height. The 320x240 image is displayed at this
    /// aspect so non-4:3 cameras aren't distorted.
    pub src_aspect: f32,
    /// 21 landmarks, normalized to the source frame.
    pub landmarks: Option<[[f32; 2]; 21]>,
    pub fps: f32,
    pub infer_ms: f32,
}

/// UI -> pipeline.
pub enum Control {
    Record { name: String },
}

/// Pipeline -> UI (and stdout).
pub enum StatusEvent {
    ConfigLoaded { count: usize, notes: Vec<String> },
    ConfigError(String),
    Countdown(u64),
    Progress { got: usize, total: usize },
    Recorded { name: String, stability: f32, path: String },
    RecordFailed(String),
    Fired(String),
}

impl fmt::Display for StatusEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConfigLoaded { count, notes } => {
                write!(f, "gestures.toml loaded: {count} gesture(s)")?;
                for n in notes {
                    write!(f, "\n  warning: {n}")?;
                }
                Ok(())
            }
            Self::ConfigError(e) => write!(f, "gestures.toml error (keeping previous set): {e}"),
            Self::Countdown(s) => write!(f, "recording starts in {s}... hold your pose"),
            Self::Progress { got, total } => write!(f, "recording {got}/{total}"),
            Self::Recorded { name, stability, path } => write!(
                f,
                "captured '{name}' (stability {stability:.3}), saving to {path}; edit its [gesture.action] to make it do something"
            ),
            Self::RecordFailed(e) => write!(f, "recording failed: {e}"),
            Self::Fired(n) => write!(f, "gesture fired: {n}"),
        }
    }
}

pub struct GuiHandles {
    pub frames: Receiver<PreviewFrame>,
    pub status: Receiver<StatusEvent>,
    pub control: Sender<Control>,
    pub config_requests: Sender<ConfigRequest>,
    pub preview_enabled: Arc<AtomicBool>,
    pub frame_pending: Arc<AtomicBool>,
    pub quit: Arc<AtomicBool>,
    pub config_path: PathBuf,
}

// ---------------------------------------------------------------------------
// Preview frame builder (called on the pipeline thread, only when enabled)
// ---------------------------------------------------------------------------

/// 2x2 box downscale to 320x240 RGBA, pure Rust. About 1 ms for a 640x480
/// source; it runs after the frame's mouse events have been dispatched.
pub fn make_preview_frame(rgb: &RgbImage, hand: Option<&Hand>, fps: f32, infer_ms: f32) -> PreviewFrame {
    let sw = rgb.width() as usize;
    let sh = rgb.height() as usize;
    let src = rgb.as_raw();

    let mut out = vec![255u8; PREVIEW_W * PREVIEW_H * 4];
    for y in 0..PREVIEW_H {
        let y0 = y * sh / PREVIEW_H;
        let y1 = (y0 + 1).min(sh - 1);
        for x in 0..PREVIEW_W {
            let x0 = x * sw / PREVIEW_W;
            let x1 = (x0 + 1).min(sw - 1);
            let i00 = (y0 * sw + x0) * 3;
            let i01 = (y0 * sw + x1) * 3;
            let i10 = (y1 * sw + x0) * 3;
            let i11 = (y1 * sw + x1) * 3;
            let o = (y * PREVIEW_W + x) * 4;
            for c in 0..3 {
                let s = src[i00 + c] as u16 + src[i01 + c] as u16 + src[i10 + c] as u16 + src[i11 + c] as u16;
                out[o + c] = (s >> 2) as u8;
            }
        }
    }

    let landmarks = hand.and_then(|h| {
        if h.landmarks.len() < 21 {
            return None;
        }
        let mut lm = [[0.0f32; 2]; 21];
        for (dst, p) in lm.iter_mut().zip(h.landmarks.iter()) {
            *dst = [p.x, p.y];
        }
        Some(lm)
    });

    PreviewFrame {
        rgba: out,
        src_aspect: sw as f32 / sh as f32,
        landmarks,
        fps,
        infer_ms,
    }
}

// ---------------------------------------------------------------------------
// Window
// ---------------------------------------------------------------------------

const HAND_BONES: &[(usize, usize)] = &[
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
];

/// Spawn the GUI thread. If the window is closed normally the whole app is
/// asked to quit; if the GUI cannot start (no display, no GL) the pipeline
/// simply continues headless.
pub fn spawn(handles: GuiHandles) -> Result<()> {
    std::thread::Builder::new()
        .name("gui".into())
        .spawn(move || {
            let quit = handles.quit.clone();
            let preview = handles.preview_enabled.clone();

            let options = eframe::NativeOptions {
                viewport: egui::ViewportBuilder::default()
                    .with_title("Palmar Velle")
                    .with_app_id("palmar-velle") // matches palmar-velle.desktop on Wayland
                    .with_inner_size([700.0, 680.0])
                    .with_min_inner_size([420.0, 420.0]),
                // winit normally insists on the main thread; both backends
                // allow opting out on Linux.
                event_loop_builder: Some(Box::new(|builder| {
                    EventLoopBuilderExtWayland::with_any_thread(builder, true);
                    EventLoopBuilderExtX11::with_any_thread(builder, true);
                })),
                ..Default::default()
            };

            let result = eframe::run_native(
                "Palmar Velle",
                options,
                Box::new(move |_cc| Ok(Box::new(App::new(handles)))),
            );

            match result {
                Ok(()) => quit.store(true, Ordering::Relaxed),
                Err(e) => {
                    log::error!("GUI failed to start ({e}); continuing headless");
                    preview.store(false, Ordering::Relaxed);
                    // `handles` is dropped with the closure, which disconnects
                    // the channels so the pipeline stops producing previews.
                }
            }
        })
        .context("spawning GUI thread")?;
    Ok(())
}

struct App {
    h: GuiHandles,
    texture: Option<TextureHandle>,
    last: Option<(f32, Option<[[f32; 2]; 21]>, f32, f32)>, // aspect, landmarks, fps, infer_ms
    log: VecDeque<String>,
    record_name: String,
    recording: Option<String>,
    gesture_count: usize,
}

impl App {
    fn new(h: GuiHandles) -> Self {
        Self {
            h,
            texture: None,
            last: None,
            log: VecDeque::new(),
            record_name: String::new(),
            recording: None,
            gesture_count: 0,
        }
    }

    fn push_log(&mut self, line: String) {
        if self.log.len() >= 200 {
            self.log.pop_front();
        }
        self.log.push_back(line);
    }

    fn drain_frames(&mut self, ctx: &egui::Context) {
        let mut newest = None;
        while let Ok(f) = self.h.frames.try_recv() {
            newest = Some(f);
        }
        if let Some(f) = newest {
            self.h.frame_pending.store(false, Ordering::Release);
            let img = ColorImage::from_rgba_unmultiplied([PREVIEW_W, PREVIEW_H], &f.rgba);
            match &mut self.texture {
                Some(t) => t.set(img, TextureOptions::LINEAR),
                None => self.texture = Some(ctx.load_texture("camera", img, TextureOptions::LINEAR)),
            }
            self.last = Some((f.src_aspect, f.landmarks, f.fps, f.infer_ms));
        }
    }

    fn drain_status(&mut self) {
        while let Ok(ev) = self.h.status.try_recv() {
            match &ev {
                StatusEvent::ConfigLoaded { count, .. } => self.gesture_count = *count,
                StatusEvent::Countdown(s) => self.recording = Some(format!("starting in {s}...")),
                StatusEvent::Progress { got, total } => {
                    self.recording = Some(format!("capturing {got}/{total}"))
                }
                StatusEvent::Recorded { .. } | StatusEvent::RecordFailed(_) => self.recording = None,
                StatusEvent::ConfigError(_) | StatusEvent::Fired(_) => {}
            }
            // Per-frame-ish progress ticks would just flood the log.
            if !matches!(ev, StatusEvent::Progress { .. }) {
                self.push_log(ev.to_string());
            }
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain_frames(ctx);
        self.drain_status();

        let preview_on = self.h.preview_enabled.load(Ordering::Relaxed);

        egui::TopBottomPanel::top("controls").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                let mut on = preview_on;
                if ui.toggle_value(&mut on, "Camera feed").changed() {
                    self.h.preview_enabled.store(on, Ordering::Relaxed);
                }
                ui.separator();
                if ui.button("Reload gestures.toml").clicked() {
                    let _ = self.h.config_requests.send(ConfigRequest::Reload);
                }
                ui.separator();
                ui.label(format!("{} custom gesture(s)", self.gesture_count));
            });
            ui.add_space(4.0);
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            // --- camera preview ---
            if preview_on {
                if let (Some(tex), Some((aspect, landmarks, fps, infer_ms))) = (&self.texture, &self.last) {
                    let w = ui.available_width().min(640.0);
                    let (rect, _) = ui.allocate_exact_size(vec2(w, w / aspect.max(0.1)), Sense::hover());
                    let painter = ui.painter_at(rect);
                    // Mirrored, like the cursor mapping, so it behaves like a mirror.
                    painter.image(
                        tex.id(),
                        rect,
                        Rect::from_min_max(pos2(1.0, 0.0), pos2(0.0, 1.0)),
                        Color32::WHITE,
                    );
                    if let Some(lm) = landmarks {
                        draw_skeleton(&painter, rect, lm);
                    }
                    ui.label(format!("pipeline {fps:.0} fps, inference {infer_ms:.1} ms"));
                } else {
                    ui.label("Waiting for first frame...");
                }
            } else {
                ui.label(
                    "Camera feed is off. The pipeline is not converting or sending preview \
                     frames, so it costs nothing.",
                );
            }

            ui.separator();

            // --- recording ---
            ui.heading("Record a custom gesture");
            ui.horizontal(|ui| {
                ui.add(
                    egui::TextEdit::singleline(&mut self.record_name)
                        .hint_text("name, e.g. peace-sign")
                        .desired_width(220.0),
                );
                let name_ok = valid_name(&self.record_name).is_ok();
                let idle = self.recording.is_none();
                if ui.add_enabled(name_ok && idle, egui::Button::new("Record pose")).clicked() {
                    let name = self.record_name.trim().to_string();
                    if self.h.control.send(Control::Record { name }).is_ok() {
                        self.recording = Some("starting...".into());
                    }
                }
                if let Some(state) = &self.recording {
                    ui.label(format!("Recording: {state}"));
                }
            });
            ui.small("3 s countdown, then 30 frames are averaged. The pointer is frozen while recording.");
            ui.small(format!("Config: {}", self.h.config_path.display()));

            ui.separator();
            ui.label("Log");
            egui::ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
                for line in &self.log {
                    ui.monospace(line);
                }
            });
        });

        let recording = self.recording.is_some();
        ctx.request_repaint_after(if preview_on || recording {
            Duration::from_millis(16)
        } else {
            Duration::from_millis(200)
        });
    }
}

fn draw_skeleton(painter: &egui::Painter, rect: Rect, lm: &[[f32; 2]; 21]) {
    let to_screen = |p: &[f32; 2]| {
        pos2(
            rect.left() + (1.0 - p[0]) * rect.width(),
            rect.top() + p[1] * rect.height(),
        )
    };
    for &(a, b) in HAND_BONES {
        painter.line_segment(
            [to_screen(&lm[a]), to_screen(&lm[b])],
           Stroke::new(2.0_f32, Color32::from_rgb(80, 220, 180)),
        );
    }
    for (i, p) in lm.iter().enumerate() {
        let tip = matches!(i, 4 | 8 | 12 | 16 | 20);
        painter.circle_filled(
            to_screen(p),
            if tip { 4.0 } else { 2.5 },
            if tip { Color32::YELLOW } else { Color32::WHITE },
        );
    }
}
