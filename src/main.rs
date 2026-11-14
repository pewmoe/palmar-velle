mod actions;
mod camera;
mod config;
mod gestures;
mod gui;
mod hand_model;
mod landmarks;
mod matcher;
mod mouse;

use anyhow::{Context, Result};
use clap::Parser;
use opencv::{core, prelude::*};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Sender, SyncSender, TrySendError};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use actions::ActionRunner;
use camera::WebcamCapture;
use config::{ConfigRequest, UpdateKind};
use gestures::{GestureEvent, GestureRecognizer};
use gui::{Control, GuiHandles, PreviewFrame, StatusEvent};
use hand_model::HandModel;
use matcher::{normalize_pose, GestureMatcher, PoseRecorder, RecordStep};
use mouse::VirtualMouse;

/// Preview is capped at ~30 fps regardless of camera rate.
const PREVIEW_MIN_INTERVAL: Duration = Duration::from_millis(33);

#[derive(Parser)]
#[command(name = "palmar-velle", version, about = "Webcam hand-gesture mouse, shortcut and command control")]
struct Cli {
    /// Hand-landmark ONNX model (default: $PALMAR_VELLE_MODEL, ./hand_landmark.onnx,
    /// /usr/local/share/palmar-velle/, /usr/share/palmar-velle/)
    model: Option<PathBuf>,

    /// Camera index
    camera: Option<u32>,

    /// Show the preview/control window
    #[arg(long, conflicts_with = "headless")]
    gui: bool,

    /// No window (default). Preview and record triggers are available via signals.
    #[arg(long)]
    headless: bool,

    /// Record a custom gesture called NAME into gestures.toml, then exit
    /// (with --gui the app keeps running)
    #[arg(long, value_name = "NAME")]
    record: Option<String>,

    /// Alternate gestures.toml path
    #[arg(long, value_name = "PATH")]
    config: Option<PathBuf>,
}

fn resolve_model(arg: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(p) = arg {
        return Ok(p);
    }
    if let Some(p) = std::env::var_os("PALMAR_VELLE_MODEL").map(PathBuf::from) {
        return Ok(p);
    }
    const CANDIDATES: &[&str] = &[
        "hand_landmark.onnx",
        "/usr/local/share/palmar-velle/hand_landmark.onnx",
        "/usr/share/palmar-velle/hand_landmark.onnx",
    ];
    CANDIDATES
        .iter()
        .map(PathBuf::from)
        .find(|p| p.is_file())
        .context("hand_landmark.onnx not found: pass a path or set PALMAR_VELLE_MODEL")
}

/// Print (cheap, rare events only) and forward to the GUI without blocking.
fn report(tx: &Option<SyncSender<StatusEvent>>, ev: StatusEvent) {
    match &ev {
        StatusEvent::Fired(_) => log::info!("{ev}"),
        _ => println!("{ev}"),
    }
    if let Some(tx) = tx {
        let _ = tx.try_send(ev);
    }
}

fn apply_events(mouse: &mut VirtualMouse, events: Vec<GestureEvent>) -> Result<()> {
    for event in events {
        match event {
            GestureEvent::Move { x, y } => mouse.move_to(x, y)?,
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
            GestureEvent::Scroll { dy } => mouse.scroll(dy)?,
            GestureEvent::DoubleClick => {
                log::debug!("Double click");
                mouse.double_click()?;
            }
            GestureEvent::HandLost => {
                log::debug!("Hand lost");
                mouse.reset();
            }
        }
    }
    Ok(())
}

/// Signals (all handled off the hot path):
///   SIGUSR1  toggle the camera preview feed
///   SIGUSR2  record a gesture named gesture-<unix-time>
///   SIGHUP   reload gestures.toml
///   SIGINT / SIGTERM  graceful quit (a second SIGINT force-exits)
fn spawn_signal_thread(
    quit: Arc<AtomicBool>,
    preview: Arc<AtomicBool>,
    control: Sender<Control>,
    config_requests: Sender<ConfigRequest>,
) -> Result<()> {
    use signal_hook::consts::{SIGHUP, SIGINT, SIGTERM, SIGUSR1, SIGUSR2};
    use signal_hook::iterator::Signals;

    let mut signals = Signals::new([SIGUSR1, SIGUSR2, SIGHUP, SIGINT, SIGTERM])
        .context("installing signal handlers")?;

    std::thread::Builder::new()
        .name("signals".into())
        .spawn(move || {
            let mut interrupts = 0u32;
            for sig in signals.forever() {
                match sig {
                    SIGUSR1 => {
                        preview.fetch_xor(true, Ordering::Relaxed);
                    }
                    SIGUSR2 => {
                        let secs = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .map(|d| d.as_secs())
                            .unwrap_or(0);
                        let _ = control.send(Control::Record { name: format!("gesture-{secs}") });
                    }
                    SIGHUP => {
                        let _ = config_requests.send(ConfigRequest::Reload);
                    }
                    SIGINT | SIGTERM => {
                        interrupts += 1;
                        if interrupts > 1 {
                            std::process::exit(130);
                        }
                        quit.store(true, Ordering::Relaxed);
                    }
                    _ => {}
                }
            }
        })
        .context("spawning signal thread")?;
    Ok(())
}

fn main() -> Result<()> {
    env_logger::init();

    // Initialize ort runtime environment (required for v2.0)
    let _ = ort::init().commit();

    let cli = Cli::parse();
    let model_path = resolve_model(cli.model.clone())?;
    let camera_index = cli.camera.unwrap_or(0);
    let config_path = cli.config.clone().unwrap_or_else(config::default_path);
    let use_gui = cli.gui;
    let exit_after_record = cli.record.is_some() && !use_gui;

    println!("Palmar Velle");
    println!("------------");
    println!("Model  : {}", model_path.display());
    println!("Camera : {camera_index}");
    println!("Config : {}", config_path.display());
    println!("Mode   : {}", if use_gui { "gui" } else { "headless" });
    println!();

    // ------------------------------------------------------------------
    // Shared state + control plane (everything the loop touches is
    // non-blocking: atomics, try_recv, try_send, unbounded send).
    // ------------------------------------------------------------------
    let quit = Arc::new(AtomicBool::new(false));
    let preview_enabled = Arc::new(AtomicBool::new(false));
    let frame_pending = Arc::new(AtomicBool::new(false));
    let (control_tx, control_rx) = mpsc::channel::<Control>();

    let cfg = config::spawn(config_path.clone())?;
    spawn_signal_thread(
        quit.clone(),
        preview_enabled.clone(),
        control_tx.clone(),
        cfg.requests.clone(),
    )?;

    // ------------------------------------------------------------------
    // Hardware + model
    // ------------------------------------------------------------------
    println!("Starting camera...");
    let mut camera = WebcamCapture::open(camera_index)?;
    let (camera_w, camera_h) = camera.resolution();
    println!("Camera resolution: {camera_w}x{camera_h}");

    println!("Loading hand model...");
    let mut model = HandModel::new(model_path.to_str().context("model path is not valid UTF-8")?)?;

    println!("Creating virtual mouse and keyboard...");
    let mut mouse = VirtualMouse::new()?;
    let mut runner = ActionRunner::new()?;

    // ------------------------------------------------------------------
    // Optional GUI thread
    // ------------------------------------------------------------------
    let mut preview_tx: Option<SyncSender<PreviewFrame>> = None;
    let mut status_tx: Option<SyncSender<StatusEvent>> = None;
    if use_gui {
        let (ptx, prx) = mpsc::sync_channel::<PreviewFrame>(1);
        let (stx, srx) = mpsc::sync_channel::<StatusEvent>(64);
        preview_tx = Some(ptx);
        status_tx = Some(stx);
        preview_enabled.store(true, Ordering::Relaxed);
        gui::spawn(GuiHandles {
            frames: prx,
            status: srx,
            control: control_tx.clone(),
            config_requests: cfg.requests.clone(),
            preview_enabled: preview_enabled.clone(),
            frame_pending: frame_pending.clone(),
            quit: quit.clone(),
            config_path: config_path.clone(),
        })?;
    }

    if let Some(name) = cli.record.clone() {
        let _ = control_tx.send(Control::Record { name });
    }

    println!();
    println!("Built-in gestures:");
    println!("  Palm position         = move cursor");
    println!("  Thumb + index         = left click / drag");
    println!("  Thumb + middle        = right click");
    println!("  Thumb + index+middle  = scroll");
    println!("  Two quick pinches     = double click");
    println!("Custom gestures: {}", config_path.display());
    println!("Signals: USR1 toggle preview, USR2 record, HUP reload. Ctrl+C to stop.");
    println!();

    let mut recognizer = GestureRecognizer::new();
    let mut matcher = GestureMatcher::new();
    let mut recorder: Option<PoseRecorder> = None;

    let mut fps = 0.0f32;
    let mut t_prev = Instant::now();
    let mut last_preview = Instant::now() - PREVIEW_MIN_INTERVAL;

    while !quit.load(Ordering::Relaxed) {
        // ------------------------------------------------------------
        // 0. Non-blocking control plane
        // ------------------------------------------------------------
        while let Ok(ctrl) = control_rx.try_recv() {
            match ctrl {
                Control::Record { name } => {
                    if let Err(e) = config::valid_name(&name) {
                        report(&status_tx, StatusEvent::RecordFailed(e));
                    } else if recorder.is_some() {
                        report(&status_tx, StatusEvent::RecordFailed("already recording".into()));
                    } else {
                        // Release any held button and reset pointer state:
                        // the pointer is frozen for the duration.
                        let events = recognizer.process(None);
                        let _ = apply_events(&mut mouse, events);
                        matcher.reset();
                        recorder = Some(PoseRecorder::new(name, Instant::now()));
                    }
                }
            }
        }

        while let Ok(update) = cfg.updates.try_recv() {
            match update.gestures {
                Some(defs) => {
                    let mut notes = update.notes;
                    notes.extend(matcher.set_gestures(&defs));
                    report(&status_tx, StatusEvent::ConfigLoaded { count: matcher.len(), notes });
                }
                None => report(&status_tx, StatusEvent::ConfigError(update.notes.join("; "))),
            }
            match update.kind {
                UpdateKind::Appended(name) => {
                    // stability was reported when capture finished; this confirms the write
                    println!("saved gesture '{name}' to {}", config_path.display());
                    if exit_after_record {
                        quit.store(true, Ordering::Relaxed);
                    }
                }
                UpdateKind::AppendFailed(e) => {
                    report(&status_tx, StatusEvent::RecordFailed(e));
                    if exit_after_record {
                        quit.store(true, Ordering::Relaxed);
                    }
                }
                UpdateKind::Initial | UpdateKind::Reloaded => {}
            }
        }

        // ------------------------------------------------------------
        // 1. Capture RGB frame from webcam (blocks until a frame is ready)
        // ------------------------------------------------------------
        let rgb = camera
            .next_frame()
            .context("failed to capture camera frame")?;

        let dt = t_prev.elapsed().as_secs_f32();
        t_prev = Instant::now();
        if dt > 0.0 {
            fps = if fps == 0.0 { 1.0 / dt } else { fps * 0.9 + 0.1 / dt };
        }

        let height = rgb.height() as i32;
        let aspect = rgb.width() as f32 / rgb.height() as f32;

        // ------------------------------------------------------------
        // 2. Wrap the RGB image as an OpenCV Mat (RGB, no conversion)
        // ------------------------------------------------------------
        let rgb_data = rgb.as_raw();
        let rgb_mat = core::Mat::from_slice(rgb_data)
            .context("failed to create OpenCV Mat from camera frame")?;
        let rgb_mat = rgb_mat
            .reshape(3, height)
            .context("failed to reshape camera frame")?;

        // ------------------------------------------------------------
        // 3. Hand detection / landmark inference
        // ------------------------------------------------------------
        let t_infer = Instant::now();
        let hand = model
            .detect(&rgb_mat)
            .context("hand model inference failed")?;
        let infer_ms = t_infer.elapsed().as_secs_f32() * 1000.0;

        let now = Instant::now();
        let pose = hand.as_ref().and_then(|h| normalize_pose(h, aspect));

        // ------------------------------------------------------------
        // 4. Input dispatch. Mouse/gesture events go out first; nothing
        //    optional (preview, status) runs before them.
        // ------------------------------------------------------------
        if let Some(rec) = recorder.as_mut() {
            match rec.feed(now, pose.as_ref()) {
                RecordStep::Idle => {}
                RecordStep::Countdown(s) => report(&status_tx, StatusEvent::Countdown(s)),
                RecordStep::Progress { got, total } => {
                    report(&status_tx, StatusEvent::Progress { got, total })
                }
                RecordStep::Finished(Ok((def, stability))) => {
                    recorder = None;
                    report(
                        &status_tx,
                        StatusEvent::Recorded {
                            name: def.name.clone(),
                            stability,
                            path: config_path.display().to_string(),
                        },
                    );
                    let _ = cfg.requests.send(ConfigRequest::Append(def));
                }
                RecordStep::Finished(Err(e)) => {
                    recorder = None;
                    report(&status_tx, StatusEvent::RecordFailed(e));
                    if exit_after_record {
                        quit.store(true, Ordering::Relaxed);
                    }
                }
            }
        } else {
            let events = recognizer.process(hand.as_ref());
            let engaged = recognizer.is_engaged();
            apply_events(&mut mouse, events)?;

            // Custom gestures never compete with a pinch/scroll in progress.
            let candidate = if engaged { None } else { pose.as_ref() };
            if let Some(fired) = matcher.update(candidate, now) {
                let name = fired.name;
                if let Err(e) = runner.run(name, fired.action, &mut mouse) {
                    log::warn!("gesture '{name}' action failed: {e:#}");
                }
                report(&status_tx, StatusEvent::Fired(name.to_owned()));
            }
        }

        // ------------------------------------------------------------
        // 5. Preview: skipped entirely (one relaxed atomic load) when
        //    disabled; throttled and dropped when the UI is behind.
        // ------------------------------------------------------------
        if preview_enabled.load(Ordering::Relaxed) {
            if let Some(tx) = &preview_tx {
                if !frame_pending.load(Ordering::Acquire)
                    && last_preview.elapsed() >= PREVIEW_MIN_INTERVAL
                {
                    last_preview = Instant::now();
                    let frame = gui::make_preview_frame(&rgb, hand.as_ref(), fps, infer_ms);
                    match tx.try_send(frame) {
                        Ok(()) => frame_pending.store(true, Ordering::Release),
                        Err(TrySendError::Full(_)) => {}
                        Err(TrySendError::Disconnected(_)) => {
                            // GUI thread is gone: stop paying for previews.
                            preview_enabled.store(false, Ordering::Relaxed);
                            preview_tx = None;
                            status_tx = None;
                        }
                    }
                }
            }
        }
    }

    // Never leave a button held on shutdown.
    let _ = apply_events(&mut mouse, recognizer.process(None));
    println!("Stopped.");
    Ok(())
}
