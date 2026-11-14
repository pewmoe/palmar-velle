//! User gesture configuration: schema, validation, atomic append and a
//! hot-reload service.
//!
//! All file I/O and TOML parsing happens on the dedicated "config" thread.
//! The capture/inference/uinput loop only ever does a non-blocking
//! `try_recv()` on `ConfigService::updates`.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread;
use std::time::{Duration, SystemTime};

use crate::actions;
use crate::matcher::POSE_DIM;

pub const DEFAULT_THRESHOLD: f64 = 0.97;
pub const DEFAULT_COOLDOWN_MS: u64 = 800;
pub const DEFAULT_HOLD_FRAMES: u32 = 3;

fn default_threshold() -> f64 {
    DEFAULT_THRESHOLD
}
fn default_cooldown_ms() -> u64 {
    DEFAULT_COOLDOWN_MS
}
fn default_hold_frames() -> u32 {
    DEFAULT_HOLD_FRAMES
}

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MouseButtonAction {
    LeftClick,
    RightClick,
    MiddleClick,
    DoubleClick,
}

/// What a gesture does when it fires.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ActionSpec {
    /// Press all keys in order, then release in reverse: ["ctrl","alt","t"].
    KeyCombo { keys: Vec<String> },
    Mouse { button: MouseButtonAction },
    /// Wheel detents. Positive = up, negative = down.
    Scroll { notches: i32 },
    /// Run via `sh -c`, detached, off the hot path.
    Shell { command: String },
}

impl ActionSpec {
    pub fn validate(&self) -> Result<()> {
        match self {
            ActionSpec::KeyCombo { keys } => {
                if keys.is_empty() {
                    bail!("key_combo needs at least one key");
                }
                for k in keys {
                    if actions::parse_key(k).is_none() {
                        bail!("unknown key '{k}'");
                    }
                }
            }
            ActionSpec::Mouse { .. } => {}
            ActionSpec::Scroll { notches } => {
                if *notches == 0 || notches.abs() > 50 {
                    bail!("scroll notches must be non-zero and within +/-50");
                }
            }
            ActionSpec::Shell { command } => {
                if command.trim().is_empty() {
                    bail!("shell command is empty");
                }
            }
        }
        Ok(())
    }
}

/// One `[[gesture]]` table. Field order matters for TOML serialization:
/// plain values first, the nested `action` table last.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GestureDef {
    pub name: String,
    /// 63 floats: 21 landmarks x (x,y,z), wrist-centered, palm-length scaled.
    pub pose: Vec<f64>,
    /// Cosine similarity required to match (0..=1].
    #[serde(default = "default_threshold")]
    pub threshold: f64,
    #[serde(default = "default_cooldown_ms")]
    pub cooldown_ms: u64,
    /// Consecutive matching frames required before firing.
    #[serde(default = "default_hold_frames")]
    pub hold_frames: u32,
    /// If true, keeps firing every `cooldown_ms` while the pose is held.
    /// Default: fire once, then require the pose to be released.
    #[serde(default)]
    pub repeat: bool,
    pub action: ActionSpec,
}

impl GestureDef {
    pub fn validate(&self) -> Result<()> {
        valid_name(&self.name).map_err(anyhow::Error::msg)?;
        if self.pose.len() != POSE_DIM {
            bail!("pose must contain {POSE_DIM} floats, found {}", self.pose.len());
        }
        if self.pose.iter().any(|v| !v.is_finite()) {
            bail!("pose contains NaN or infinity");
        }
        if self.pose.iter().map(|v| v * v).sum::<f64>().sqrt() < 1e-3 {
            bail!("pose vector is all zeros");
        }
        if !(self.threshold > 0.0 && self.threshold <= 1.0) {
            bail!("threshold must be in (0, 1], got {}", self.threshold);
        }
        if self.hold_frames == 0 {
            bail!("hold_frames must be >= 1");
        }
        self.action.validate()
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct ConfigFile {
    #[serde(default, rename = "gesture")]
    gestures: Vec<GestureDef>,
}

/// Gesture names are embedded in generated shell commands and log lines.
pub fn valid_name(name: &str) -> std::result::Result<(), String> {
    let ok_chars = name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    if name.is_empty() || name.len() > 48 || !ok_chars {
        return Err("gesture name must be 1-48 chars of [A-Za-z0-9_-]".to_string());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Paths + file handling
// ---------------------------------------------------------------------------

/// `$XDG_CONFIG_HOME/palmar-velle/gestures.toml`, i.e. normally
/// `~/.config/palmar-velle/gestures.toml`.
pub fn default_path() -> PathBuf {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
        .unwrap_or_else(|| PathBuf::from(".config"));
    base.join("palmar-velle").join("gestures.toml")
}

const TEMPLATE: &str = r#"# palmar-velle custom gestures
#
# Record a pose:  palmar-velle --record my-gesture      (CLI, then exits)
#                 "Record pose" button in the GUI
#                 pkill -USR2 -x palmar-velle           (running instance)
# This file is hot-reloaded on save (or: pkill -HUP -x palmar-velle).
#
# Record mode appends a [[gesture]] block with a placeholder `notify-send`
# action. Edit the [gesture.action] table to make it do something useful.
#
# [[gesture]]
# name        = "open-terminal"
# pose        = [ ...63 floats written by record mode... ]
# threshold   = 0.97    # cosine similarity needed, (0, 1]
# cooldown_ms = 1000
# hold_frames = 3       # optional: frames the pose must be held
# repeat      = false   # optional: re-fire while held
#
# [gesture.action]      # pick ONE of:
# type = "key_combo"    # keys = ["ctrl", "alt", "t"]
# type = "mouse"        # button = "left_click" | "right_click" | "middle_click" | "double_click"
# type = "scroll"       # notches = -3   (positive = up)
# type = "shell"        # command = "firefox"
"#;

fn ensure_default_file(path: &Path) -> Result<()> {
    if !path.exists() {
        atomic_write(path, TEMPLATE)
            .with_context(|| format!("creating default {}", path.display()))?;
    }
    Ok(())
}

fn atomic_write(path: &Path, contents: &str) -> Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("toml.tmp");
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(contents.as_bytes())?;
        f.sync_all()?;
    }
    fs::rename(&tmp, path)?;
    Ok(())
}

/// Parse the file gesture-by-gesture so one bad entry doesn't discard the
/// rest. Returns (valid gestures, human-readable warnings). A syntax error in
/// the file as a whole is an `Err`.
pub fn load_file(path: &Path) -> Result<(Vec<GestureDef>, Vec<String>)> {
    let text = match fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok((vec![], vec![])),
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };

    let root: toml::Table = toml::from_str(&text)
        .with_context(|| format!("parsing {}", path.display()))?;

    let mut out: Vec<GestureDef> = Vec::new();
    let mut notes = Vec::new();

    let Some(items) = root.get("gesture") else {
        return Ok((out, notes));
    };
    let Some(items) = items.as_array() else {
        bail!("`gesture` must be an array of tables ([[gesture]])");
    };

    for (i, item) in items.iter().enumerate() {
        let def: GestureDef = match item.clone().try_into() {
            Ok(d) => d,
            Err(e) => {
                notes.push(format!("gesture #{}: {e}", i + 1));
                continue;
            }
        };
        if let Err(e) = def.validate() {
            notes.push(format!("gesture '{}': {e:#}", def.name));
            continue;
        }
        if out.iter().any(|g| g.name == def.name) {
            notes.push(format!("gesture '{}': duplicate name, ignored", def.name));
            continue;
        }
        out.push(def);
    }
    Ok((out, notes))
}

/// Append one gesture block, preserving the user's comments/formatting.
fn append_gesture(path: &Path, def: &GestureDef) -> Result<()> {
    def.validate()?;
    let (existing, _) = load_file(path).context("refusing to append to an unparsable file")?;
    if existing.iter().any(|g| g.name == def.name) {
        bail!("a gesture named '{}' already exists", def.name);
    }

    let block = toml::to_string(&ConfigFile { gestures: vec![def.clone()] })
        .context("serializing gesture")?;

    let mut text = fs::read_to_string(path).unwrap_or_default();
    if !text.is_empty() && !text.ends_with('\n') {
        text.push('\n');
    }
    text.push('\n');
    text.push_str(&block);
    atomic_write(path, &text)
}

// ---------------------------------------------------------------------------
// Hot-reload service
// ---------------------------------------------------------------------------

pub enum ConfigRequest {
    Reload,
    Append(GestureDef),
}

#[derive(Debug)]
pub enum UpdateKind {
    Initial,
    Reloaded,
    Appended(String),
    AppendFailed(String),
}

#[derive(Debug)]
pub struct ConfigUpdate {
    /// `None` when the file could not be parsed: keep the previous set.
    pub gestures: Option<Vec<GestureDef>>,
    /// Warnings (skipped entries) or the parse error text when `gestures` is None.
    pub notes: Vec<String>,
    pub kind: UpdateKind,
}

pub struct ConfigService {
    pub requests: Sender<ConfigRequest>,
    pub updates: Receiver<ConfigUpdate>,
}

fn mtime(path: &Path) -> Option<SystemTime> {
    fs::metadata(path).and_then(|m| m.modified()).ok()
}

fn build_update(path: &Path, kind: UpdateKind) -> ConfigUpdate {
    match load_file(path) {
        Ok((gestures, notes)) => ConfigUpdate { gestures: Some(gestures), notes, kind },
        Err(e) => ConfigUpdate { gestures: None, notes: vec![format!("{e:#}")], kind },
    }
}

/// Start the config thread. The initial load is delivered through `updates`
/// like any other, so the main loop needs just one code path.
pub fn spawn(path: PathBuf) -> Result<ConfigService> {
    ensure_default_file(&path)?;

    let (req_tx, req_rx) = mpsc::channel::<ConfigRequest>();
    let (upd_tx, upd_rx) = mpsc::channel::<ConfigUpdate>();

    let _ = upd_tx.send(build_update(&path, UpdateKind::Initial));

    thread::Builder::new()
        .name("config".into())
        .spawn(move || {
            let mut last = mtime(&path);
            loop {
                match req_rx.recv_timeout(Duration::from_millis(400)) {
                    Ok(ConfigRequest::Reload) => {
                        last = mtime(&path);
                        let _ = upd_tx.send(build_update(&path, UpdateKind::Reloaded));
                    }
                    Ok(ConfigRequest::Append(def)) => {
                        let name = def.name.clone();
                        let kind = match append_gesture(&path, &def) {
                            Ok(()) => UpdateKind::Appended(name),
                            Err(e) => UpdateKind::AppendFailed(format!("{e:#}")),
                        };
                        last = mtime(&path);
                        let _ = upd_tx.send(build_update(&path, kind));
                    }
                    Err(RecvTimeoutError::Timeout) => {
                        let now = mtime(&path);
                        if now != last {
                            last = now;
                            let _ = upd_tx.send(build_update(&path, UpdateKind::Reloaded));
                        }
                    }
                    Err(RecvTimeoutError::Disconnected) => break,
                }
            }
        })
        .context("spawning config thread")?;

    Ok(ConfigService { requests: req_tx, updates: upd_rx })
}
