//! Custom gesture matching and pose recording.
//!
//! Pose vector: 21 landmarks x (x, y, z) = 63 floats, translated so the wrist
//! (landmark 0) is the origin and divided by palm length (wrist -> middle
//! finger MCP, landmark 9). Matching is cosine similarity against stored
//! poses. Everything on the per-frame path is allocation-free.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::actions::Action;
use crate::config::{
    ActionSpec, GestureDef, DEFAULT_COOLDOWN_MS, DEFAULT_HOLD_FRAMES, DEFAULT_THRESHOLD,
};
use crate::landmarks::{Hand, MIDDLE_MCP, WRIST};

pub const NUM_LANDMARKS: usize = 21;
pub const POSE_DIM: usize = NUM_LANDMARKS * 3;
pub type Pose = [f32; POSE_DIM];

// ---------------------------------------------------------------------------
// Normalization + similarity
// ---------------------------------------------------------------------------

/// `aspect` = frame width / frame height. Landmark x is normalized by frame
/// width and y by frame height, so x is multiplied by `aspect` to put both
/// axes in the same unit (frame heights) before measuring lengths. z must be
/// in that same unit too (see the hand_model.rs patch).
pub fn normalize_pose(hand: &Hand, aspect: f32) -> Option<Pose> {
    if hand.landmarks.len() < NUM_LANDMARKS {
        return None;
    }
    let v = |i: usize| {
        let p = &hand.landmarks[i];
        [p.x * aspect, p.y, p.z]
    };
    let w = v(WRIST);
    let m = v(MIDDLE_MCP);
    let palm = ((m[0] - w[0]).powi(2) + (m[1] - w[1]).powi(2) + (m[2] - w[2]).powi(2)).sqrt();
    if !palm.is_finite() || palm < 1e-4 {
        return None;
    }
    let inv = 1.0 / palm;

    let mut out = [0.0f32; POSE_DIM];
    for i in 0..NUM_LANDMARKS {
        let p = v(i);
        out[i * 3] = (p[0] - w[0]) * inv;
        out[i * 3 + 1] = (p[1] - w[1]) * inv;
        out[i * 3 + 2] = (p[2] - w[2]) * inv;
    }
    Some(out)
}

#[inline]
fn dot(a: &Pose, b: &Pose) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline]
fn norm(a: &Pose) -> f32 {
    dot(a, a).sqrt()
}

pub fn cosine_similarity(a: &Pose, b: &Pose) -> f32 {
    let d = norm(a) * norm(b);
    if d < 1e-9 {
        0.0
    } else {
        dot(a, b) / d
    }
}

// ---------------------------------------------------------------------------
// Matcher
// ---------------------------------------------------------------------------

struct CompiledGesture {
    name: String,
    unit: Pose,
    threshold: f32,
    cooldown: Duration,
    hold_frames: u32,
    repeat: bool,
    action: Action,
}

#[derive(Clone, Copy, Default)]
struct GestureState {
    streak: u32,
    latched: bool,
    last_fired: Option<Instant>,
}

pub struct Fired<'a> {
    pub name: &'a str,
    pub action: &'a Action,
}

#[derive(Default)]
pub struct GestureMatcher {
    gestures: Vec<CompiledGesture>,
    states: Vec<GestureState>,
}

impl GestureMatcher {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.gestures.len()
    }

    /// Replace the gesture set (hot reload). Per-gesture state is carried
    /// over by name so reloading while holding a pose doesn't re-fire it.
    /// Returns warnings for gestures that failed to compile.
    pub fn set_gestures(&mut self, defs: &[GestureDef]) -> Vec<String> {
        let mut old: HashMap<String, GestureState> = self
            .gestures
            .drain(..)
            .zip(self.states.drain(..))
            .map(|(g, s)| (g.name, s))
            .collect();

        let mut warnings = Vec::new();
        for def in defs {
            let action = match Action::compile(&def.action) {
                Ok(a) => a,
                Err(e) => {
                    warnings.push(format!("gesture '{}': {e:#}", def.name));
                    continue;
                }
            };
            let mut pose = [0.0f32; POSE_DIM];
            for (dst, src) in pose.iter_mut().zip(def.pose.iter()) {
                *dst = *src as f32;
            }
            let n = norm(&pose);
            if n < 1e-6 {
                warnings.push(format!("gesture '{}': zero pose", def.name));
                continue;
            }
            let unit = pose.map(|v| v / n);

            self.states.push(old.remove(&def.name).unwrap_or_default());
            self.gestures.push(CompiledGesture {
                name: def.name.clone(),
                unit,
                threshold: def.threshold as f32,
                cooldown: Duration::from_millis(def.cooldown_ms),
                hold_frames: def.hold_frames.max(1),
                repeat: def.repeat,
                action,
            });
        }
        warnings
    }

    pub fn reset(&mut self) {
        for s in &mut self.states {
            s.streak = 0;
            s.latched = false;
        }
    }

    /// Feed one frame. Only the best-scoring gesture above its own threshold
    /// advances; it fires after `hold_frames` consecutive frames, outside its
    /// cooldown, and (unless `repeat`) only once per hold.
    pub fn update(&mut self, pose: Option<&Pose>, now: Instant) -> Option<Fired<'_>> {
        let Some(pose) = pose else {
            self.reset();
            return None;
        };
        let n = norm(pose);
        if n < 1e-6 {
            self.reset();
            return None;
        }
        let inv = 1.0 / n;

        let mut best: Option<(usize, f32)> = None;
        for (i, g) in self.gestures.iter().enumerate() {
            let sim = dot(pose, &g.unit) * inv;
            if sim >= g.threshold && best.map_or(true, |(_, s)| sim > s) {
                best = Some((i, sim));
            }
        }
        let best_idx = best.map(|(i, _)| i);

        for (i, st) in self.states.iter_mut().enumerate() {
            if Some(i) == best_idx {
                st.streak = st.streak.saturating_add(1);
            } else {
                st.streak = 0;
                st.latched = false;
            }
        }

        let i = best_idx?;
        let g = &self.gestures[i];
        let st = &mut self.states[i];
        let cooled = st
            .last_fired
            .map_or(true, |t| now.duration_since(t) >= g.cooldown);
        if st.streak >= g.hold_frames && cooled && (g.repeat || !st.latched) {
            st.latched = true;
            st.last_fired = Some(now);
            return Some(Fired { name: &g.name, action: &g.action });
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Recorder
// ---------------------------------------------------------------------------

pub const RECORD_FRAMES: usize = 30;
const COUNTDOWN: Duration = Duration::from_secs(3);
const CAPTURE_TIMEOUT: Duration = Duration::from_secs(6);
const MIN_STABILITY: f32 = 0.90;

pub enum RecordStep {
    Idle,
    Countdown(u64),
    Progress { got: usize, total: usize },
    /// Ok((gesture, average similarity of frames to the mean pose))
    Finished(Result<(GestureDef, f32), String>),
}

/// Countdown, then average `RECORD_FRAMES` normalized poses.
pub struct PoseRecorder {
    name: String,
    capture_from: Instant,
    give_up_at: Instant,
    frames: Vec<Pose>,
    last_countdown: u64,
}

impl PoseRecorder {
    pub fn new(name: String, now: Instant) -> Self {
        Self {
            name,
            capture_from: now + COUNTDOWN,
            give_up_at: now + COUNTDOWN + CAPTURE_TIMEOUT,
            frames: Vec::with_capacity(RECORD_FRAMES),
            last_countdown: 0,
        }
    }

    pub fn feed(&mut self, now: Instant, pose: Option<&Pose>) -> RecordStep {
        if now < self.capture_from {
            let left = (self.capture_from - now).as_secs_f32().ceil() as u64;
            if left != self.last_countdown {
                self.last_countdown = left;
                return RecordStep::Countdown(left);
            }
            return RecordStep::Idle;
        }
        if now >= self.give_up_at {
            return RecordStep::Finished(Err(
                "hand not visible long enough; keep it in frame and retry".into(),
            ));
        }
        if let Some(p) = pose {
            self.frames.push(*p);
            let got = self.frames.len();
            if got >= RECORD_FRAMES {
                return RecordStep::Finished(self.finish());
            }
            if got % 5 == 0 {
                return RecordStep::Progress { got, total: RECORD_FRAMES };
            }
        }
        RecordStep::Idle
    }

    fn finish(&self) -> Result<(GestureDef, f32), String> {
        let n = self.frames.len() as f32;
        let mut mean = [0.0f32; POSE_DIM];
        for f in &self.frames {
            for (m, v) in mean.iter_mut().zip(f.iter()) {
                *m += *v;
            }
        }
        for m in mean.iter_mut() {
            *m /= n;
        }
        if norm(&mean) < 1e-6 {
            return Err("degenerate pose".into());
        }

        let stability =
            self.frames.iter().map(|f| cosine_similarity(f, &mean)).sum::<f32>() / n;
        if stability < MIN_STABILITY {
            return Err(format!(
                "pose was unstable (similarity {stability:.3} < {MIN_STABILITY}); hold still and retry"
            ));
        }

        let def = GestureDef {
            name: self.name.clone(),
            // Round so the TOML stays readable.
            pose: mean.iter().map(|v| ((*v as f64) * 1e5).round() / 1e5).collect(),
            threshold: DEFAULT_THRESHOLD,
            cooldown_ms: DEFAULT_COOLDOWN_MS,
            hold_frames: DEFAULT_HOLD_FRAMES,
            repeat: false,
            // Placeholder so a fresh recording visibly does something.
            // Names are restricted to [A-Za-z0-9_-], so this is quote-safe.
            action: ActionSpec::Shell {
                command: format!("notify-send 'palmar-velle' 'Gesture {} triggered'", self.name),
            },
        };
        Ok((def, stability))
    }
}
