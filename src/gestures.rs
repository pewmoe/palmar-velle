//! Gesture recognizer for webcam-based mouse control.
//!
//! Gestures:
//!   - Palm center position -> cursor movement
//!   - Thumb + index pinch -> primary click / drag
//!   - Thumb + middle pinch -> secondary click
//!   - Thumb + index + middle pinch -> vertical scrolling
//!   - Two quick index pinches -> double click
//!
//! Pinch distances are normalized by hand scale (wrist-to-index-knuckle
//! distance), not raw frame-fraction distance, so the same physical pinch
//! triggers consistently whether your hand is close to or far from the
//! camera. Pinch state changes also require a few consecutive confirming
//! frames before firing, to reject single-frame landmark jitter.

use crate::landmarks::{Hand, INDEX_TIP, MIDDLE_TIP, THUMB_TIP};

#[derive(Debug, Clone, Copy)]
pub enum GestureEvent {
    Move { x: f32, y: f32 },

    PrimaryDown,
    PrimaryUp,

    SecondaryDown,
    SecondaryUp,

    Scroll { dy: f32 },

    DoubleClick,

    HandLost,
}

/// Tracks a boolean pinch/no-pinch state with hysteresis (different enter
/// vs exit thresholds) plus a frame-count debounce, so a state change only
/// fires once the new condition has been true for several consecutive
/// frames in a row -- not just one noisy frame.
struct DebouncedPinch {
    is_pinched: bool,
    enter_streak: u32,
    exit_streak: u32,
}

impl DebouncedPinch {
    fn new() -> Self {
        Self {
            is_pinched: false,
            enter_streak: 0,
            exit_streak: 0,
        }
    }

    /// `raw_pinched` / `raw_released` are this frame's instantaneous
    /// threshold checks (with hysteresis already applied at the caller).
    /// Returns the debounced state for this frame.
    fn update(&mut self, raw_pinched: bool, raw_released: bool, confirm_frames: u32) -> bool {
        if !self.is_pinched {
            if raw_pinched {
                self.enter_streak += 1;
                if self.enter_streak >= confirm_frames {
                    self.is_pinched = true;
                    self.enter_streak = 0;
                    self.exit_streak = 0;
                }
            } else {
                self.enter_streak = 0;
            }
        } else {
            if raw_released {
                self.exit_streak += 1;
                if self.exit_streak >= confirm_frames {
                    self.is_pinched = false;
                    self.exit_streak = 0;
                    self.enter_streak = 0;
                }
            } else {
                self.exit_streak = 0;
            }
        }

        self.is_pinched
    }

    fn reset(&mut self) {
        self.is_pinched = false;
        self.enter_streak = 0;
        self.exit_streak = 0;
    }
}

pub struct GestureRecognizer {
    is_primary_down: bool,
    is_secondary_down: bool,

    is_scrolling: bool,
    last_scroll_y: Option<f32>,

    last_primary_release_ms: Option<u64>,

    index_pinch: DebouncedPinch,
    middle_pinch: DebouncedPinch,

    start_time: std::time::Instant,
}

// Number of consecutive frames a pinch condition must hold before it's
// treated as a real state change. At ~30fps this is roughly 65-100ms --
// enough to reject single-frame landmark jitter without adding
// perceptible input lag.
const PINCH_CONFIRM_FRAMES: u32 = 2;

// Pinch thresholds as a RATIO of hand scale (wrist-to-index-knuckle
// distance), not raw frame-normalized distance. This makes the threshold
// invariant to how close your hand is to the camera. Tune these two if
// pinches feel too easy/hard to trigger; keep enter < exit for hysteresis.
const PINCH_ENTER_RATIO: f32 = 0.45;
const PINCH_EXIT_RATIO: f32 = 0.65;

impl GestureRecognizer {
    pub fn new() -> Self {
        Self {
            is_primary_down: false,
            is_secondary_down: false,

            is_scrolling: false,
            last_scroll_y: None,

            last_primary_release_ms: None,

            index_pinch: DebouncedPinch::new(),
            middle_pinch: DebouncedPinch::new(),

            start_time: std::time::Instant::now(),
        }
    }

    fn now_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }

    pub fn process(&mut self, hand_option: Option<&Hand>) -> Vec<GestureEvent> {
        let mut events = Vec::new();

        let hand = match hand_option {
            Some(hand) => hand,

            None => {
                // Never leave a mouse button physically held when
                // the camera loses the hand.
                if self.is_primary_down {
                    self.is_primary_down = false;
                    events.push(GestureEvent::PrimaryUp);
                }

                if self.is_secondary_down {
                    self.is_secondary_down = false;
                    events.push(GestureEvent::SecondaryUp);
                }

                self.is_scrolling = false;
                self.last_scroll_y = None;

                self.index_pinch.reset();
                self.middle_pinch.reset();

                events.push(GestureEvent::HandLost);

                return events;
            }
        };

        let thumb = &hand.landmarks[THUMB_TIP];
        let index = &hand.landmarks[INDEX_TIP];
        let middle = &hand.landmarks[MIDDLE_TIP];

        // ------------------------------------------------------------
        // Cursor movement (palm center centroid)
        // ------------------------------------------------------------
        let wrist = &hand.landmarks[0]; // WRIST
        let index_mcp = &hand.landmarks[5]; // INDEX_FINGER_MCP
        let pinky_mcp = &hand.landmarks[17]; // PINKY_MCP

        let palm_x = (wrist.x + index_mcp.x + pinky_mcp.x) / 3.0;
        let palm_y = (wrist.y + index_mcp.y + pinky_mcp.y) / 3.0;

        events.push(GestureEvent::Move {
            x: palm_x.clamp(0.0, 1.0),
            y: palm_y.clamp(0.0, 1.0),
        });

        // ------------------------------------------------------------
        // Hand scale reference: wrist-to-index-knuckle distance stays
        // roughly constant for a given hand regardless of pinch state,
        // and scales with how close the hand is to the camera -- so
        // dividing pinch distances by it makes the threshold
        // distance-invariant. Floor it to avoid divide-by-near-zero on
        // a bad frame.
        // ------------------------------------------------------------
        let hand_scale = wrist.dist(index_mcp).max(0.02);

        // ------------------------------------------------------------
        // Pinch ratios (distance-invariant)
        // ------------------------------------------------------------

        let thumb_index_ratio = thumb.dist(index) / hand_scale;
        let thumb_middle_ratio = thumb.dist(middle) / hand_scale;

        let index_raw_pinched = thumb_index_ratio < PINCH_ENTER_RATIO;
        let index_raw_released = thumb_index_ratio > PINCH_EXIT_RATIO;

        let middle_raw_pinched = thumb_middle_ratio < PINCH_ENTER_RATIO;
        let middle_raw_released = thumb_middle_ratio > PINCH_EXIT_RATIO;

        let index_pinched =
            self.index_pinch
                .update(index_raw_pinched, index_raw_released, PINCH_CONFIRM_FRAMES);
        let middle_pinched =
            self.middle_pinch
                .update(middle_raw_pinched, middle_raw_released, PINCH_CONFIRM_FRAMES);

        let index_released = !index_pinched;
        let middle_released = !middle_pinched;

        let scroll_pinched = index_pinched && middle_pinched;

        // ------------------------------------------------------------
        // Thumb + index + middle pinch = scroll
        // ------------------------------------------------------------
        //
        // While scrolling, normal left/right clicks are suppressed.
        // ------------------------------------------------------------

        if scroll_pinched {
            if !self.is_scrolling {
                self.is_scrolling = true;
                self.last_scroll_y = Some(palm_y);
            } else if let Some(last_y) = self.last_scroll_y {
                let dy = palm_y - last_y;

                const SCROLL_DEADZONE: f32 = 0.0025;

                if dy.abs() > SCROLL_DEADZONE {
                    events.push(GestureEvent::Scroll { dy });
                }

                self.last_scroll_y = Some(palm_y);
            }

            // If a button was held when the scroll pinch began,
            // release it before entering scroll mode.
            if self.is_primary_down {
                self.is_primary_down = false;
                events.push(GestureEvent::PrimaryUp);
            }

            if self.is_secondary_down {
                self.is_secondary_down = false;
                events.push(GestureEvent::SecondaryUp);
            }

            return events;
        }

        if self.is_scrolling {
            self.is_scrolling = false;
            self.last_scroll_y = None;
        }

        // ------------------------------------------------------------
        // Primary click / drag
        // ------------------------------------------------------------

        // Only allow primary pinch if the middle finger is NOT
        // simultaneously pinched.
        if index_pinched && !middle_pinched && !self.is_primary_down {
            self.is_primary_down = true;

            let now = self.now_ms();

            // Two primary pinch activations close together are a
            // double click.
            if let Some(previous_release) = self.last_primary_release_ms {
                if now.saturating_sub(previous_release) <= 350 {
                    self.is_primary_down = false;
                    events.push(GestureEvent::DoubleClick);
                } else {
                    events.push(GestureEvent::PrimaryDown);
                }
            } else {
                events.push(GestureEvent::PrimaryDown);
            }
        } else if index_released && self.is_primary_down {
            self.is_primary_down = false;

            self.last_primary_release_ms = Some(self.now_ms());

            events.push(GestureEvent::PrimaryUp);
        }

        // ------------------------------------------------------------
        // Secondary click / drag
        // ------------------------------------------------------------

        // Only allow secondary pinch when index isn't also pinched.
        if middle_pinched && !index_pinched && !self.is_secondary_down {
            self.is_secondary_down = true;
            events.push(GestureEvent::SecondaryDown);
        } else if middle_released && self.is_secondary_down {
            self.is_secondary_down = false;
            events.push(GestureEvent::SecondaryUp);
        }

        events
    }
}