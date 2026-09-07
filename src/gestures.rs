//! Gesture recognizer for webcam-based mouse control.
//!
//! Gestures:
//!   - Index finger position -> cursor movement
//!   - Thumb + index pinch -> primary click / drag
//!   - Thumb + middle pinch -> secondary click
//!   - Thumb + index + middle pinch -> vertical scrolling
//!   - Two quick index pinches -> double click
//!
//! All distances are normalized landmark distances.

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

pub struct GestureRecognizer {
    is_primary_down: bool,
    is_secondary_down: bool,

    // Three-finger scrolling state.
    is_scrolling: bool,
    last_scroll_y: Option<f32>,

    // Used for double-click detection.
    last_primary_release_ms: Option<u64>,

    // Monotonic timestamp supplied internally.
    start_time: std::time::Instant,
}

impl GestureRecognizer {
    pub fn new() -> Self {
        Self {
            is_primary_down: false,
            is_secondary_down: false,

            is_scrolling: false,
            last_scroll_y: None,

            last_primary_release_ms: None,

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

                events.push(GestureEvent::HandLost);

                return events;
            }
        };

        let thumb = &hand.landmarks[THUMB_TIP];
        let index = &hand.landmarks[INDEX_TIP];
        let middle = &hand.landmarks[MIDDLE_TIP];

        // ------------------------------------------------------------
        // Cursor movement
        // ------------------------------------------------------------

        events.push(GestureEvent::Move {
            x: index.x.clamp(0.0, 1.0),
            y: index.y.clamp(0.0, 1.0),
        });

        // ------------------------------------------------------------
        // Pinch distances
        // ------------------------------------------------------------

        let thumb_index = thumb.dist(index);
        let thumb_middle = thumb.dist(middle);

        // Small threshold = pinch.
        //
        // Larger exit threshold gives us hysteresis and prevents
        // button flickering when the fingertips hover around the
        // threshold.
        const PINCH_ENTER: f32 = 0.08;
        const PINCH_EXIT: f32 = 0.12;

        let index_pinched = thumb_index < PINCH_ENTER;
        let index_released = thumb_index > PINCH_EXIT;

        let middle_pinched = thumb_middle < PINCH_ENTER;
        let middle_released = thumb_middle > PINCH_EXIT;

        // ------------------------------------------------------------
        // Three-finger pinch = scroll
        // ------------------------------------------------------------
        //
        // Both index and middle must be touching the thumb.
        //
        // While scrolling, normal left/right clicks are suppressed.
        // ------------------------------------------------------------

        let three_finger_pinch = index_pinched && middle_pinched;

        if three_finger_pinch {
            if !self.is_scrolling {
                self.is_scrolling = true;
                self.last_scroll_y = Some(index.y);
            } else if let Some(last_y) = self.last_scroll_y {
                let dy = index.y - last_y;

                // Ignore microscopic camera noise.
                const SCROLL_DEADZONE: f32 = 0.003;

                if dy.abs() > SCROLL_DEADZONE {
                    events.push(GestureEvent::Scroll {
                        dy,
                    });
                }

                self.last_scroll_y = Some(index.y);
            }

            // If a button was held when the three-finger pinch began,
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

        // Three-finger pinch ended.
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