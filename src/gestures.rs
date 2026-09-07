//! Gesture recognizer for webcam-based mouse control.
//!
//! Gestures:
//!   - Palm center position -> cursor movement
//!   - Thumb + index pinch -> primary click / drag
//!   - Thumb + middle pinch -> secondary click
//!   - Thumb + ring pinch -> vertical scrolling
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

    // Scrolling state.
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
        // Cursor movement (Palm Center Centroid)
        // ------------------------------------------------------------
        let wrist = &hand.landmarks[0];     // WRIST
        let index_mcp = &hand.landmarks[5]; // INDEX_FINGER_MCP
        let pinky_mcp = &hand.landmarks[17]; // PINKY_MCP

        let palm_x = (wrist.x + index_mcp.x + pinky_mcp.x) / 3.0;
        let palm_y = (wrist.y + index_mcp.y + pinky_mcp.y) / 3.0;

        events.push(GestureEvent::Move {
            x: palm_x.clamp(0.0, 1.0),
            y: palm_y.clamp(0.0, 1.0),
        });

        // ------------------------------------------------------------
        // Pinch distances
        // ------------------------------------------------------------

        let thumb_index = thumb.dist(index);
        let thumb_middle = thumb.dist(middle);

        // Small threshold = pinch.
        const PINCH_ENTER: f32 = 0.10;
        const PINCH_EXIT: f32 = 0.14;

        let index_pinched = thumb_index < PINCH_ENTER;
        let index_released = thumb_index > PINCH_EXIT;

        let middle_pinched = thumb_middle < PINCH_ENTER;
        let middle_released = thumb_middle > PINCH_EXIT;

        let scroll_pinched = index_pinched && middle_pinched;

        // ------------------------------------------------------------
        // Thumb + Index + Middle pinch = scroll
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

        // Ring pinch ended.
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