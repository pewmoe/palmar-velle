use anyhow::Result;
use std::time::Instant;
use uinput::event::relative::{Position as RelPosition, Relative};
use uinput::event::Event;

struct OneEuroFilter {
    min_cutoff: f32,
    beta: f32,
    d_cutoff: f32,
    x_prev: Option<f32>,
    dx_prev: f32,
    t_prev: Option<Instant>,
}

impl OneEuroFilter {
    fn new(min_cutoff: f32, beta: f32) -> Self {
        Self {
            min_cutoff,
            beta,
            d_cutoff: 1.0,
            x_prev: None,
            dx_prev: 0.0,
            t_prev: None,
        }
    }

    fn filter(&mut self, x: f32, t: Instant) -> f32 {
        if let (Some(x_prev), Some(t_prev)) = (self.x_prev, self.t_prev) {
            let dt = (t - t_prev).as_secs_f32().max(1e-5);
            let dx = (x - x_prev) / dt;

            let alpha_d = self.alpha(dt, self.d_cutoff);
            let dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev;

            let cutoff = self.min_cutoff + self.beta * dx_hat.abs();
            let alpha = self.alpha(dt, cutoff);

            let x_hat = alpha * x + (1.0 - alpha) * x_prev;

            self.x_prev = Some(x_hat);
            self.dx_prev = dx_hat;
            self.t_prev = Some(t);
            x_hat
        } else {
            self.x_prev = Some(x);
            self.t_prev = Some(t);
            x
        }
    }

    fn alpha(&self, dt: f32, cutoff: f32) -> f32 {
        let tau = 1.0 / (2.0 * std::f32::consts::PI * cutoff);
        1.0 / (1.0 + tau / dt)
    }
}

pub struct VirtualMouse {
    device: uinput::Device,
    filter_x: OneEuroFilter,
    filter_y: OneEuroFilter,
    // Last filtered (not raw) position, used to compute a relative delta.
    // None right after creation or right after a HandLost reset, so the
    // very next move_to() only primes state instead of emitting a
    // potentially huge jump.
    last_pos: Option<(f32, f32)>,
}

impl VirtualMouse {
    pub fn new() -> Result<Self> {
        // NOTE: this device intentionally uses REL_X/REL_Y (relative motion),
        // not ABS_X/ABS_Y. A uinput device advertising ABS_X/ABS_Y plus
        // BTN_LEFT (== BTN_MOUSE, same kernel code) with no BTN_TOUCH /
        // BTN_TOOL_PEN gets classified by udev/libinput as a graphics
        // tablet, not a pointer -- so compositors silently drop every
        // position event. REL_X/REL_Y + BTN_LEFT is the unambiguous
        // signature of a standard mouse and just works everywhere,
        // including Wayland.
        let device = uinput::default()?
            .name("Gesture Control Virtual Mouse")?
            .event(Event::Controller(uinput::event::Controller::Mouse(
                uinput::event::controller::Mouse::Left,
            )))?
            .event(Event::Controller(uinput::event::Controller::Mouse(
                uinput::event::controller::Mouse::Right,
            )))?
            .event(Event::Relative(Relative::Position(RelPosition::X)))?
            .event(Event::Relative(Relative::Position(RelPosition::Y)))?
            .event(uinput::event::Relative::Wheel(
                uinput::event::relative::Wheel::Vertical,
            ))?
            .create()?;

        std::thread::sleep(std::time::Duration::from_millis(100));

        Ok(Self {
            device,
            filter_x: OneEuroFilter::new(0.8, 0.02),
            filter_y: OneEuroFilter::new(0.8, 0.02),
            last_pos: None,
        })
    }

    /// x, y are normalized [0.0, 1.0] hand position (as before). Internally
    /// this now converts the smoothed absolute target into a relative delta
    /// before sending it to the kernel.
    pub fn move_to(&mut self, x: f32, y: f32) -> Result<()> {
        let x = 1.0 - x.clamp(0.0, 1.0);
        let y = y.clamp(0.0, 1.0);

        let now = Instant::now();
        let target_x = self.filter_x.filter(x, now);
        let target_y = self.filter_y.filter(y, now);

        if let Some((last_x, last_y)) = self.last_pos {
            // Tune this to taste -- higher = faster cursor travel per unit
            // of normalized hand movement. 65535 was the old ABS max; this
            // scale gives roughly comparable full-screen travel distance.
            const SENSITIVITY: f32 = 3000.0;

            let dx = ((target_x - last_x) * SENSITIVITY) as i32;
            let dy = ((target_y - last_y) * SENSITIVITY) as i32;

            if dx != 0 {
                self.device.send(RelPosition::X, dx)?;
            }
            if dy != 0 {
                self.device.send(RelPosition::Y, dy)?;
            }
            if dx != 0 || dy != 0 {
                self.device.synchronize()?;
            }
        }

        self.last_pos = Some((target_x, target_y));

        Ok(())
    }

    /// Call this whenever the hand is lost (GestureEvent::HandLost) so that
    /// the next detected position doesn't produce one huge delta jump (or a
    /// stale drag) when the hand reappears somewhere else in frame.
    pub fn reset(&mut self) {
        self.last_pos = None;
        self.filter_x = OneEuroFilter::new(0.8, 0.02);
        self.filter_y = OneEuroFilter::new(0.8, 0.02);
    }

    pub fn left_button(&mut self, pressed: bool) -> Result<()> {
        let value = if pressed { 1 } else { 0 };

        self.device.send(
            Event::Controller(uinput::event::Controller::Mouse(
                uinput::event::controller::Mouse::Left,
            )),
            value,
        )?;

        self.device.synchronize()?;

        Ok(())
    }

    pub fn right_button(&mut self, pressed: bool) -> Result<()> {
        let value = if pressed { 1 } else { 0 };

        self.device.send(
            Event::Controller(uinput::event::Controller::Mouse(
                uinput::event::controller::Mouse::Right,
            )),
            value,
        )?;

        self.device.synchronize()?;

        Ok(())
    }

    pub fn double_click(&mut self) -> Result<()> {
        self.left_button(true)?;
        self.left_button(false)?;

        std::thread::sleep(std::time::Duration::from_millis(60));

        self.left_button(true)?;
        self.left_button(false)?;

        Ok(())
    }

    pub fn scroll(&mut self, dy: f32) -> Result<()> {
        let value = (dy * 120.0).clamp(-120.0, 120.0).round() as i32;
        if value == 0 {
            return Ok(());
        }

        self.device.send(
            uinput::event::Relative::Wheel(uinput::event::relative::Wheel::Vertical),
            value,
        )?;

        self.device.synchronize()?;

        Ok(())
    }
}