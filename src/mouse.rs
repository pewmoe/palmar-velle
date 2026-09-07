use anyhow::Result;
use uinput::event::absolute::{Absolute, Position};
use uinput::event::Event;

pub struct VirtualMouse {
    device: uinput::Device,
    last_x: Option<f32>,
    last_y: Option<f32>,
}

impl VirtualMouse {
    pub fn new() -> Result<Self> {
        let device = uinput::default()?
            .name("Gesture Control Virtual Mouse")?

            .event(Event::Controller(uinput::event::Controller::Mouse(uinput::event::controller::Mouse::Left)))?
            .event(Event::Controller(uinput::event::Controller::Mouse(uinput::event::controller::Mouse::Right)))?

            .event(Event::Absolute(
                Absolute::Position(Position::X),
            ))?
            .min(0)
            .max(65535)

            .event(Event::Absolute(
                Absolute::Position(Position::Y),
            ))?
            .min(0)
            .max(65535)

            .event(uinput::event::Relative::Wheel(uinput::event::relative::Wheel::Vertical))?

            .create()?;

        std::thread::sleep(
            std::time::Duration::from_millis(100)
        );

        Ok(Self { 
            device,
            last_x: None,
            last_y: None,
        })
    }

    pub fn move_to(
        &mut self,
        x: f32,
        y: f32,
    ) -> Result<()> {
        // Mirror the X coordinate so moving right on camera moves cursor right on screen
        let x = 1.0 - x.clamp(0.0, 1.0);
        let y = y.clamp(0.0, 1.0);

        // Smoothing and deadzone configuration
        const SMOOTHING: f32 = 0.3; // Responsive yet smooth
        const DEADZONE: f32 = 0.0015; // Eliminates micro-jitter when holding still

        let (target_x, target_y) = match (self.last_x, self.last_y) {
            (Some(lx), Some(ly)) => {
                let dx = (x - lx).abs();
                let dy = (y - ly).abs();

                if dx < DEADZONE && dy < DEADZONE {
                    return Ok(());
                }

                let sx = lx + (x - lx) * SMOOTHING;
                let sy = ly + (y - ly) * SMOOTHING;
                (sx, sy)
            }
            _ => (x, y), 
        };

        self.last_x = Some(target_x);
        self.last_y = Some(target_y);

        let abs_x = (target_x * 65535.0) as i32;
        let abs_y = (target_y * 65535.0) as i32;

        self.device.send(
            Absolute::Position(Position::X),
            abs_x,
        )?;

        self.device.send(
            Absolute::Position(Position::Y),
            abs_y,
        )?;

        self.device.synchronize()?;

        Ok(())
    }

    pub fn left_button(&mut self, pressed: bool) -> Result<()> {
        let value = if pressed { 1 } else { 0 };

        self.device.send(
            Event::Controller(uinput::event::Controller::Mouse(uinput::event::controller::Mouse::Left)),
            value,
        )?;

        self.device.synchronize()?;

        Ok(())
    }

    pub fn right_button(&mut self, pressed: bool) -> Result<()> {
        let value = if pressed { 1 } else { 0 };

        self.device.send(
            Event::Controller(uinput::event::Controller::Mouse(uinput::event::controller::Mouse::Right)),
            value,
        )?;

        self.device.synchronize()?;

        Ok(())
    }

    pub fn double_click(&mut self) -> Result<()> {
        self.left_button(true)?;
        self.left_button(false)?;

        std::thread::sleep(
            std::time::Duration::from_millis(60)
        );

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
}the scroll gesture is effectively dead for two separate reasons.

The recognizer is checking the wrong finger combination
In src/gestures.rs, the scroll branch is:
let ring = &hand.landmarks[16];
let ring_pinched = thumb_ring < PINCH_ENTER;
then if ring_pinched { ... emit Scroll ... }
So the code scrolls on “thumb + ring pinch”, not “thumb + index + middle”.
That conflicts with the UI text in README.md, which says “Thumb + Index + Middle = Vertical Scroll”.
In practice, the ring finger is often much less stable/less easy to curl than index/middle, so this gesture is easy to miss.
Even when the scroll event fires, the value sent to the OS is almost always zero
In src/mouse.rs:
let value = dy as i32;
But dy from src/gestures.rs is a normalized palm movement delta:
dy = palm_y - last_y
and palm_y is in the range 0.0..=1.0
That means dy is a tiny fraction like 0.01, 0.002, etc.
Casting that to i32 truncates it to 0 immediately.
So the system receives a wheel event with value 0, which does nothing.
That means even if the detector succeeds, the final stage kills scrolling before Linux ever sees it.

What to fix:

Make the gesture match the intended behavior:
either change the scroll trigger to a thumb+index+middle pinch, or intentionally keep thumb+ring if that’s your chosen gesture.
Scale the wheel delta to a real integer wheel step before sending:
e.g. let value = (dy * 120.0).round() as i32;
or use a discrete step like 1 / -1 when the delta exceeds threshold
Also increase the scroll deadzone or threshold if needed, because the current SCROLL_DEADZONE is tiny and the extracted dy is small.
So the root cause is not just “bad camera tracking”; it’s largely this combination:

wrong finger used for scroll
and zero-valued wheel events sent to uinput
