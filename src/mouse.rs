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
}

