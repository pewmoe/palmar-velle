//! Executes custom-gesture actions.
//!
//! Hot-path rules:
//!   * key chords / mouse buttons / wheel are a handful of non-blocking
//!     uinput writes, no sleeps;
//!   * shell commands are handed to a worker thread through an unbounded
//!     channel (`send` never blocks) and spawned there.

use anyhow::{bail, Result};
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Sender};
use std::thread;
use std::time::Duration;
use uinput::event::keyboard::Key;

use crate::config::{ActionSpec, MouseButtonAction};
use crate::mouse::VirtualMouse;

/// An `ActionSpec` with key names already resolved, so firing it does no
/// parsing.
pub enum Action {
    Keys(Vec<Key>),
    Mouse(MouseButtonAction),
    Scroll(i32),
    Shell(String),
}

impl Action {
    pub fn compile(spec: &ActionSpec) -> Result<Self> {
        Ok(match spec {
            ActionSpec::KeyCombo { keys } => {
                let mut resolved = Vec::with_capacity(keys.len());
                for k in keys {
                    match parse_key(k) {
                        Some(key) => resolved.push(key),
                        None => bail!("unknown key '{k}'"),
                    }
                }
                Action::Keys(resolved)
            }
            ActionSpec::Mouse { button } => Action::Mouse(*button),
            ActionSpec::Scroll { notches } => Action::Scroll(*notches),
            ActionSpec::Shell { command } => Action::Shell(command.clone()),
        })
    }
}

/// Key-name table used in gestures.toml (case-insensitive).
///
/// If rustc rejects a variant name here, adjust this one table to match the
/// `uinput::event::keyboard::Key` enum of the uinput version you pinned.
pub fn parse_key(name: &str) -> Option<Key> {
    let n = name.trim().to_ascii_lowercase();
    Some(match n.as_str() {
        // modifiers
        "ctrl" | "control" | "leftctrl" | "lctrl" => Key::LeftControl,
        "rightctrl" | "rctrl" => Key::RightControl,
        "shift" | "leftshift" | "lshift" => Key::LeftShift,
        "rightshift" | "rshift" => Key::RightShift,
        "alt" | "leftalt" | "lalt" => Key::LeftAlt,
        "altgr" | "rightalt" | "ralt" => Key::RightAlt,
        "super" | "meta" | "win" | "logo" | "leftmeta" => Key::LeftMeta,
        "rightmeta" | "rightsuper" => Key::RightMeta,

        // editing / navigation
        "esc" | "escape" => Key::Esc,
        "enter" | "return" => Key::Enter,
        "space" => Key::Space,
        "tab" => Key::Tab,
        "backspace" => Key::BackSpace,
        "delete" | "del" => Key::Delete,
        "insert" | "ins" => Key::Insert,
        "home" => Key::Home,
        "end" => Key::End,
        "pageup" | "pgup" => Key::PageUp,
        "pagedown" | "pgdn" => Key::PageDown,
        "up" => Key::Up,
        "down" => Key::Down,
        "left" => Key::Left,
        "right" => Key::Right,
        "capslock" => Key::CapsLock,

        // punctuation
        "minus" => Key::Minus,
        "equal" => Key::Equal,
        "comma" => Key::Comma,
        "dot" | "period" => Key::Dot,
        "slash" => Key::Slash,
        "backslash" => Key::BackSlash,
        "semicolon" => Key::SemiColon,
        "apostrophe" => Key::Apostrophe,
        "grave" | "backtick" => Key::Grave,
        "leftbrace" => Key::LeftBrace,
        "rightbrace" => Key::RightBrace,

        // function keys
        "f1" => Key::F1,
        "f2" => Key::F2,
        "f3" => Key::F3,
        "f4" => Key::F4,
        "f5" => Key::F5,
        "f6" => Key::F6,
        "f7" => Key::F7,
        "f8" => Key::F8,
        "f9" => Key::F9,
        "f10" => Key::F10,
        "f11" => Key::F11,
        "f12" => Key::F12,

        // letters
        "a" => Key::A,
        "b" => Key::B,
        "c" => Key::C,
        "d" => Key::D,
        "e" => Key::E,
        "f" => Key::F,
        "g" => Key::G,
        "h" => Key::H,
        "i" => Key::I,
        "j" => Key::J,
        "k" => Key::K,
        "l" => Key::L,
        "m" => Key::M,
        "n" => Key::N,
        "o" => Key::O,
        "p" => Key::P,
        "q" => Key::Q,
        "r" => Key::R,
        "s" => Key::S,
        "t" => Key::T,
        "u" => Key::U,
        "v" => Key::V,
        "w" => Key::W,
        "x" => Key::X,
        "y" => Key::Y,
        "z" => Key::Z,

        // digits
        "0" => Key::_0,
        "1" => Key::_1,
        "2" => Key::_2,
        "3" => Key::_3,
        "4" => Key::_4,
        "5" => Key::_5,
        "6" => Key::_6,
        "7" => Key::_7,
        "8" => Key::_8,
        "9" => Key::_9,

        _ => return None,
    })
}

/// Separate uinput device so the existing mouse device keeps its
/// unambiguous "pointer" classification in udev/libinput.
struct VirtualKeyboard {
    device: uinput::Device,
}

impl VirtualKeyboard {
    fn new() -> Result<Self> {
        let device = uinput::default()?
            .name("Palmar Velle Virtual Keyboard")?
            .event(uinput::event::Keyboard::All)?
            .create()?;
        // Give the compositor time to register the device. Startup only.
        thread::sleep(Duration::from_millis(100));
        Ok(Self { device })
    }

    /// Press in order, release in reverse order.
    fn chord(&mut self, keys: &[Key]) -> Result<()> {
        for k in keys {
            self.device.press(k)?;
        }
        self.device.synchronize()?;
        for k in keys.iter().rev() {
            self.device.release(k)?;
        }
        self.device.synchronize()?;
        Ok(())
    }
}

pub struct ActionRunner {
    keyboard: VirtualKeyboard,
    shell_tx: Sender<(String, String)>,
}

impl ActionRunner {
    pub fn new() -> Result<Self> {
        let keyboard = VirtualKeyboard::new()?;

        let (shell_tx, shell_rx) = mpsc::channel::<(String, String)>();
        thread::Builder::new()
            .name("shell-actions".into())
            .spawn(move || {
                for (gesture, command) in shell_rx {
                    let spawned = Command::new("sh")
                        .arg("-c")
                        .arg(&command)
                        .env("PALMAR_VELLE_GESTURE", &gesture)
                        .stdin(Stdio::null())
                        .stdout(Stdio::null())
                        .stderr(Stdio::null())
                        .spawn();
                    match spawned {
                        // Reap in a throwaway thread so a long-running command
                        // never delays the next one.
                        Ok(mut child) => {
                            thread::spawn(move || {
                                let _ = child.wait();
                            });
                        }
                        Err(e) => log::warn!("gesture '{gesture}': failed to spawn command: {e}"),
                    }
                }
            })?;

        Ok(Self { keyboard, shell_tx })
    }

    pub fn run(&mut self, gesture: &str, action: &Action, mouse: &mut VirtualMouse) -> Result<()> {
        match action {
            Action::Keys(keys) => self.keyboard.chord(keys)?,
            Action::Mouse(kind) => match kind {
                MouseButtonAction::LeftClick => {
                    mouse.left_button(true)?;
                    mouse.left_button(false)?;
                }
                MouseButtonAction::RightClick => {
                    mouse.right_button(true)?;
                    mouse.right_button(false)?;
                }
                MouseButtonAction::MiddleClick => {
                    mouse.middle_button(true)?;
                    mouse.middle_button(false)?;
                }
                // Four back-to-back events, no sleep: toolkits detect
                // double-clicks from event timestamps.
                MouseButtonAction::DoubleClick => {
                    for _ in 0..2 {
                        mouse.left_button(true)?;
                        mouse.left_button(false)?;
                    }
                }
            },
            Action::Scroll(notches) => mouse.wheel(*notches)?,
            Action::Shell(command) => {
                let _ = self.shell_tx.send((gesture.to_owned(), command.clone()));
            }
        }
        Ok(())
    }
}
