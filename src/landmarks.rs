#![allow(dead_code)]

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Point3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Point3 {
    pub fn dist(&self, other: &Point3) -> f32 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2) + (self.z - other.z).powi(2)).sqrt()
    }
}

pub const WRIST: usize = 0;
pub const THUMB_MCP: usize = 2;
pub const THUMB_TIP: usize = 4;
pub const INDEX_MCP: usize = 5;
pub const INDEX_TIP: usize = 8;
pub const MIDDLE_MCP: usize = 9;
pub const MIDDLE_TIP: usize = 12;
pub const RING_TIP: usize = 16;
pub const PINKY_TIP: usize = 20;

#[derive(Debug, Clone)]
pub struct Hand {
    pub landmarks: Vec<Point3>,
    pub points: Vec<Point3>,
    pub presence: f32,
    pub handedness_right: f32,
}