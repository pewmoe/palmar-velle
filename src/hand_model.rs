use crate::landmarks::{Hand, Point3};
use anyhow::{Context, Result};
use opencv::{core, imgproc, prelude::*};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

const MODEL_INPUT: i32 = 224;
const CONF_THRESHOLD: f32 = 0.5;
const MIN_CROP_FRACTION: f32 = 0.4;
const CONFIDENCE_HOLD_FRAMES: u32 = 5;

#[derive(Clone, Copy)]
struct Roi {
    x0: f32,
    y0: f32,
    size: f32,
}

pub struct HandModel {
    session: Session,
    input_name: String,
    roi: Option<Roi>,
    misses: u32,
    last_hand: Option<Hand>,
}

pub type HandTracker = HandModel;

impl HandModel {
    pub fn new(model_path: &str) -> Result<Self> {
        Self::load(model_path)
    }

    pub fn load(model_path: &str) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("session builder failed: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("optimization setup failed: {e}"))?
            .with_intra_threads(2)
            .map_err(|e| anyhow::anyhow!("thread setup failed: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("loading model failed: {e}"))?;

        let input_name = session
            .inputs()
            .first()
            .map(|input| input.name().to_string())
            .context("model has no inputs")?;

        Ok(Self {
            session,
            input_name,
            roi: None,
            misses: 0,
            last_hand: None,
        })
    }

    pub fn detect(&mut self, frame: &impl core::MatTraitConst) -> Result<Option<Hand>> {
        let fw = frame.cols() as f32;
        let fh = frame.rows() as f32;

        if fw < 1.0 || fh < 1.0 || frame.channels() != 3 {
            return Ok(None);
        }

        let roi = self.roi.unwrap_or_else(|| centered_roi(fw, fh));

        if !roi.x0.is_finite()
            || !roi.y0.is_finite()
            || !roi.size.is_finite()
            || roi.size <= 0.0
        {
            self.roi = None;
            return Ok(None);
        }

        let x0 = roi.x0.round().clamp(0.0, fw - 1.0) as i32;
        let y0 = roi.y0.round().clamp(0.0, fh - 1.0) as i32;
        let width = roi.size.round().clamp(1.0, fw - x0 as f32) as i32;
        let height = roi.size.round().clamp(1.0, fh - y0 as f32) as i32;

        let crop = core::Mat::roi(frame, core::Rect::new(x0, y0, width, height))?;

        let mut resized = core::Mat::default();
        imgproc::resize(
            &crop,
            &mut resized,
            core::Size::new(MODEL_INPUT, MODEL_INPUT),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;

        let mut lab = core::Mat::default();
        imgproc::cvt_color(&resized, &mut lab, imgproc::COLOR_RGB2Lab, 0)?;

        let mut channels = core::Vector::<core::Mat>::new();
        core::split(&lab, &mut channels)?;

        if channels.len() != 3 {
            return Ok(None);
        }

        let l_channel = channels.get(0)?;
        let mut equalized_l = core::Mat::default();
        let mut clahe = imgproc::create_clahe(2.0, core::Size::new(8, 8))?;
        clahe.apply(&l_channel, &mut equalized_l)?;
        channels.set(0, equalized_l)?;

        let mut enhanced = core::Mat::default();
        let mut merged = core::Mat::default();
        core::merge(&channels, &mut merged)?;
        imgproc::cvt_color(&merged, &mut enhanced, imgproc::COLOR_Lab2RGB, 0)?;

        let width = MODEL_INPUT as usize;
        let height = MODEL_INPUT as usize;
        let spatial = width * height;
        let expected_len = spatial * 3;
        let row_stride = enhanced.step1(0)?;
        let bytes = enhanced.data_bytes()?;

        if enhanced.rows() != MODEL_INPUT
            || enhanced.cols() != MODEL_INPUT
            || enhanced.channels() != 3
            || row_stride < width * 3
            || bytes.len() < row_stride * height
        {
            return Ok(None);
        }

        let mut input_data = vec![0.0f32; expected_len];

        for row in 0..height {
            let row_start = row * row_stride;

            for col in 0..width {
                let pixel = row_start + col * 3;
                let index = row * width + col;

                input_data[index] = bytes[pixel] as f32 / 255.0;
                input_data[spatial + index] = bytes[pixel + 1] as f32 / 255.0;
                input_data[2 * spatial + index] = bytes[pixel + 2] as f32 / 255.0;
            }
        }

        let input = Tensor::from_array((
            [1usize, 3, width, height],
            input_data,
        ))
        .map_err(|e| anyhow::anyhow!("creating input tensor failed: {e}"))?;

        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => input])
            .map_err(|e| anyhow::anyhow!("inference failed: {e}"))?;

        if outputs.len() < 3 {
            return Ok(None);
        }

        let (_, landmarks) = outputs[0].try_extract_tensor::<f32>()?;
        let (_, confidence) = outputs[1].try_extract_tensor::<f32>()?;
        let (_, handedness) = outputs[2].try_extract_tensor::<f32>()?;

        if landmarks.len() < 63 {
            return Ok(None);
        }

        let conf = confidence.first().copied().unwrap_or(0.0);

        if !conf.is_finite() {
            return Ok(None);
        }

        if conf < CONF_THRESHOLD {
            self.misses = self.misses.saturating_add(1);

            if self.misses <= CONFIDENCE_HOLD_FRAMES {
                return Ok(self.last_hand.clone());
            }

            self.roi = None;
            self.last_hand = None;
            return Ok(None);
        }

        self.misses = 0;

        let mut points = Vec::with_capacity(21);
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;

        for i in 0..21 {
            let x = landmarks[i * 3];
            let y = landmarks[i * 3 + 1];
            let z = landmarks[i * 3 + 2];

            let point = Point3 {
                x: (roi.x0 + x / MODEL_INPUT as f32 * roi.size) / fw,
                y: (roi.y0 + y / MODEL_INPUT as f32 * roi.size) / fh,
                z: z / MODEL_INPUT as f32,
            };

            if !point.x.is_finite() || !point.y.is_finite() || !point.z.is_finite() {
                return Ok(None);
            }

            min_x = min_x.min(point.x);
            min_y = min_y.min(point.y);
            max_x = max_x.max(point.x);
            max_y = max_y.max(point.y);
            points.push(point);
        }

        let bbox_w = (max_x - min_x) * fw;
        let bbox_h = (max_y - min_y) * fh;

        if bbox_w <= 0.0 || bbox_h <= 0.0 {
            return Ok(None);
        }

        let center_x = (min_x + max_x) * 0.5 * fw;
        let center_y = (min_y + max_y) * 0.5 * fh;

        let min_size = fw.min(fh) * MIN_CROP_FRACTION;
        let max_size = fw.min(fh) * 0.9;
        let target_size = (bbox_w.max(bbox_h) * 1.5).max(min_size);
        let size = (roi.size * 0.6 + target_size * 0.4).clamp(min_size, max_size);

        self.roi = Some(Roi {
            x0: (center_x - size * 0.5).clamp(0.0, fw - size),
            y0: (center_y - size * 0.5).clamp(0.0, fh - size),
            size,
        });

        let hand = Hand {
            landmarks: points.clone(),
            points,
            presence: conf,
            handedness_right: handedness.first().copied().unwrap_or(0.5),
        };

        self.last_hand = Some(hand.clone());
        Ok(Some(hand))
    }
}

fn centered_roi(width: f32, height: f32) -> Roi {
    let size = width.min(height) * 0.8;

    Roi {
        x0: (width - size) * 0.5,
        y0: (height - size) * 0.5,
        size,
    }
}