use crate::landmarks::{Hand, Point3};
use anyhow::{Context, Result};
use opencv::{core, imgproc, prelude::*};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

const MODEL_INPUT: i32 = 224;
const CONF_THRESHOLD: f32 = 0.5;
const MIN_CROP_FRACTION: f32 = 0.4;

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
}

pub type HandTracker = HandModel;

impl HandModel {
    pub fn new(model_path: &str) -> Result<Self> {
        Self::load(model_path)
    }

    pub fn load(model_path: &str) -> Result<Self> {
        // Explicitly map ort::Error types since they are no longer Send/Sync compatible with anyhow's ?
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("session builder failed: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("failed to set optimization level: {e}"))?
            .with_intra_threads(2)
            .map_err(|e| anyhow::anyhow!("failed to set intra threads: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("loading ONNX model at {model_path}: {e}"))?;

        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .context("model has no inputs")?;

        Ok(Self {
            session,
            input_name,
            roi: None,
            misses: 0,
        })
    }

    pub fn detect(&mut self, mat_frame: &core::Mat) -> Result<Option<Hand>> {
        let fw = mat_frame.cols() as f32;
        let fh = mat_frame.rows() as f32;

        if fw == 0.0 || fh == 0.0 {
            return Ok(None);
        }

        // 1. Fall back to centered crop if ROI is lost
        let roi = self.roi.unwrap_or_else(|| centered_roi(fw, fh));

        // 2. OpenCV Native Crop & Resize
        let crop_rect = core::Rect::new(
            (roi.x0.round() as i32).clamp(0, fw as i32 - 1),
            (roi.y0.round() as i32).clamp(0, fh as i32 - 1),
            (roi.size.round() as i32).min(fw as i32 - (roi.x0.round() as i32).clamp(0, fw as i32 - 1)).max(1),
            (roi.size.round() as i32).min(fh as i32 - (roi.y0.round() as i32).clamp(0, fh as i32 - 1)).max(1),
        );

        let cropped = core::Mat::roi(mat_frame, crop_rect)?;
        let mut rgb_crop = core::Mat::default();
        imgproc::cvt_color(&cropped, &mut rgb_crop, imgproc::COLOR_BGR2RGB, 0)?;

        let mut resized = core::Mat::default();
        imgproc::resize(
            &rgb_crop,
            &mut resized,
            core::Size::new(MODEL_INPUT, MODEL_INPUT),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;

        // 3. Fast Mat to Tensor transformation (NCHW layout)
        let mut input_data = vec![0.0f32; (3 * MODEL_INPUT * MODEL_INPUT) as usize];
        let bytes = resized.data_bytes()?;
        let spatial_size = (MODEL_INPUT * MODEL_INPUT) as usize;

        for i in 0..spatial_size {
            input_data[i] = bytes[i * 3] as f32 / 255.0; // R
            input_data[spatial_size + i] = bytes[i * 3 + 1] as f32 / 255.0; // G
            input_data[2 * spatial_size + i] = bytes[i * 3 + 2] as f32 / 255.0; // B
        }

        let input_tensor = Tensor::from_array((
            [1usize, 3usize, MODEL_INPUT as usize, MODEL_INPUT as usize],
            input_data,
        ))
        .map_err(|e| anyhow::anyhow!("building input tensor: {e}"))?;

        // 4. Inference
        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => input_tensor])
            .map_err(|e| anyhow::anyhow!("running inference: {e}"))?;

        let (_, landmarks_raw) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("reading landmarks output: {e}"))?;
        let (_, conf_raw) = outputs[1]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("reading confidence output: {e}"))?;
        let (_, handedness_raw) = outputs[2]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("reading handedness output: {e}"))?;

        let conf = conf_raw.first().copied().unwrap_or(0.0);

        if conf < CONF_THRESHOLD {
            self.misses += 1;
            if self.misses > 3 {
                self.roi = None;
            }
            return Ok(None);
        }
        self.misses = 0;

        // 5. Landmark Coordinate Remapping
        let mut points = [Point3::default(); 21];
        let (mut min_x, mut min_y, mut max_x, mut max_y) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);

        for i in 0..21 {
            let rx = landmarks_raw[i * 3];
            let ry = landmarks_raw[i * 3 + 1];
            let rz = landmarks_raw[i * 3 + 2];

            let nx = rx / MODEL_INPUT as f32;
            let ny = ry / MODEL_INPUT as f32;

            let fx = (roi.x0 + nx * roi.size) / fw;
            let fy = (roi.y0 + ny * roi.size) / fh;
            let fz = rz / MODEL_INPUT as f32;

            points[i] = Point3 { x: fx, y: fy, z: fz };
            min_x = min_x.min(fx);
            max_x = max_x.max(fx);
            min_y = min_y.min(fy);
            max_y = max_y.max(fy);
        }

        // Stable ROI update
        let bbox_w = (max_x - min_x) * fw;
        let bbox_h = (max_y - min_y) * fh;
        let cx = (min_x + max_x) * 0.5 * fw;
        let cy = (min_y + max_y) * 0.5 * fh;

        let target_size = (bbox_w.max(bbox_h) * 1.5).max(fw.min(fh) * MIN_CROP_FRACTION);
        
        // Smooth ROI transitions using simple EMA
        let new_size = roi.size * 0.6 + target_size * 0.4;
        
        self.roi = Some(Roi {
            x0: cx - new_size / 2.0,
            y0: cy - new_size / 2.0,
            size: new_size,
        });

        let vec_pts = points.to_vec();

        Ok(Some(Hand {
            landmarks: vec_pts.clone(),
            points: vec_pts,
            presence: conf,
            handedness_right: handedness_raw.first().copied().unwrap_or(0.5),
        }))
    }
}

fn centered_roi(fw: f32, fh: f32) -> Roi {
    let size = fw.min(fh) * 0.8;
    Roi {
        x0: (fw - size) / 2.0,
        y0: (fh - size) / 2.0,
        size,
    }
}