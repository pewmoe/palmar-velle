use crate::landmarks::{Hand, Point3};
use anyhow::{Context, Result};
use image::{imageops::FilterType, RgbImage};
use opencv::{core, imgproc, prelude::*};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

const MODEL_INPUT: u32 = 224;
const CONF_THRESHOLD: f32 = 0.5;
const TRACK_ENLARGE_FACTOR: f32 = 1.7;
const MIN_CROP_FRACTION: f32 = 0.25;

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
        fn build(model_path: &str) -> ort::Result<Session> {
            Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(2)?
                .commit_from_file(model_path)
        }

        let session = build(model_path)
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
        let frame = mat_to_rgb_image(mat_frame)?;
        let (fw, fh) = (frame.width() as f32, frame.height() as f32);
        let roi = self.roi.unwrap_or_else(|| centered_roi(fw, fh));

        let crop = extract_square_crop(&frame, roi);
        let resized = image::imageops::resize(&crop, MODEL_INPUT, MODEL_INPUT, FilterType::Triangle);

        // Convert from interleaved RGB (NHWC) to planar NCHW layout ([1, 3, 224, 224])
        let mut r_chan = Vec::with_capacity((MODEL_INPUT * MODEL_INPUT) as usize);
        let mut g_chan = Vec::with_capacity((MODEL_INPUT * MODEL_INPUT) as usize);
        let mut b_chan = Vec::with_capacity((MODEL_INPUT * MODEL_INPUT) as usize);

        for px in resized.pixels() {
            r_chan.push(px[0] as f32 / 255.0); // Red
            g_chan.push(px[1] as f32 / 255.0); // Green
            b_chan.push(px[2] as f32 / 255.0); // Blue
        }

        let mut input_data = Vec::with_capacity(r_chan.len() + g_chan.len() + b_chan.len());
        input_data.extend(r_chan);
        input_data.extend(g_chan);
        input_data.extend(b_chan);

        let input_tensor = Tensor::from_array((
            [1usize, 3usize, MODEL_INPUT as usize, MODEL_INPUT as usize],
            input_data,
        ))
        .map_err(|e| anyhow::anyhow!("building input tensor: {e}"))?;

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
        log::debug!("Model confidence: {:.3}", conf);

        if conf < CONF_THRESHOLD {
            self.misses += 1;
            if self.misses > 5 {
                self.roi = None;
            }
            return Ok(None);
        }
        self.misses = 0;

        let mut points = [Point3::default(); 21];
        let (mut min_x, mut min_y, mut max_x, mut max_y) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for i in 0..21 {
            let rx = landmarks_raw[i * 3];
            let ry = landmarks_raw[i * 3 + 1];
            let rz = landmarks_raw[i * 3 + 2];

            let nx = (rx / MODEL_INPUT as f32).clamp(-0.5, 1.5);
            let ny = (ry / MODEL_INPUT as f32).clamp(-0.5, 1.5);
            let fx = (roi.x0 + nx * roi.size) / fw;
            let fy = (roi.y0 + ny * roi.size) / fh;
            let fz = rz / MODEL_INPUT as f32;

            points[i] = Point3 { x: fx, y: fy, z: fz };
            min_x = min_x.min(fx);
            max_x = max_x.max(fx);
            min_y = min_y.min(fy);
            max_y = max_y.max(fy);
        }

        let bbox_w = (max_x - min_x) * fw;
        let bbox_h = (max_y - min_y) * fh;
        let cx = (min_x + max_x) * 0.5 * fw;
        let cy = (min_y + max_y) * 0.5 * fh;
        let size = (bbox_w.max(bbox_h) * TRACK_ENLARGE_FACTOR).max(fw.min(fh) * MIN_CROP_FRACTION);
        self.roi = Some(Roi {
            x0: cx - size / 2.0,
            y0: cy - size / 2.0,
            size,
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

fn mat_to_rgb_image(mat: &core::Mat) -> Result<RgbImage> {
    let mut rgb_mat = core::Mat::default();
    imgproc::cvt_color(mat, &mut rgb_mat, imgproc::COLOR_BGR2RGB, 0)?;
    let data = rgb_mat.data_bytes()?;
    let width = rgb_mat.cols() as u32;
    let height = rgb_mat.rows() as u32;
    RgbImage::from_raw(width, height, data.to_vec())
        .context("Failed to create RgbImage from Mat buffer")
}

fn centered_roi(fw: f32, fh: f32) -> Roi {
    let size = fw.min(fh) * 0.7;
    Roi {
        x0: (fw - size) / 2.0,
        y0: (fh - size) / 2.0,
        size,
    }
}

fn extract_square_crop(frame: &RgbImage, roi: Roi) -> RgbImage {
    let (fw, fh) = (frame.width() as i64, frame.height() as i64);
    let size = roi.size.round().max(1.0) as i64;
    let x0 = (roi.x0.round() as i64).clamp(0, (fw - 1).max(0));
    let y0 = (roi.y0.round() as i64).clamp(0, (fh - 1).max(0));
    let w = size.min(fw - x0).max(1);
    let h = size.min(fh - y0).max(1);

    image::imageops::crop_imm(frame, x0 as u32, y0 as u32, w as u32, h as u32).to_image()
}