use anyhow::{Context, Result};
use image::RgbImage;
use nokhwa::{
    pixel_format::RgbFormat,
    utils::{CameraIndex, RequestedFormat, RequestedFormatType},
    Camera,
};

pub struct WebcamCapture {
    camera: Camera,
}

impl WebcamCapture {
    pub fn open(index: u32) -> Result<Self> {
        let index = CameraIndex::Index(index);
        let format =
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
        let mut camera =
            Camera::new(index, format).context("opening camera (is another app using it?)")?;
        camera
            .open_stream()
            .context("starting camera stream -- check that your user can access /dev/video*")?;
        Ok(Self { camera })
    }

    /// Grab the next frame as an RGB image. Blocks until a frame is ready.
    pub fn next_frame(&mut self) -> Result<RgbImage> {
        let frame = self.camera.frame().context("reading camera frame")?;
        let decoded = frame
            .decode_image::<RgbFormat>()
            .context("decoding camera frame")?;
        Ok(decoded)
    }

    pub fn resolution(&self) -> (u32, u32) {
        let res = self.camera.resolution();
        (res.width(), res.height())
    }
}
