# 🕶️ Augmented Reality with ArUco Markers
======================================

This project implements an Augmented Reality system using OpenCV and ArUco markers.
It allows detecting markers, estimating their pose, and rendering shaded 3D objects
(cube, sphere, etc.) aligned with the marker in real-world images and videos.

--------------------------------------
# 📂 Project Structure
--------------------------------------
```bash
augmented_reality/
├── config.yaml                # Configuration file (marker size, parameters)
├── database/                  # Input data
│   ├── aruco_images/          # Sample images with markers
│   ├── aruco_videos/          # Sample video with markers
│   ├── calibration_images/    # Chessboard images for calibration
│   └── camera_parameters/     # Saved calibration data
├── processed_output/          # Results (processed images/videos)
├── scripts/                   # Pipelines (images, videos, calibration)
├── utils/                     # Core utilities (AR system, processors)
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git files control
└── README.md                  # Project documentation
```

--------------------------------------
# 🚀 Installation
--------------------------------------
```bash
git clone https://github.com/your-username/augmented_reality.git
cd augmented_reality
pip install -r requirements.txt
```

--------------------------------------
# ▶️ Usage
--------------------------------------
## 1️⃣ Calibrate the camera
   Use chessboard images inside database/calibration_images/ to compute intrinsic parameters.
   The calibration result will be saved in database/camera_parameters/calibration_data.npz.

```bash
   python -m scripts.calibration_pipeline
```

## 2️⃣ Process sample images
   Apply marker detection and render 3D objects on images.

```bash
   python -m scripts.images_pipeline
```

## 3️⃣ Process a video
   Apply marker detection and render 3D objects on video frames.

```bash
   python -m scripts.video_pipeline
```

--------------------------------------
# 📸 Example Output
--------------------------------------
Results are saved under processed_output/:
- Images → processed_output/images/
- Videos → processed_output/videos/
