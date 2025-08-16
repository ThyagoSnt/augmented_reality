import glob
import os
from utils.augmented_reality_marker import ARMarkerCube
from utils.images_processor import ImageBatchProcessor
from utils.data_extractor import load_config

if __name__ == "__main__":
    cfg = load_config()

    # Collect all images from the configured folder
    image_files = glob.glob(os.path.join(cfg.paths.aruco_images, "*.png"))

    # Initialize AR system
    ar_system = ARMarkerCube(cfg.paths.parameters, marker_length=cfg.calibration.marker_length)

    # Process images
    processor = ImageBatchProcessor(image_files, cfg.paths.processed_images, ar_system)
    processor.run()
