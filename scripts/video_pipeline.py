from utils.augmented_reality_marker import ARMarker
from utils.video_processor import VideoProcessor
from utils.data_extractor import load_config

if __name__ == "__main__":
    cfg = load_config()

    ar_system = ARMarker(cfg.paths.parameters, marker_length=cfg.calibration.marker_length)
    
    processor = VideoProcessor(cfg.paths.aruco_video, cfg.paths.processed_videos, ar_system)
    processor.run()
