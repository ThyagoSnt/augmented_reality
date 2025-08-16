from utils.augmented_reality_marker import ARMarkerCube
from utils.video_processor import VideoProcessor

if __name__ == "__main__":
    camera_params_path = "./database/camera_parameters/calibration_data.npz"

    ar_system = ARMarkerCube(camera_params_path,
                             marker_length=0.10
                             )
    
    processor = VideoProcessor("./database/aruco_videos/video.mp4",
                               "./processed_output/videos/video.mp4",
                               ar_system
                               )
    processor.run()
