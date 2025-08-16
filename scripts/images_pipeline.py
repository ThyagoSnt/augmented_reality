from utils.augmented_reality_marker import ARMarkerCube
from utils.images_processor import ImageBatchProcessor

if __name__ == "__main__":
    camera_params_path = "./database/camera_parameters/calibration_data.npz"

    # Example: list of images
    image_files = [
    "./database/aruco_images/aruco_1.png",
    "./database/aruco_images/aruco_2.png",
    "./database/aruco_images/aruco_3.png",
    "./database/aruco_images/aruco_4.png",
    "./database/aruco_images/aruco_5.png",
    "./database/aruco_images/aruco_6.png",
    "./database/aruco_images/aruco_7.png"
    ]

    ar_system = ARMarkerCube(camera_params_path, marker_length=0.10)
    processor = ImageBatchProcessor(image_files, "./processed_output/images", ar_system)
    processor.run()
