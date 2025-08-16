from utils.calibrator import CameraCalibrator

import yaml

def load_calibration_config(path="calibration_setup.yaml"):
    """Load calibration parameters from a YAML file."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    calibration = config["calibration"]

    chessboard_size = tuple(calibration["chessboard_size"])
    square_size = calibration["square_size"]
    marker_length = calibration["marker_length"]

    return chessboard_size, square_size, marker_length


if __name__ == "__main__":
    images_path = "./database/calibration_images"
    save_path = "./database/camera_parameters/calibration_data.npz"
    yaml_path = "./config.yaml"

    calibrator = CameraCalibrator(chessboard_size=(10, 7), square_size=0.025)
    calibrator.process_images(images_path, show=True)
    calibrator.calibrate()
    error = calibrator.compute_reprojection_error()

    print("Intrinsic Parameters (K):")
    print(calibrator.camera_matrix)

    print("\nDistortion Coefficients:")
    print(calibrator.dist_coeffs)
    
    print("\nMean Reprojection Error:", error)

    calibrator.save(save_path)
