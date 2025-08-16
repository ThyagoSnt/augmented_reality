from utils.calibrator import CameraCalibrator
from utils.data_extractor import load_config

if __name__ == "__main__":
    cfg = load_config()

    calibrator = CameraCalibrator(chessboard_size=cfg.calibration.chessboard_size, square_size=cfg.calibration.square_size)
    calibrator.process_images(cfg.paths.calibration_images, show=True)
    calibrator.calibrate()
    error = calibrator.compute_reprojection_error()

    print("Intrinsic Parameters (K):")
    print(calibrator.camera_matrix)

    print("\nDistortion Coefficients:")
    print(calibrator.dist_coeffs)
    
    print("\nMean Reprojection Error:", error)

    calibrator.save(cfg.paths.parameters)
