import yaml
from dataclasses import dataclass

@dataclass
class CalibrationConfig:
    chessboard_size: tuple
    square_size: float
    marker_length: float

@dataclass
class PathsConfig:
    calibration_images: str
    aruco_images: str
    aruco_video: str
    parameters: str
    processed_images: str
    processed_videos: str

@dataclass
class FullConfig:
    calibration: CalibrationConfig
    paths: PathsConfig

def load_config(path="config.yaml") -> FullConfig:
    """Load calibration and paths configuration from a YAML file."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    calibration = config["calibration"]
    paths = config["paths"]

    return FullConfig(
        calibration=CalibrationConfig(
            chessboard_size=tuple(calibration["chessboard_size"]),
            square_size=calibration["square_size"],
            marker_length=calibration["marker_length"]
        ),
        paths=PathsConfig(
            calibration_images=paths["calibration_images"],
            aruco_images=paths["aruco_images"],
            aruco_video=paths["aruco_video"],
            parameters=paths["parameters"],
            processed_images=paths["processed_images"],
            processed_videos=paths["processed_videos"]
        )
    )
