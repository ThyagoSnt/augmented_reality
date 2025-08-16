import cv2
import numpy as np
import glob

class CameraCalibrator:
    def __init__(self, chessboard_size, square_size):
        """
        Args:
            chessboard_size (tuple): Number of inner corners per chessboard row and column (cols, rows).
            square_size (float): Size of one square in meters.
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size

        # Prepare a single grid of object points (3D points in real-world space)
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        objp *= square_size
        self.objp = objp

        # Storage for all images
        self.objpoints = []  # 3D points
        self.imgpoints = []  # 2D points
        self.image_size = None

        # Calibration results
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.reprojection_error = None

    def process_images(self, image_folder="images/*.png", show=False):
        """Detect chessboard corners in all images from the given folder."""
        images = glob.glob(image_folder)

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.image_size = gray.shape[::-1]

            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

            if ret:
                self.objpoints.append(self.objp)
                corners2 = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
                self.imgpoints.append(corners2)

                if show:
                    cv2.drawChessboardCorners(img, self.chessboard_size, corners2, ret)
                    cv2.imshow('Chessboard Detection', img)
                    cv2.waitKey(100)

        if show:
            cv2.destroyAllWindows()

    def calibrate(self):
        """Run camera calibration using detected points."""
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, self.image_size, None, None
        )
        self.camera_matrix = mtx
        self.dist_coeffs = dist
        self.rvecs = rvecs
        self.tvecs = tvecs
        return ret

    def compute_reprojection_error(self):
        """Compute the mean reprojection error across all calibration images."""
        total_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                self.objpoints[i], self.rvecs[i], self.tvecs[i],
                self.camera_matrix, self.dist_coeffs
            )
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        self.reprojection_error = total_error / len(self.objpoints)
        return self.reprojection_error

    def save(self, output_path="calibration_data.npz"):
        """Save calibration results to a .npz file."""
        np.savez(
            output_path,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            reprojection_error=self.reprojection_error
        )
        print(f"Calibration data saved to {output_path}")

    def load(self, input_path="calibration_data.npz"):
        """Load calibration results from a .npz file."""
        data = np.load(input_path)
        self.camera_matrix = data["camera_matrix"]
        self.dist_coeffs = data["dist_coeffs"]
        self.reprojection_error = data["reprojection_error"]
        print(f"Calibration data loaded from {input_path}")
