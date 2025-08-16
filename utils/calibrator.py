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

    def l2_norm_manual(self, points1, points2):
        """
        Implements the L2 norm (Euclidean norm) manually, equivalent to cv2.norm(..., cv2.NORM_L2).
        
        This function is based on the concept of reprojection error as described in:
        - Hartley & Zisserman, Multiple View Geometry in Computer Vision (2nd Edition),
        Chapter 6: Camera Calibration and 3D Reconstruction, Section 6.3.1 (p. 312–314),
        where the geometric error is defined as the Euclidean distance between observed image points
        and reprojected points from the estimated camera model.

        Args:
            points1: np.ndarray of shape (N, 1, 2) or (N, 2)
                    These are the observed 2D image points.
            points2: np.ndarray of same shape as points1
                    These are the reprojected 2D image points from the estimated model.

        Returns:
            float: L2 norm between the sets of points (total Euclidean error),
                equivalent to: sqrt(Σ ||p_i - p̂_i||²)
        """
        # Remove extra dimension if present (OpenCV returns shape (N,1,2) from projectPoints)
        p1 = np.squeeze(points1)
        p2 = np.squeeze(points2)

        # Difference between each point pair
        diff = p1 - p2

        # Compute squared Euclidean distances for each point
        squared_dist = np.sum(diff ** 2, axis=-1)

        # Sum of all squared distances (Σ ||p_i - p̂_i||²)
        total = np.sum(squared_dist)

        # Final L2 norm: √(Σ ||p_i - p̂_i||²)
        return np.sqrt(total)
    

    def project_points(self, object_points, rvec, tvec, camera_matrix, dist_coeffs):
        """
        Manual implementation of cv2.projectPoints.
        Based on Hartley & Zisserman (Multiple View Geometry, 2nd Ed.),
        Chapter 6: Camera Calibration, Section 6.3.1 (pp. 312–314).

        Args:
            object_points: (N, 3) array of 3D object points in world coordinates.
            rvec: (3, 1) Rodrigues rotation vector.
            tvec: (3, 1) translation vector.
            camera_matrix: (3, 3) intrinsic matrix.
            dist_coeffs: (k1, k2, p1, p2, k3) distortion coefficients.

        Returns:
            image_points: (N, 1, 2) array of projected 2D image points.
        """

        # Convert rvec to rotation matrix using Rodrigues formula
        theta = np.linalg.norm(rvec)
        if theta < 1e-14:
            R = np.eye(3)
        else:
            r = rvec.flatten() / theta
            K = np.array([
                [0, -r[2], r[1]],
                [r[2], 0, -r[0]],
                [-r[1], r[0], 0]
            ])
            R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

        # Transform 3D points to camera coordinates
        object_points = np.asarray(object_points).reshape(-1, 3).T  # (3, N)
        X_cam = (R @ object_points) + tvec.reshape(3, 1)  # (3, N)

        # Normalize (divide by z) -> pinhole projection
        x = X_cam[0, :] / X_cam[2, :]
        y = X_cam[1, :] / X_cam[2, :]

        # Radial distortion
        r2 = x**2 + y**2
        k1, k2, p1, p2, k3 = dist_coeffs.flatten()
        radial = 1 + k1*r2 + k2*r2**2 + k3*r2**3

        # Tangential distortion
        x_dist = x*radial + 2*p1*x*y + p2*(r2 + 2*x**2)
        y_dist = y*radial + p1*(r2 + 2*y**2) + 2*p2*x*y

        # Apply intrinsics
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        u = fx * x_dist + cx
        v = fy * y_dist + cy

        # Return in OpenCV shape (N,1,2)
        image_points = np.stack([u, v], axis=1).reshape(-1, 1, 2)
        return image_points, None


    def compute_reprojection_error(self):
        """Compute the mean reprojection error across all calibration images."""
        total_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = self.project_points(
                self.objpoints[i], self.rvecs[i], self.tvecs[i],
                self.camera_matrix, self.dist_coeffs
            )
            error = self.l2_norm_manual(self.imgpoints[i], imgpoints2) / len(imgpoints2)
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
