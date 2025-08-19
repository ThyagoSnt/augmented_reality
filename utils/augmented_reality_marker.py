import cv2
import numpy as np
from utils.camera_math import CameraMathUtils


class ARMarker(CameraMathUtils):
    def __init__(self, camera_params_path, marker_length):
        # Load camera intrinsic parameters
        data = np.load(camera_params_path)
        self.camera_matrix = data["camera_matrix"]
        self.dist_coeffs = data["dist_coeffs"]

        # ArUco setup
        self.marker_length = marker_length
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

    # -------------------------------------------------------------------------
    # Marker detection and pose estimation
    # -------------------------------------------------------------------------
    def detect_markers(self, gray_frame):
        """Detect ArUco markers in a grayscale frame."""
        corners, ids, _ = self.detector.detectMarkers(gray_frame)
        return corners, ids

    def estimate_pose(self, corners):
        """
        Estimate pose (rvec, tvec) of a detected ArUco marker
        using the known square geometry.
        """
        obj_points = np.array([
            [-self.marker_length / 2,  self.marker_length / 2, 0],
            [ self.marker_length / 2,  self.marker_length / 2, 0],
            [ self.marker_length / 2, -self.marker_length / 2, 0],
            [-self.marker_length / 2, -self.marker_length / 2, 0],
        ], dtype=np.float32)

        img_points = corners.reshape(-1, 2)
        rvec, tvec = self.solve_pnp(obj_points, img_points, self.camera_matrix)
        return rvec, tvec

    # -------------------------------------------------------------------------
    # Drawing functions
    # -------------------------------------------------------------------------
    def draw_bbox_from_marker(self, frame, rvec, tvec, marker_length, height=0.10):
        """
        Draw a 3D bounding box with the ArUco marker as its base.
        """
        # Define base vertices in the marker's local coordinate system
        base_vertices = np.float32([
            [-marker_length / 2,  marker_length / 2, 0],
            [ marker_length / 2,  marker_length / 2, 0],
            [ marker_length / 2, -marker_length / 2, 0],
            [-marker_length / 2, -marker_length / 2, 0],
        ])

        # Top vertices extruded along +Z in marker's local frame
        top_vertices = base_vertices + np.array([0, 0, height], dtype=np.float32)

        # Combine into 8 vertices (base + top)
        points_3d = np.vstack([base_vertices, top_vertices])

        # Project into image
        points_2d, _ = self.project_points(points_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        points_2d = np.int32(points_2d).reshape(-1, 2)

        # Draw base (green)
        cv2.polylines(frame, [points_2d[:4]], isClosed=True, color=(0, 255, 0), thickness=2)
        # Draw top (blue)
        cv2.polylines(frame, [points_2d[4:]], isClosed=True, color=(255, 0, 0), thickness=2)
        # Draw vertical edges (red)
        for i in range(4):
            cv2.line(frame, tuple(points_2d[i]), tuple(points_2d[i+4]), (0, 0, 255), 2)

        return points_3d  # return 3D vertices so we can use the center later

    def draw_metallic_sphere(self, frame, rvec, tvec, bbox_points_3d):
        """
        Draw a metallic-looking sphere at the center of the given bbox.
        """
        # Compute center of bbox in 3D (average of vertices)
        center_3d = np.mean(bbox_points_3d, axis=0).reshape(1, 3)

        # Project center
        center_2d, _ = self.project_points(center_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        center_2d = tuple(map(int, center_2d.ravel()))

        # Estimate pixel radius (relative to bbox size)
        corner_3d = bbox_points_3d[0].reshape(1, 3)
        corner_2d, _ = self.project_points(corner_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        radius_px = int(np.linalg.norm(center_2d - corner_2d.ravel()))

        # Radial gradient fill
        for r in range(radius_px, 0, -1):
            intensity = int(200 * (1 - r / radius_px) + 55)
            cv2.circle(frame, center_2d, r, (intensity, intensity, intensity), -1)

    # -------------------------------------------------------------------------
    # Main processing
    # -------------------------------------------------------------------------
    def process_frame(self, frame):
        """Process one frame: detect markers, draw axes, sphere, and cube."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = self.detect_markers(gray)

        if ids is not None and set([0, 1]).issubset(set(ids.flatten())):
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            rvecs, tvecs = {}, {}
            for corner, marker_id in zip(corners, ids.flatten()):
                rvec, tvec = self.estimate_pose(corner)
                rvecs[marker_id] = rvec
                tvecs[marker_id] = tvec
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs,
                                rvec, tvec, 0.05)

            # Marker 1 gives orientation and XY
            rvec_base = rvecs[1]
            tvec_base = tvecs[1].copy()

            # Z comes from marker 0
            tvec_base[2, 0] = tvecs[0][2, 0]

            # Get corners of marker 1 (base)
            idx_marker1 = np.where(ids.flatten() == 1)[0][0]
            base_corners = corners[idx_marker1]

            # 1) Compute bbox points first
            bbox_points_3d = self.draw_bbox_from_marker(
                frame, rvec_base, tvec_base, self.marker_length, height=0.10
            )

            # 2) Draw the metallic sphere BEFORE re-drawing bbox
            # (so bbox will be drawn over it)
            self.draw_metallic_sphere(frame, rvec_base, tvec_base, bbox_points_3d)

        return frame
