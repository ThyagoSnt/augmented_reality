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


    def detect_markers(self, gray_frame):
        """Detect ArUco markers in a grayscale frame."""
        corners, ids, _ = self.detector.detectMarkers(gray_frame)
        return corners, ids

    def estimate_pose(self, corners):
        """Estimate pose of a single marker using solvePnP."""
        obj_points = np.array([
            [-self.marker_length/2,  self.marker_length/2, 0],
            [ self.marker_length/2,  self.marker_length/2, 0],
            [ self.marker_length/2, -self.marker_length/2, 0],
            [-self.marker_length/2, -self.marker_length/2, 0],
        ], dtype=np.float32)

        img_points = corners.reshape(-1, 2)

        success, rvec, tvec = cv2.solvePnP(
            obj_points, img_points, self.camera_matrix, self.dist_coeffs
        )
        return rvec, tvec

    def draw_shaded_cuboid(self, frame, rvec, tvec, height_scale=1.0):
        """Draw a shaded cuboid centered and aligned with ArUco marker 1."""
        w, h = self.marker_length, self.marker_length   # base matches marker size
        d = self.marker_length * height_scale           # cuboid height relative to marker

        # Base aligned with the ArUco plane (Z = 0)
        points_3d = np.float32([
            [-w/2, -h/2, 0], [ w/2, -h/2, 0], [ w/2,  h/2, 0], [-w/2,  h/2, 0],   # base
            [-w/2, -h/2, -d], [ w/2, -h/2, -d], [ w/2,  h/2, -d], [-w/2,  h/2, -d] # top
        ])

        # Project 3D -> 2D
        points_2d, _ = self.project_points(points_3d, rvec, tvec,
                                         self.camera_matrix, self.dist_coeffs)
        points_2d = np.int32(points_2d).reshape(-1, 2)

        # Face definitions
        faces = [
            [0, 1, 2, 3], [4, 5, 6, 7],   # base, top
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7],
        ]

        # Rotation matrix
        R, _, _ = self.rodrigues(rvec)

        # Light direction
        light_dir = np.array([0.0, -1.0, -0.8], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)

        # Depth sorting
        face_depths = []
        for face in faces:
            pts_cam = (R @ points_3d[face].T).T + tvec.reshape(1, 3)
            avg_depth = np.mean(pts_cam[:, 2])
            face_depths.append((avg_depth, face))
        face_depths.sort(key=lambda x: -x[0])

        # Draw faces with shading
        for _, face in face_depths:
            pts_3d_face = points_3d[face]
            v1 = pts_3d_face[1] - pts_3d_face[0]
            v2 = pts_3d_face[2] - pts_3d_face[0]
            normal = np.cross(v1, v2)
            normal /= (np.linalg.norm(normal) + 1e-6)
            normal_cam = R @ normal

            intensity = np.dot(normal_cam, light_dir)
            intensity = max(0.2, intensity)

            base_color = np.array([80, 160, 220])
            color = (base_color * intensity).astype(np.uint8)
            color = tuple(int(c) for c in color)

            pts_2d_face = points_2d[face]
            cv2.fillConvexPoly(frame, pts_2d_face, color)
            cv2.polylines(frame, [pts_2d_face], True, (0, 0, 0), 1)

    def process_frame(self, frame):
        """Process a frame: detect both markers, draw cuboid if IDs 0 and 1 are found."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = self.detect_markers(gray)

        if ids is not None and set([0, 1]).issubset(set(ids.flatten())):
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Estimate poses
            rvecs, tvecs = {}, {}
            for corner, marker_id in zip(corners, ids.flatten()):
                rvec, tvec = self.estimate_pose(corner)
                rvecs[marker_id] = rvec
                tvecs[marker_id] = tvec
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.05)

            # Marker 1 defines orientation and center
            rvec_base, tvec_base = rvecs[1], tvecs[1]

            # Height = distance between marker 0 and marker 1
            height_scale = np.linalg.norm(tvecs[0] - tvecs[1]) / self.marker_length

            self.draw_shaded_cuboid(frame, rvec_base, tvec_base, height_scale)

        return frame
