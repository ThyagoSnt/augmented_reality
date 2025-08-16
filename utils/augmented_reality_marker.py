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

        rvec, tvec = self.solve_pnp(obj_points, img_points, self.camera_matrix)
        return rvec, tvec


    def draw_shaded_cuboid(self, frame, rvec, tvec, size_m=0.10,
                        light_pos_cam=(0.0, 0.0, 0.0),
                        light_dir_cam=None,
                        ambient=0.18, kd=0.90, ks=0.35, shininess=32,
                        apply_gamma=True):
        """
        Draw a shaded cuboid centered at the marker's origin, with fixed size.
        Shading: Blinn–Phong (ambient + diffuse + specular), optional gamma correction.

        Args:
            size_m: edge length (meters) for width/height/depth (default 0.10 m).
            light_pos_cam: point light position in camera coords (default: camera origin).
            light_dir_cam: if not None, uses directional light (normalized) instead of point light.
            ambient, kd, ks: ambient/diffuse/specular coefficients.
            shininess: Blinn–Phong exponent.
            apply_gamma: if True, converts linear color to sRGB with gamma≈2.2.
        """
        w = h = d = float(size_m)

        # Cuboid vertices centered at origin in the marker's local coordinates
        points_3d = np.float32([
            [-w/2, -h/2, -d/2], [ w/2, -h/2, -d/2], [ w/2,  h/2, -d/2], [-w/2,  h/2, -d/2],  # near (z=-d/2)
            [-w/2, -h/2,  d/2], [ w/2, -h/2,  d/2], [ w/2,  h/2,  d/2], [-w/2,  h/2,  d/2],  # far  (z=+d/2)
        ])

        # Face indices (quads)
        faces = [
            [0, 1, 2, 3],  # near
            [4, 5, 6, 7],  # far
            [0, 1, 5, 4],  # bottom
            [1, 2, 6, 5],  # right
            [2, 3, 7, 6],  # top
            [3, 0, 4, 7],  # left
        ]

        # Rotation matrix from rvec
        R, _, _ = self.rodrigues(rvec)
        t = tvec.reshape(3, 1)

        # Base color (BGR for OpenCV)
        base_color_bgr = np.array([80, 160, 220], dtype=np.float32) / 255.0  # linear

        # Helper: linear -> sRGB gamma
        def to_srgb(c):
            if not apply_gamma:
                return np.clip(c, 0.0, 1.0)
            return np.clip(np.power(np.clip(c, 0.0, 1.0), 1.0 / 2.2), 0.0, 1.0)

        # Depth sort: draw far -> near using average camera-Z
        face_depths = []
        for face in faces:
            pts_cam = (R @ points_3d[face].T).T + t.T  # (4,3)
            avg_z = float(np.mean(pts_cam[:, 2]))
            face_depths.append((avg_z, face))
        face_depths.sort(key=lambda x: -x[0])  # far to near

        # Project all points once
        points_2d, _ = self.project_points(points_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        points_2d = np.int32(points_2d).reshape(-1, 2)

        # Lighting setup
        light_pos_cam = np.asarray(light_pos_cam, dtype=np.float32).reshape(3)
        directional = (light_dir_cam is not None)
        if directional:
            Ldir = np.asarray(light_dir_cam, dtype=np.float32).reshape(3)
            Ldir /= (np.linalg.norm(Ldir) + 1e-12)

        for _, face in face_depths:
            # Geometry in local
            p0, p1, p2 = points_3d[face[0]], points_3d[face[1]], points_3d[face[2]]
            v1 = p1 - p0
            v2 = p2 - p0
            n_local = np.cross(v1, v2)
            n_local /= (np.linalg.norm(n_local) + 1e-12)

            # Transform normal and center to camera coords
            n_cam = R @ n_local
            face_center_local = np.mean(points_3d[face], axis=0)
            face_center_cam = (R @ face_center_local.reshape(3, 1) + t).reshape(3)

            # View direction (camera at origin)
            V = -face_center_cam
            V /= (np.linalg.norm(V) + 1e-12)

            # Light direction
            if directional:
                L = Ldir
            else:
                L = light_pos_cam - face_center_cam
                L /= (np.linalg.norm(L) + 1e-12)

            # Blinn–Phong terms
            ndotl = max(0.0, float(np.dot(n_cam, L)))
            H = L + V
            H /= (np.linalg.norm(H) + 1e-12)
            ndoth = max(0.0, float(np.dot(n_cam, H)))
            spec = (ndoth ** shininess)

            # Final color in linear space
            color_linear = base_color_bgr * (ambient + kd * ndotl) + ks * spec

            # Gamma to sRGB and to 8-bit
            color_srgb = (to_srgb(color_linear) * 255.0).astype(np.uint8)
            color_bgr = tuple(int(c) for c in color_srgb)  # BGR

            # Rasterize
            pts_2d_face = points_2d[face]
            cv2.fillConvexPoly(frame, pts_2d_face, color_bgr)
            cv2.polylines(frame, [pts_2d_face], True, (0, 0, 0), 1)



    def process_frame(self, frame):
        """
        Detect markers 0 and 1. Draw a 10 cm cuboid:
        - centered on marker 1 (orientation and XY from marker 1),
        - placed at the Z height of marker 0 in camera coordinates.
        """
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

            # Orientation and XY center from marker 1
            rvec_base = rvecs[1]
            tvec_base = tvecs[1].copy()

            # Match the camera-space Z ("height") to marker 0
            tvec_base[2, 0] = tvecs[0][2, 0]  # align height (camera Z)

            # Draw fixed-size cuboid 0.10 m (10 cm) per edge
            self.draw_shaded_cuboid(frame, rvec_base, tvec_base, size_m=0.10)

        return frame
