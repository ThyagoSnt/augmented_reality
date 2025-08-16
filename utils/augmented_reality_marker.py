import cv2
import numpy as np

class ARMarker:
    def __init__(self, camera_params_path, marker_length):
        # Load camera parameters
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

    def draw_shaded_cuboid(self, frame, rvec, tvec, size=(0.05, 0.05, 0.1)):
        """Draw a shaded cuboid with proper top-left-front lighting."""
        w, h, d = size
        points_3d = np.float32([
            [0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0],      # base
            [0, 0, -d], [w, 0, -d], [w, h, -d], [0, h, -d]   # top
        ])

        # Project to 2D
        points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec,
                                        self.camera_matrix, self.dist_coeffs)
        points_2d = np.int32(points_2d).reshape(-1, 2)

        # Faces of the cuboid
        faces = [
            [0, 1, 2, 3],  # bottom
            [4, 5, 6, 7],  # top
            [0, 1, 5, 4],  # side
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
        ]

        # Rotation matrix from rvec
        R, _ = cv2.Rodrigues(rvec)

        # ✅ Light from above, slightly left and a bit in front
        light_dir = np.array([-0.3, 1.0, 0.3], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)

        # Depth sorting (painter’s algorithm)
        face_depths = []
        for face in faces:
            pts_cam = (R @ points_3d[face].T).T + tvec.reshape(1, 3)
            avg_depth = np.mean(pts_cam[:, 2])
            face_depths.append((avg_depth, face))

        face_depths.sort(key=lambda x: -x[0])  # back to front

        for _, face in face_depths:
            pts_3d = points_3d[face]

            # Compute 3D face normal
            v1 = pts_3d[1] - pts_3d[0]
            v2 = pts_3d[2] - pts_3d[0]
            normal = np.cross(v1, v2)
            normal /= (np.linalg.norm(normal) + 1e-6)

            # Transform normal into camera space
            normal_cam = R @ normal

            # Lighting intensity
            intensity = np.dot(normal_cam, light_dir)
            intensity = max(0.2, intensity)  # keep a minimum brightness

            # Base metallic bluish color
            base_color = np.array([80, 160, 220])
            color = (base_color * intensity).astype(np.uint8)
            color = tuple(int(c) for c in color)

            pts_2d = points_2d[face]
            cv2.fillConvexPoly(frame, pts_2d, color)
            cv2.polylines(frame, [pts_2d], isClosed=True, color=(0, 0, 0), thickness=1)

    def process_frame(self, frame):
        """Process a frame: detect marker, draw axis and shaded cuboid."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = self.detect_markers(gray)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for corner, marker_id in zip(corners, ids.flatten()):
                rvec, tvec = self.estimate_pose(corner)
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.05)
                if marker_id == 0:
                    self.draw_shaded_cuboid(frame, rvec, tvec)

        return frame
