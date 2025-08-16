import cv2
import numpy as np

class ARMarkerCube:
    def __init__(self, camera_params_path, marker_length):
        # Camera parameters
        data = np.load(camera_params_path)
        self.camera_matrix = data["camera_matrix"]
        self.dist_coeffs = data["dist_coeffs"]

        # Aruco parameters
        self.marker_length = marker_length

        # ArUco setup
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

    def draw_cuboid(self, frame, rvec, tvec, size=(0.05, 0.05, 0.1)):
        """Draw a shaded cuboid on top of the detected marker."""
        w, h, d = size
        points = np.float32([
            [0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0],      # base
            [0, 0, -d], [w, 0, -d], [w, h, -d], [0, h, -d]   # top
        ])
        points, _ = cv2.projectPoints(points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        points = np.int32(points).reshape(-1, 2)

        faces = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
        ]

        for face in faces:
            pts_face = points[face]
            v1 = np.array([pts_face[1][0] - pts_face[0][0],
                           pts_face[1][1] - pts_face[0][1], 0.0])
            v2 = np.array([pts_face[2][0] - pts_face[0][0],
                           pts_face[2][1] - pts_face[0][1], 0.0])
            normal = np.cross(v1, v2)

            brightness = max(
                0.3,
                float(np.dot(normal, np.array([0, 0, -1]))) / (np.linalg.norm(normal) + 1e-6)
            )

            color = (int(0 * brightness), int(150 * brightness), int(255 * brightness))
            cv2.fillConvexPoly(frame, pts_face, color)

    def process_frame(self, frame):
        """Process a frame: detect marker, draw axis and cuboid."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = self.detect_markers(gray)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for corner, marker_id in zip(corners, ids.flatten()):
                rvec, tvec = self.estimate_pose(corner)
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.05)
                if marker_id == 0:
                    self.draw_cuboid(frame, rvec, tvec)

        return frame