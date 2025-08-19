import cv2
import numpy as np
import itertools
from utils.camera_math import CameraMathUtils


class ARMarker(CameraMathUtils):
    def __init__(self, camera_params_path, marker_length, sphere_radius_px: int = 40, base_marker_id: int = 1):
        # Load camera intrinsic parameters
        data = np.load(camera_params_path)
        self.camera_matrix = data["camera_matrix"]
        self.dist_coeffs = data["dist_coeffs"]

        # ArUco setup
        self.marker_length = marker_length
        self.base_marker_id = base_marker_id
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

        # Sphere radius (in pixels, fixed)
        self.sphere_radius_px = sphere_radius_px

        # Local 3D coordinates of the marker corners (used for pose/top extrusion)
        # Order: top-left, top-right, bottom-right, bottom-left (OpenCV/ArUco convention)
        self.marker_obj_points = np.array([
            [-self.marker_length / 2,  self.marker_length / 2, 0],
            [ self.marker_length / 2,  self.marker_length / 2, 0],
            [ self.marker_length / 2, -self.marker_length / 2, 0],
            [-self.marker_length / 2, -self.marker_length / 2, 0],
        ], dtype=np.float32)

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
        img_points = corners.reshape(-1, 2)
        rvec, tvec = self.solve_pnp(self.marker_obj_points, img_points, self.camera_matrix)
        return rvec, tvec

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def compute_bbox_points_local(self, height: float):
        """
        Build bbox in marker-local 3D coordinates.
        Base = marker_obj_points (z=0), Top = z + height.
        """
        base_3d = self.marker_obj_points
        top_3d = base_3d + np.array([0, 0, height], dtype=np.float32)
        return np.vstack([base_3d, top_3d])
    
    def bbox_dimensions(self, bbox_points_3d_local):
        """
        Compute width (X), height (Z), depth (Y) of the bbox in marker-local coords.
        Assumes bbox_points_3d_local is shape (8,3).
        """
        # Base = first 4 points (z=0 plane)
        base = bbox_points_3d_local[:4]

        # Width = distance between base[0] and base[1]
        width = np.linalg.norm(base[0] - base[1])
        # Depth = distance between base[1] and base[2]
        depth = np.linalg.norm(base[1] - base[2])
        # Height = distance between base[0] and bbox_points_3d_local[4] (top directly above)
        height = np.linalg.norm(bbox_points_3d_local[0] - bbox_points_3d_local[4])

        return width, depth, height

    def _best_matching_order(self, rvec, tvec, detected_corners_2d):
        """
        Find the permutation of detected base corners that best matches the
        projection order of self.marker_obj_points. This ensures vertical
        edges connect corresponding vertices.
        """
        proj_base2d, _ = self.project_points(
            self.marker_obj_points, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        proj_base2d = proj_base2d.reshape(-1, 2)
        det = detected_corners_2d.reshape(-1, 2)

        best_perm = None
        best_cost = float('inf')
        for perm in itertools.permutations(range(4)):
            cost = 0.0
            for i in range(4):
                cost += np.linalg.norm(det[perm[i]] - proj_base2d[i])
            if cost < best_cost:
                best_cost = cost
                best_perm = perm

        ordered = det[list(best_perm)]
        return ordered  # shape (4,2), ordered to align with marker_obj_points order

    # -------------------------------------------------------------------------
    # Drawing functions
    # -------------------------------------------------------------------------
    def draw_bbox_aligned_to_detected_base(self, frame, rvec, tvec, base_corners_img2d, height):
        """
        Draw a bbox whose BASE is EXACTLY the detected ArUco corners (pixel-perfect),
        with vertical edges and TOP computed from the marker pose and desired height.
        """
        # Ensure base corners are ordered to match the 3D object points
        base2d_ordered = self._best_matching_order(rvec, tvec, base_corners_img2d)

        # Compute and project TOP corners from local 3D points
        bbox_points_3d_local = self.compute_bbox_points_local(height)
        top3d = bbox_points_3d_local[4:]  # 4 top vertices

        top2d, _ = self.project_points(top3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        top2d = top2d.reshape(-1, 2)

        # Draw BASE using the EXACT detected corners (no reprojection) -> perfect alignment
        base_poly = np.int32(base2d_ordered).reshape(-1, 1, 2)
        cv2.polylines(frame, [base_poly], isClosed=True, color=(0, 255, 0), thickness=2)  # green

        # Draw TOP (projected)
        top_poly = np.int32(top2d).reshape(-1, 1, 2)
        cv2.polylines(frame, [top_poly], isClosed=True, color=(255, 0, 0), thickness=2)  # blue

        # Draw VERTICAL EDGES: connect each base[i] (detected) -> top[i] (projected)
        for i in range(4):
            p_base = tuple(np.int32(base2d_ordered[i]))
            p_top  = tuple(np.int32(top2d[i]))
            cv2.line(frame, p_base, p_top, (0, 0, 255), 2)  # red

        return bbox_points_3d_local  # return local 3D for sphere center, if desired

    def draw_metallic_sphere(self, frame, rvec, tvec, bbox_points_3d_local):
        """Draw a metallic-looking sphere at the center of the given bbox."""
        center_3d = np.mean(bbox_points_3d_local, axis=0).reshape(1, 3)
        center_2d, _ = self.project_points(center_3d, rvec, tvec,
                                           self.camera_matrix, self.dist_coeffs)
        center_2d = tuple(map(int, center_2d.ravel()))

        radius_px = self.sphere_radius_px
        for r in range(radius_px, 0, -1):
            intensity = int(200 * (1 - r / radius_px) + 55)
            cv2.circle(frame, center_2d, r, (intensity, intensity, intensity), -1)

    # -------------------------------------------------------------------------
    # Main processing
    # -------------------------------------------------------------------------
    def process_frame(self, frame):
        """
        Detect markers, and draw a bbox whose base is EXACTLY the detected
        corners of ArUco index 0. Top and vertical edges come from the pose.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = self.detect_markers(gray)

        if ids is not None:
            ids_flat = ids.flatten()
            id_set = set(ids_flat)

            # draw detected markers (optional)
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            rvecs, tvecs = {}, {}
            marker_corners = {}
            for corner, marker_id in zip(corners, ids_flat):
                rvec, tvec = self.estimate_pose(corner)
                rvecs[marker_id] = rvec
                tvecs[marker_id] = tvec
                marker_corners[marker_id] = corner
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.05)

            # We require marker as the base
            if self.base_marker_id in rvecs:
                rvec_base = rvecs[self.base_marker_id]
                tvec_base = tvecs[self.base_marker_id]
                base_corners_img2d = marker_corners[self.base_marker_id]

                # Height choice:
                # If marker 1 exists, use vertical distance between 0 and 1; else, fallback.
                if 1 in tvecs:
                    rel_pos = tvecs[0] - tvecs[1]
                    y_rel = float(rel_pos[1, 0])
                    bbox_height = abs(y_rel)
                else:
                    bbox_height = float(self.marker_length * 0.5)  # fallback height

                # Draw BBOX aligned to detected base corners
                bbox_points_3d_local = self.draw_bbox_aligned_to_detected_base(
                    frame, rvec_base, tvec_base, base_corners_img2d, bbox_height
                )

                # Draw metallic sphere centered inside this bbox
                self.draw_metallic_sphere(frame, rvec_base, tvec_base, bbox_points_3d_local)

                # Compute bbox dimensions
                w, d, h = self.bbox_dimensions(bbox_points_3d_local)
                print(f"BBox dimensions -> Width (X): {w:.3f} m, Depth (Y): {d:.3f} m, Height (Z): {h:.3f} m")

        return frame