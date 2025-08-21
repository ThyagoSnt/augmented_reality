import cv2
import numpy as np
import itertools
import trimesh

from utils.camera_math import CameraMathUtils


class ARMarker(CameraMathUtils):
    def __init__(self, camera_params_path, marker_length, base_marker_id: int = 1,
                 model_path="database/models/16433_Pig.obj",
                 mtl_path="database/models/Blank.mtl"):
        
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

        # ⚠️ Faltava isso aqui!
        self.marker_obj_points = np.array([
            [-self.marker_length / 2,  self.marker_length / 2, 0],
            [ self.marker_length / 2,  self.marker_length / 2, 0],
            [ self.marker_length / 2, -self.marker_length / 2, 0],
            [-self.marker_length / 2, -self.marker_length / 2, 0],
        ], dtype=np.float32)

        # Load 3D model
        self.mesh = trimesh.load(model_path, force="mesh")

        # If user passes .mtl path, try to merge colors
        self.mtl_colors = None
        if mtl_path is not None:
            self.mtl_colors = self._load_mtl_colors(mtl_path)

        # Normalize/center model
        self.mesh.apply_translation(-self.mesh.centroid)
        scale = self.marker_length / max(self.mesh.extents)
        self.mesh.apply_scale(scale)

    def _load_mtl_colors(self, mtl_path):
        """Parse a .mtl file for diffuse colors (Kd)."""
        colors = {}
        current = None
        with open(mtl_path, "r") as f:
            for line in f:
                if line.startswith("newmtl"):
                    current = line.split()[1].strip()
                elif line.startswith("Kd") and current:
                    vals = [float(v) for v in line.split()[1:4]]
                    rgb = tuple(int(v*255) for v in vals)
                    colors[current] = rgb
        return colors

    # -------------------------------------------------------------------------
    def detect_markers(self, gray_frame):
        corners, ids, _ = self.detector.detectMarkers(gray_frame)
        return corners, ids

    def estimate_pose(self, corners):
        img_points = corners.reshape(-1, 2)
        rvec, tvec = self.solve_pnp(self.marker_obj_points, img_points, self.camera_matrix)
        return rvec, tvec

    # -------------------------------------------------------------------------
    def compute_bbox_points_local(self, height: float):
        base_3d = self.marker_obj_points
        top_3d = base_3d + np.array([0, 0, height], dtype=np.float32)
        return np.vstack([base_3d, top_3d])

    def bbox_dimensions(self, bbox_points_3d_local):
        base = bbox_points_3d_local[:4]
        width = np.linalg.norm(base[0] - base[1])
        depth = np.linalg.norm(base[1] - base[2])
        height = np.linalg.norm(bbox_points_3d_local[0] - bbox_points_3d_local[4])
        return width, depth, height

    def draw_bbox(self, frame, rvec, tvec, corners_2d, height):
        bbox_points_3d_local = self.compute_bbox_points_local(height)
        top3d = bbox_points_3d_local[4:]

        top2d, _ = self.project_points(top3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        top2d = top2d.reshape(-1, 2)

        base_poly = np.int32(corners_2d).reshape(-1, 1, 2)
        cv2.polylines(frame, [base_poly], True, (0, 255, 0), 2)
        top_poly = np.int32(top2d).reshape(-1, 1, 2)
        cv2.polylines(frame, [top_poly], True, (255, 0, 0), 2)

        for i in range(4):
            p_base = tuple(np.int32(corners_2d[i]))
            p_top = tuple(np.int32(top2d[i]))
            cv2.line(frame, p_base, p_top, (0, 0, 255), 2)

        return bbox_points_3d_local

    # -------------------------------------------------------------------------
    def draw_obj(self, frame, rvec, tvec):
        vertices = np.asarray(self.mesh.vertices, dtype=np.float32)
        faces = np.asarray(self.mesh.faces)

        projected, _ = cv2.projectPoints(vertices, rvec, tvec,
                                         self.camera_matrix, self.dist_coeffs)
        projected = projected.reshape(-1, 2)

        # Decide colors
        if self.mtl_colors:  # preferir MTL
            # aplica a primeira cor do MTL (simplificado, 1 material)
            color_list = list(self.mtl_colors.values())
            base_color = np.array(color_list[0][::-1])  # RGB->BGR
            face_colors = np.tile(base_color, (len(faces), 1))
        elif hasattr(self.mesh.visual, "face_colors") and len(self.mesh.visual.face_colors) > 0:
            face_colors = self.mesh.visual.face_colors[:, :3]
            if np.all(face_colors == 0):
                pig_pink = np.array([203, 192, 255])
                face_colors = np.tile(pig_pink, (len(faces), 1))
        else:
            pig_pink = np.array([203, 192, 255])
            face_colors = np.tile(pig_pink, (len(faces), 1))

        # Painter’s Algorithm
        R, _ = cv2.Rodrigues(rvec)
        verts_cam = (R @ vertices.T + tvec).T
        z_means = [np.mean(verts_cam[face, 2]) for face in faces]
        face_order = np.argsort(z_means)[::-1]

        # Render
        for fi in face_order:
            face = faces[fi]
            pts = projected[face].astype(int).reshape(-1, 1, 2)
            color = tuple(int(c) for c in face_colors[fi])
            cv2.fillPoly(frame, [pts], color)
            cv2.polylines(frame, [pts], True, (0, 0, 0), 1)

    # -------------------------------------------------------------------------
    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = self.detect_markers(gray)

        if ids is not None:
            ids_flat = ids.flatten()
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Guarda todos os rvecs/tvecs
            rvecs, tvecs = {}, {}
            marker_corners = {}
            for corner, marker_id in zip(corners, ids_flat):
                rvec, tvec = self.estimate_pose(corner)
                rvecs[marker_id] = rvec
                tvecs[marker_id] = tvec
                marker_corners[marker_id] = corner
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.05)

            # Se o marcador base foi detectado
            # Se o marcador base foi detectado
            if self.base_marker_id in rvecs:
                rvec_base = rvecs[self.base_marker_id]
                tvec_base = tvecs[self.base_marker_id]
                base_corners_img2d = marker_corners[self.base_marker_id].reshape(-1, 2)

                # --- altura relativa: usa diferença entre base(id=1) e id=0 ---
                if 0 in tvecs:
                    rel_pos = tvec_base - tvecs[0]
                    bbox_height = abs(float(rel_pos[1, 0]))  # eixo Y na câmera
                else:
                    bbox_height = float(self.marker_length * 0.5)  # fallback

                # desenha bbox
                bbox_points_3d_local = self.draw_bbox(
                    frame, rvec_base, tvec_base, base_corners_img2d, bbox_height
                )

                # imprime dimensões
                w, d, h = self.bbox_dimensions(bbox_points_3d_local)
                print(f"BBox dimensions -> W: {w:.3f} m, D: {d:.3f} m, H: {h:.3f} m")

                # desenha o objeto (mantém escala fixa definida no __init__)
                self.draw_obj(frame, rvec_base, tvec_base)

        return frame
