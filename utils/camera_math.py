import numpy as np


class CameraMathUtils:
    """
    Utility class implementing fundamental camera geometry functions,
    based strictly on Hartley & Zisserman (Multiple View Geometry, 2nd Edition).
    """

    # -------------------------------------------------------------------------
    # Rotation utilities
    # -------------------------------------------------------------------------
    @staticmethod
    def rodrigues(rvec: np.ndarray):
        """
        Rodrigues' rotation formula (vector -> rotation matrix).
        Returns (R, None, rvec) to stay compatible with OpenCV-like signature.

        Reference:
        - H&Z, Appendix A: Exponential map for SO(3).
        """
        rvec = np.asarray(rvec, dtype=float).reshape(3)
        theta = np.linalg.norm(rvec)

        if theta < 1e-12:
            R = np.eye(3, dtype=float)
        else:
            k = rvec / theta
            kx, ky, kz = k
            K = np.array([[0, -kz, ky],
                          [kz, 0, -kx],
                          [-ky, kx, 0]], dtype=float)
            I = np.eye(3, dtype=float)
            # R = I cosθ + (1-cosθ) kk^T + sinθ K
            R = I * np.cos(theta) + (1.0 - np.cos(theta)) * np.outer(k, k) + np.sin(theta) * K

        jacobian = None
        return R, jacobian, rvec

    @staticmethod
    def rotation_matrix_to_rvec(R: np.ndarray) -> np.ndarray:
        """
        Inverse Rodrigues: rotation matrix -> rotation vector.

        Reference:
        - H&Z, Appendix A; also classic SO(3) logarithm map.
        """
        R = np.asarray(R, dtype=float).reshape(3, 3)
        tr = float(np.trace(R))
        # Clip for numeric safety
        cos_theta = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        if theta < 1e-12:
            return np.zeros((3, ), dtype=float)

        if np.pi - theta < 1e-6:
            # Near 180 degrees: use robust extraction from diagonal
            rx = np.sqrt(max((R[0, 0] + 1.0) / 2.0, 0.0))
            ry = np.sqrt(max((R[1, 1] + 1.0) / 2.0, 0.0))
            rz = np.sqrt(max((R[2, 2] + 1.0) / 2.0, 0.0))
            rx = np.copysign(rx, R[2, 1] - R[1, 2])
            ry = np.copysign(ry, R[0, 2] - R[2, 0])
            rz = np.copysign(rz, R[1, 0] - R[0, 1])
            axis = np.array([rx, ry, rz], dtype=float)
            n = np.linalg.norm(axis) + 1e-12
            axis /= n
            return axis * theta

        axis = np.array([R[2, 1] - R[1, 2],
                         R[0, 2] - R[2, 0],
                         R[1, 0] - R[0, 1]], dtype=float) / (2.0 * np.sin(theta))
        axis /= (np.linalg.norm(axis) + 1e-12)
        return axis * theta

    # -------------------------------------------------------------------------
    # Error / norms
    # -------------------------------------------------------------------------
    @staticmethod
    def l2_norm_manual(points1: np.ndarray, points2: np.ndarray) -> float:
        """
        Manual L2 norm equivalent to cv2.norm(..., NORM_L2) over full arrays.

        H&Z 6.3.1: geometric error as Euclidean distance between observed
        and reprojected image points.
        """
        p1 = np.squeeze(points1)
        p2 = np.squeeze(points2)
        diff = p1 - p2
        return float(np.sqrt(np.sum(diff * diff)))

    # -------------------------------------------------------------------------
    # Projection
    # -------------------------------------------------------------------------
    @staticmethod
    def project_points(object_points: np.ndarray,
                       rvec: np.ndarray,
                       tvec: np.ndarray,
                       camera_matrix: np.ndarray,
                       dist_coeffs: np.ndarray,
                       compute_jacobian: bool = False,
                       eps: float = 1e-6):
        """
        Manual version of cv2.projectPoints with optional numerical Jacobian.

        Distortion model: (k1, k2, p1, p2, k3).

        References:
        - H&Z 6.1–6.3 (pinhole projection and distortion models).
        """
        def _parse_distortion(d: np.ndarray):
            d = np.asarray(d, dtype=float).reshape(-1)
            # Accept 4 or 5 parameters; pad k3 if missing
            if d.size == 4:
                k1, k2, p1, p2 = d
                k3 = 0.0
            elif d.size >= 5:
                k1, k2, p1, p2, k3 = d[:5]
            else:
                # No distortion provided
                k1 = k2 = p1 = p2 = k3 = 0.0
            return k1, k2, p1, p2, k3

        # --- base projection (no Jacobian) ---
        def _project(rvec_, tvec_, K_, d_):
            # Rotation
            R_, _, _ = CameraMathUtils.rodrigues(rvec_)
            X = np.asarray(object_points, dtype=float).reshape(-1, 3).T  # (3, N)
            X_cam = (R_ @ X) + np.asarray(tvec_, dtype=float).reshape(3, 1)

            x = X_cam[0, :] / X_cam[2, :]
            y = X_cam[1, :] / X_cam[2, :]
            r2 = x * x + y * y

            k1, k2, p1, p2, k3 = _parse_distortion(d_)
            radial = 1.0 + k1 * r2 + k2 * (r2 ** 2) + k3 * (r2 ** 3)
            x_dist = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            y_dist = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

            fx, fy = K_[0, 0], K_[1, 1]
            cx, cy = K_[0, 2], K_[1, 2]
            u = fx * x_dist + cx
            v = fy * y_dist + cy
            return np.stack([u, v], axis=1).reshape(-1)

        img_points_flat = _project(rvec, tvec, camera_matrix, dist_coeffs)
        image_points = img_points_flat.reshape(-1, 1, 2)

        if not compute_jacobian:
            return image_points, None

        # --- numerical Jacobian wrt [rvec(3), tvec(3), dist(5)] ---
        d = np.asarray(dist_coeffs, dtype=float).reshape(-1)
        if d.size < 5:
            d = np.pad(d, (0, 5 - d.size), constant_values=0.0)

        params = np.hstack([np.asarray(rvec, dtype=float).reshape(3),
                            np.asarray(tvec, dtype=float).reshape(3),
                            d[:5]])
        M = params.size
        N = img_points_flat.size
        J = np.zeros((N, M), dtype=float)

        for i in range(M):
            dp = np.zeros_like(params)
            dp[i] = eps
            rvec_eps = params[:3] + dp[:3]
            tvec_eps = params[3:6] + dp[3:6]
            dist_eps = params[6:] + dp[6:]
            img_eps = _project(rvec_eps, tvec_eps, camera_matrix, dist_eps)
            J[:, i] = (img_eps - img_points_flat) / eps

        return image_points, J

    # -------------------------------------------------------------------------
    # Homography-based planar PnP
    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_points_2d(pts: np.ndarray):
        """
        Hartley normalization (H&Z Sec. 4.4): translate to centroid and scale
        so mean distance to the origin equals sqrt(2).
        Returns normalized homogeneous points (N,3) and the normalization matrix T (3,3).
        """
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        c = pts.mean(axis=0)
        d = np.linalg.norm(pts - c, axis=1).mean()
        s = np.sqrt(2.0) / max(d, 1e-12)
        T = np.array([[s, 0.0, -s * c[0]],
                      [0.0, s, -s * c[1]],
                      [0.0, 0.0, 1.0]], dtype=float)
        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=float)])
        pts_n = (T @ pts_h.T).T
        return pts_n, T

    @staticmethod
    def homography_dlt(XY: np.ndarray, uv: np.ndarray) -> np.ndarray:
        """
        Direct Linear Transform for planar homography with normalization.
        XY: (N,2) plane points; uv: (N,2) image points. N >= 4.

        Reference: H&Z Sec. 4.1 and 4.4.
        """
        XYn, T1 = CameraMathUtils.normalize_points_2d(XY)
        uvn, T2 = CameraMathUtils.normalize_points_2d(uv)

        A = []
        for (X, Y, _), (u, v, _) in zip(XYn, uvn):
            A.append([0, 0, 0, -X, -Y, -1, v * X, v * Y, v])
            A.append([X, Y, 1, 0, 0, 0, -u * X, -u * Y, -u])
        A = np.asarray(A, dtype=float)

        _, _, Vt = np.linalg.svd(A)
        Hn = Vt[-1].reshape(3, 3)
        H = np.linalg.inv(T2) @ Hn @ T1
        if abs(H[2, 2]) > 1e-12:
            H /= H[2, 2]
        return H

    @staticmethod
    def pose_from_homography(H: np.ndarray, K: np.ndarray):
        """
        Recover pose [R|t] from homography with known intrinsics K.

        K^{-1} H = [λ r1 | λ r2 | λ t], enforce orthonormality via SVD.

        Reference: H&Z Sec. 7.2 (pose from calibrated homography).
        """
        K = np.asarray(K, dtype=float).reshape(3, 3)
        Kinv = np.linalg.inv(K)
        B = Kinv @ H
        b1, b2, b3 = B[:, 0], B[:, 1], B[:, 2]

        lam = 0.5 * (1.0 / (np.linalg.norm(b1) + 1e-12) +
                     1.0 / (np.linalg.norm(b2) + 1e-12))
        r1 = b1 * lam
        r2 = b2 * lam
        r3 = np.cross(r1, r2)

        R_approx = np.column_stack([r1, r2, r3])
        U, _, Vt = np.linalg.svd(R_approx)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt

        t = b3 * lam
        return R, t

    @staticmethod
    def solve_pnp(obj_points: np.ndarray,
                  img_points: np.ndarray,
                  camera_matrix: np.ndarray):
        """
        Planar PnP via homography (DLT + Hartley normalization) with known intrinsics.

        Args:
            obj_points: (N,3) object points on the marker plane (Z=0).
            img_points: (N,2) corresponding image points.
            camera_matrix: (3,3) intrinsic matrix K.

        Returns:
            rvec: (3,1) Rodrigues rotation vector.
            tvec: (3,1) translation vector.

        References:
        - H&Z Sec. 4.1 (plane-induced homography), Sec. 4.4 (normalization),
          Sec. 7.2 (pose from calibrated homography).
        """
        obj_points = np.asarray(obj_points, dtype=float).reshape(-1, 3)
        img_points = np.asarray(img_points, dtype=float).reshape(-1, 2)
        K = np.asarray(camera_matrix, dtype=float).reshape(3, 3)

        if obj_points.shape[0] < 4:
            raise ValueError("Need at least 4 correspondences for homography-based PnP.")
        if not np.allclose(obj_points[:, 2], 0.0, atol=1e-12):
            raise ValueError("Planar PnP requires Z=0 for all object points.")

        XY = obj_points[:, :2]
        uv = img_points

        H = CameraMathUtils.homography_dlt(XY, uv)
        R, t = CameraMathUtils.pose_from_homography(H, K)
        rvec = CameraMathUtils.rotation_matrix_to_rvec(R).reshape(3, 1)
        tvec = t.reshape(3, 1)
        return rvec, tvec
