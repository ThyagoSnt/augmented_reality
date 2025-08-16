import numpy as np

class CameraMathUtils:
    """
    Utility class implementing fundamental camera geometry functions,
    based strictly on Hartley & Zisserman (Multiple View Geometry, 2nd Edition).
    """

    @staticmethod
    def rodrigues(rvec):
        """
        Manual implementation of Rodrigues' rotation formula.
        Converts a 3D rotation vector into a 3x3 rotation matrix.

        Reference:
        - H&Z Appendix A, Exponential map for SO(3).
        """
        rvec = rvec.flatten().astype(float)
        theta = np.linalg.norm(rvec)

        if theta < 1e-12:
            # For very small angles, return identity
            R = np.eye(3)
        else:
            # Normalized rotation axis
            k = rvec / theta
            kx, ky, kz = k

            # Skew-symmetric cross-product matrix of k
            K = np.array([
                [0, -kz, ky],
                [kz, 0, -kx],
                [-ky, kx, 0]
            ])

            I = np.eye(3)
            # Rodrigues formula: R = I cosθ + (1-cosθ)kk^T + sinθK
            R = I * np.cos(theta) + (1 - np.cos(theta)) * np.outer(k, k) + np.sin(theta) * K

        # Jacobian not needed here
        jacobian = None
        return R, jacobian, rvec


    @staticmethod
    def l2_norm_manual(points1, points2):
        """
        Manual implementation of L2 norm (equivalent to cv2.norm with cv2.NORM_L2).
        
        Based on H&Z Section 6.3.1:
        The geometric error is defined as the Euclidean distance between
        observed image points and reprojected points.
        """
        p1 = np.squeeze(points1)
        p2 = np.squeeze(points2)

        diff = p1 - p2
        squared_dist = np.sum(diff ** 2, axis=-1)
        total = np.sum(squared_dist)

        return np.sqrt(total)


    @staticmethod
    def project_points(object_points, rvec, tvec, camera_matrix, dist_coeffs,
                       compute_jacobian=False, eps=1e-6):
        """
        Manual projectPoints with optional numerical Jacobian.
        """
        # --- inner projection function ---
        def _project(rvec, tvec, K, d):
            theta = np.linalg.norm(rvec)
            if theta < 1e-14:
                R = np.eye(3)
            else:
                r = rvec.flatten() / theta
                Kmat = np.array([
                    [0, -r[2], r[1]],
                    [r[2], 0, -r[0]],
                    [-r[1], r[0], 0]
                ])
                R = np.eye(3) + np.sin(theta) * Kmat + (1 - np.cos(theta)) * (Kmat @ Kmat)

            X = np.asarray(object_points).reshape(-1, 3).T
            X_cam = (R @ X) + tvec.reshape(3, 1)
            x = X_cam[0, :] / X_cam[2, :]
            y = X_cam[1, :] / X_cam[2, :]
            r2 = x**2 + y**2
            k1, k2, p1, p2, k3 = d.flatten()
            radial = 1 + k1*r2 + k2*r2**2 + k3*r2**3
            x_dist = x*radial + 2*p1*x*y + p2*(r2 + 2*x**2)
            y_dist = y*radial + p1*(r2 + 2*y**2) + 2*p2*x*y
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            u = fx * x_dist + cx
            v = fy * y_dist + cy
            return np.stack([u, v], axis=1).reshape(-1)

        # --- base projection ---
        img_points_flat = _project(rvec, tvec, camera_matrix, dist_coeffs)
        image_points = img_points_flat.reshape(-1, 1, 2)

        # --- numerical Jacobian ---
        if not compute_jacobian:
            return image_points, None

        params = np.hstack([rvec.flatten(), tvec.flatten(), dist_coeffs.flatten()])
        M = len(params)
        N = len(img_points_flat)
        J = np.zeros((N, M))

        for i in range(M):
            dp = np.zeros_like(params)
            dp[i] = eps
            params_eps = params + dp
            rvec_eps = params_eps[:3]
            tvec_eps = params_eps[3:6]
            d_eps = params_eps[6:]
            img_points_eps = _project(rvec_eps, tvec_eps, camera_matrix, d_eps)
            J[:, i] = (img_points_eps - img_points_flat) / eps

        return image_points, J