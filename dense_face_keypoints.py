from dataclasses import dataclass
from typing import Optional, Literal
import numpy as np

# Core plotting (optional at runtime)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

# Image IO
import cv2

from head_pose_estimation import solve_head_pose

# Remove logging and debugging classes and functions

# Only MediaPipe backend is supported now
def _to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr is None:
        raise ValueError("Input image is None. Check the path or cv2.imread result.")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR image with 3 channels, got shape {image_bgr.shape}.")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _axes_equal_3d(ax):
    # Set equal scale for 3D axes
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_mid = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_mid = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_mid = np.mean(z_limits)
    max_range = max([x_range, y_range, z_range])
    ax.set_xlim3d([x_mid - max_range/2, x_mid + max_range/2])
    ax.set_ylim3d([y_mid - max_range/2, y_mid + max_range/2])
    ax.set_zlim3d([z_mid - max_range/2, z_mid + max_range/2])
    # Modern Matplotlib: also good to enforce a cube aspect
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


@dataclass
class Face3DKeypointExtractor:
    """
    Extract dense 3D facial keypoints and normalize them using MediaPipe Face Mesh.

    - Uses MediaPipe Face Mesh: ~468-478 3D landmarks (dense). Fast and great quality.

    Normalization:
      - "unit_sphere": center at centroid, scale so max distance to centroid = 1.
      - "bbox_unit": center at centroid, scale so the largest bbox side = 1.
      - "zscore": per-axis standardization (mean=0, std=1); robust but shape-biased.

    Returns Nx3 ndarray, float32.
    """
    normalize: Literal["unit_sphere", "bbox_unit", "zscore"] = "unit_sphere"
    frontalize: bool = True  # rotate landmarks to face front while preserving translation

    _mp_facemesh: Optional[object] = None

    def _lazy_init_mediapipe(self):
        if self._mp_facemesh is not None:
            return
        try:
            import mediapipe as mp
        except ImportError as e:
            raise ImportError(
                "mediapipe is not installed. Install with `pip install mediapipe`."
            ) from e

        self._mp = mp
        self._mp_facemesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,         # adds iris + lips refinement (-> ~478 pts)
            max_num_faces=1,
            min_detection_confidence=0.5
        )

    def extract(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Extract normalized 3D landmarks from the input image (BGR).
        Returns: (N, 3) float32 array in normalized coordinates.
        """
        pts = self._extract_mediapipe(image_bgr)
        return self._normalize_points(pts, method=self.normalize)

    def _extract_mediapipe(self, image_bgr: np.ndarray) -> np.ndarray:
        self._lazy_init_mediapipe()
        rgb = _to_rgb(image_bgr)
        h, w = rgb.shape[:2]

        result = self._mp_facemesh.process(rgb)
        if not result.multi_face_landmarks:
            raise RuntimeError("No face detected by MediaPipe Face Mesh.")

        # Take the first face
        lm = result.multi_face_landmarks[0]
        pts = []
        pix2d = []
        for p in lm.landmark:
            # MediaPipe x,y are normalized to [0,1] relative to image size; z to roughly face width
            x_pix = p.x * w
            y_pix = p.y * h
            z_pix = p.z * w  # scale z to pixel-ish scale comparable to x
            pix2d.append([x_pix, y_pix])  # y downwards (OpenCV image coords)
            # Our 3D coordinate convention: flip Y to point up, invert Z to point towards camera
            pts.append([x_pix, h - y_pix, -z_pix])
        pts = np.asarray(pts, dtype=np.float32)

        if self.frontalize:
            try:
                pts = self._frontalize_mediapipe_points(
                    pts=pts,
                    pix2d=np.asarray(pix2d, dtype=np.float64),
                    image_size=(h, w)
                )
            except Exception:
                # In case pose solving fails, fall back to raw points
                pass

        return pts

    @staticmethod
    def _frontalize_mediapipe_points(pts: np.ndarray,
                                     pix2d: np.ndarray,
                                     image_size: tuple) -> np.ndarray:
        """
        Rotate points so the face looks forward, preserving translation (anchor at nose tip).
        Inputs:
          - pts: (N,3) in our convention (X right, Y up, Z towards camera) [float32]
          - pix2d: (N,2) pixel coords (x right, y down)
          - image_size: (h, w)
        Returns: (N,3) float32
        """
        solved = solve_head_pose(pix2d, image_size)
        if solved is None:
            return pts

        rvec = solved["rvec"]
        R, _ = cv2.Rodrigues(rvec)  # rotation from model to camera (OpenCV coords: x right, y down, z forward)

        # Work relative to the nose tip to preserve translation
        nose_index = 1  # MediaPipe nose tip index
        nose_point = pts[nose_index].copy()

        # Our coords are related to OpenCV camera coords by a flip: F = diag([1, -1, -1]) for relative vectors
        F = np.diag([1.0, -1.0, -1.0]).astype(np.float32)

        rel_prime = pts - nose_point  # in our coords (X right, Y up, Z towards camera)
        rel_cam = (F @ rel_prime.T).T  # convert to camera coords (X right, Y down, Z forward)

        # Undo face rotation: apply inverse rotation
        rel_cam_frontal = (R.T @ rel_cam.T).T

        # Convert back to our coords and restore translation
        rel_prime_frontal = (F @ rel_cam_frontal.T).T
        frontal_pts = rel_prime_frontal + nose_point
        return frontal_pts.astype(np.float32)

    @staticmethod
    def _normalize_points(pts: np.ndarray, method: Literal["unit_sphere", "bbox_unit", "zscore"]) -> np.ndarray:
        p = pts.copy().astype(np.float32)
        if method == "unit_sphere":
            c = p.mean(axis=0, keepdims=True)
            p -= c
            r = np.linalg.norm(p, axis=1).max()
            if r < 1e-8:
                return p  # degenerate
            p /= r
            return p
        elif method == "bbox_unit":
            c = p.mean(axis=0, keepdims=True)
            p -= c
            ranges = p.max(axis=0) - p.min(axis=0)
            s = float(np.max(ranges))
            if s < 1e-8:
                return p
            p /= s
            return p
        elif method == "zscore":
            mu = p.mean(axis=0, keepdims=True)
            std = p.std(axis=0, keepdims=True) + 1e-8
            return (p - mu) / std
        else:
            raise ValueError(f"Unknown normalization method: {method}")


def plot_3d_points(points: np.ndarray,
                   title: str = "3D Face Landmarks",
                   elev: float = 15,
                   azim: float = -70,
                   show: bool = True,
                   ax: Optional[plt.Axes] = None,
                   s: int = 8):
    """
    Simple 3D scatter plot for landmarks.
    `points` is (N, 3) in any coordinate system (already normalized by the extractor).
    """
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=s)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)
    _axes_equal_3d(ax)
    if show:
        plt.show()


def main():
    # ---- EDIT THIS PATH ----
    IMG_PATH = r"D:\TalkingFaces\fadg0\fadg0\video\head\001"  # put your test image here
    # ------------------------

    img = cv2.imread(IMG_PATH)
    if img is None:
        exit()

    # Only MediaPipe (dense, ~478 pts) is supported
    try:
        mp_extractor = Face3DKeypointExtractor(
            normalize="unit_sphere"
        )
        mp_pts = mp_extractor.extract(img)
        plot_3d_points(mp_pts, title="MediaPipe Face Mesh (normalized)")
    except Exception:
        pass


if __name__ == "__main__":
    main()
