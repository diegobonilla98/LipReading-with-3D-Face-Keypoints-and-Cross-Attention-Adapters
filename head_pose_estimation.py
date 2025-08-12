# pip install opencv-python mediapipe numpy
import cv2
import numpy as np
import mediapipe as mp
import time

# ---------- Helpers ----------
def _lm_to_pixels(landmarks, image_size):
    """Convert normalized landmarks (0..1) to pixel coords (N,2)."""
    h, w = image_size
    arr = []
    for p in landmarks:
        arr.append([p.x * w, p.y * h])
    return np.asarray(arr, dtype=np.float64)

def _rotation_matrix_to_euler_xyz(R):
    """
    Convert 3x3 rotation matrix to Euler angles (pitch=x, yaw=y, roll=z) [radians]
    OpenCV camera coords: x right, y down, z forward.
    """
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        pitch = np.arctan2(R[2,1], R[2,2])   # x
        yaw   = np.arctan2(-R[2,0], sy)      # y
        roll  = np.arctan2(R[1,0], R[0,0])   # z
    else:
        pitch = np.arctan2(-R[1,2], R[1,1])
        yaw   = np.arctan2(-R[2,0], sy)
        roll  = 0.0
    return pitch, yaw, roll

def solve_head_pose(pix, image_size, dist_coeffs=None):
    """
    pix: (N,2) pixel coords for all MP landmarks (468/478). Uses six stable points.
    Returns yaw/pitch/roll (deg), rvec, tvec, camera_matrix, dist_coeffs.
    """
    idx = {
        "nose_tip": 1,
        "chin": 152,
        "left_eye_outer": 33,
        "right_eye_outer": 263,
        "left_mouth": 61,
        "right_mouth": 291,
    }

    # Generic 3D model (approx mm). Origin at nose tip.
    model_points = np.array([
        [ 0.0,   0.0,   0.0  ],   # nose tip
        [ 0.0, -63.6, -12.5 ],    # chin
        [-43.3, 32.7, -26.0 ],    # left eye outer
        [ 43.3, 32.7, -26.0 ],    # right eye outer
        [-28.9,-28.9, -24.1 ],    # left mouth
        [ 28.9,-28.9, -24.1 ],    # right mouth
    ], dtype=np.float64)

    image_points = np.array([
        pix[idx["nose_tip"]],
        pix[idx["chin"]],
        pix[idx["left_eye_outer"]],
        pix[idx["right_eye_outer"]],
        pix[idx["left_mouth"]],
        pix[idx["right_mouth"]],
    ], dtype=np.float64)

    h, w = image_size
    f = max(w, h)
    camera_matrix = np.array([[f, 0, w/2.0],
                              [0, f, h/2.0],
                              [0, 0, 1.0]], dtype=np.float64)
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4,1), dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None

    R, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = _rotation_matrix_to_euler_xyz(R)
    return {
        "yaw":   float(np.degrees(yaw)),
        "pitch": float(np.degrees(pitch)),
        "roll":  float(np.degrees(roll)),
        "rvec": rvec, "tvec": tvec,
        "K": camera_matrix, "D": dist_coeffs
    }

def draw_axes_gizmo(frame, rvec, tvec, K, D, axis_len=60):
    """
    Draw 3D axes starting at the model origin (nose tip).
    X (right)=red, Y (down)=green, Z (forward)=blue.
    """
    axes_3d = np.float64([
        [0, 0, 0],                 # origin at nose
        [axis_len, 0, 0],          # +X
        [0, axis_len, 0],          # +Y
        [0, 0, axis_len],          # +Z
    ])
    pts2d, _ = cv2.projectPoints(axes_3d, rvec, tvec, K, D)
    pts2d = pts2d.reshape(-1, 2).astype(int)
    o, x, y, z = pts2d[0], pts2d[1], pts2d[2], pts2d[3]
    cv2.line(frame, tuple(o), tuple(x), (0, 0, 255), 3)   # X - red
    cv2.line(frame, tuple(o), tuple(y), (0, 255, 0), 3)   # Y - green
    cv2.line(frame, tuple(o), tuple(z), (255, 0, 0), 3)   # Z - blue
    # small dot at origin
    cv2.circle(frame, tuple(o), 4, (255,255,255), -1)

# ---------- Main ----------
def main():
    cap = cv2.VideoCapture(0)
    # Optional: set your preferred capture size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        refine_landmarks=True,   # gives 478 landmarks
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    prev_t = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                pix = _lm_to_pixels(lm, (h, w))

                solved = solve_head_pose(pix, (h, w))
                if solved is not None:
                    yaw, pitch, roll = solved["yaw"], solved["pitch"], solved["roll"]
                    rvec, tvec, K, D = solved["rvec"], solved["tvec"], solved["K"], solved["D"]

                    # Draw axes gizmo at nose tip
                    draw_axes_gizmo(frame, rvec, tvec, K, D, axis_len=int(0.08 * max(w, h)))

                    # Overlay angles
                    txt = f"Yaw: {yaw:+6.1f}  Pitch: {pitch:+6.1f}  Roll: {roll:+6.1f}"
                    cv2.rectangle(frame, (10, 10), (10+550, 10+36), (0,0,0), -1)
                    cv2.putText(frame, txt, (18, 36),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

            # FPS
            now = time.time()
            fps = 1.0 / (now - prev_t)
            prev_t = now
            cv2.putText(frame, f"{fps:5.1f} FPS", (w-150, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2, cv2.LINE_AA)

            cv2.imshow("Head Pose (MediaPipe + PnP)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
