import os
import sys
import time
import ctypes
import sqlite3
import threading
import subprocess
from collections import deque
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

# MediaPipe is used for facial landmarks / approximate eye-gaze features.
# The app can still run without it, but eye-gaze detection will be disabled.
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")
except Exception as exc:
    mp = None
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe is not available:", exc)

# ================= Paths (script / PyInstaller exe) =================
FROZEN = getattr(sys, "frozen", False)
if FROZEN:
    BASE_DIR = os.path.dirname(sys.executable)
    RES_DIR = getattr(sys, "_MEIPASS", BASE_DIR)
else:
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE_DIR = os.getcwd()
    RES_DIR = BASE_DIR


def find_file(name):
    for directory in (BASE_DIR, RES_DIR):
        candidate = os.path.join(directory, name)
        if os.path.exists(candidate):
            return candidate
    return name


# ================= Settings =================
SCREENSHOT_DIR = os.path.join(BASE_DIR, "alerts", "screenshots")
DB_PATH = os.path.join(BASE_DIR, "alerts", "exam_alerts.db")
ICON_PATH = find_file("icon.ico")

POSE_MODEL = find_file("yolov8s-pose.pt")
OBJECT_MODEL = find_file("yolov8n.pt")
TRACKER = "botsort.yaml"

# Speed settings
PROCESS_EVERY_N = 2
OBJECT_EVERY_N = 3
DISPLAY_W, DISPLAY_H = 800, 500

# Detection settings
PHONE_CLASS, BOOK_CLASS = 67, 73  # COCO class IDs: cell phone=67, book=73
OBJ_CONF = 0.35
KPT_CONF = 0.40
YAW_THRESHOLD = 0.20
DOWN_RATIO = 0.25
SMOOTH_WINDOW = 7

# The first 1 seconds for each tracked student are treated as that student's normal face/head pose (baseline). Validate thresholds on your videos.
GAZE_THRESHOLD = 0.11       # eye-gaze change relative to the student's own baseline
ROLL_THRESHOLD = 10.0       # head-tilt change in degrees relative to baseline
DOWN_DELTA_THRESHOLD = 0.16 # vertical face-pose change relative to baseline
CALIBRATION_SECONDS = 1   # first seconds are assumed to show the normal pose
CALIBRATION_MIN_SAMPLES = 5
BODY_LEAN_THRESHOLD = 0.08
BODY_LEAN_SECONDS = 1.0
ALERT_ONLY_WHEN_NEIGHBOR = True
HEAD_DOWN_WHEN_FACE_HIDDEN = False
DEBUG = False

SECONDS_REQUIRED = {
    "Phone Detected": 0.4,
    "Book Detected": 0.5,
    "Peeking at Neighbor": 0.3,
    "Looking Away": 1.0,
    "Looking Down": 3.0,
    "Head Down": 3.0,
}
DEFAULT_SECONDS = 1.0
WARNING_ONLY = {"Looking Down", "Head Down"}

# Firebase is optional and disabled by default.
USE_FIREBASE = False
FIREBASE_KEY = os.path.join(BASE_DIR, "serviceAccountKey.json")
FIREBASE_BUCKET = "your-project-id.appspot.com"

os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


# ================= Database =================
def init_database():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            behavior TEXT,
            confidence TEXT,
            timestamp TEXT,
            image_path TEXT
        )
    ''')
    existing = {row[1] for row in conn.execute("PRAGMA table_info(alerts)")}
    if "video_time" not in existing:
        conn.execute("ALTER TABLE alerts ADD COLUMN video_time TEXT")
    if "firebase_url" not in existing:
        conn.execute("ALTER TABLE alerts ADD COLUMN firebase_url TEXT")
    conn.commit()
    conn.close()


def save_alert(student_id, behavior, confidence, timestamp, video_time, image_path):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "INSERT INTO alerts (student_id, behavior, confidence, timestamp, video_time, image_path) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (str(student_id), behavior, confidence, timestamp, video_time, image_path),
    )
    conn.commit()
    alert_id = cur.lastrowid
    conn.close()
    return alert_id


def set_firebase_url(alert_id, url):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE alerts SET firebase_url = ? WHERE id = ?", (url, alert_id))
    conn.commit()
    conn.close()


# ================= Firebase (Firestore + Storage) =================
_fb = {"db": None, "bucket": None}


def init_firebase():
    if not USE_FIREBASE:
        return False
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore, storage
        cred = credentials.Certificate(FIREBASE_KEY)
        firebase_admin.initialize_app(cred, {"storageBucket": FIREBASE_BUCKET})
        _fb["db"] = firestore.client()
        _fb["bucket"] = storage.bucket()
        print("Firebase connected.")
        return True
    except Exception as exc:
        print("Firebase disabled:", exc)
        return False


def upload_to_firebase(alert_id, record, image_path):
    if _fb["db"] is None:
        return
    try:
        from firebase_admin import firestore
        name = os.path.basename(image_path)
        blob = _fb["bucket"].blob(f"screenshots/{name}")
        blob.upload_from_filename(image_path)
        try:
            blob.make_public()
            url = blob.public_url
        except Exception:
            url = f"gs://{_fb['bucket'].name}/screenshots/{name}"
        _fb["db"].collection("alerts").add(
            {**record, "screenshot_url": url, "created_at": firestore.SERVER_TIMESTAMP}
        )
        set_firebase_url(alert_id, url)
        print("Alert uploaded to Firebase:", url)
    except Exception as exc:
        print("Firebase upload failed:", exc)


# ================= Helpers =================
def open_path(path):
    if sys.platform.startswith("win"):
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def save_image(path, frame):
    try:
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            return False
        buf.tofile(path)
        return True
    except Exception as exc:
        print("Failed to save screenshot:", exc)
        return False


def fmt_video_time(seconds):
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def safe_median(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else None


# ================= Pose analysis =================
def pose_metrics(kpts):
    """Estimate head yaw and down-look proxy from YOLO Pose keypoints."""
    nose, l_eye, r_eye = kpts[0], kpts[1], kpts[2]
    metrics = {"yaw": None, "down": None, "face_visible": bool(nose[2] >= KPT_CONF)}
    if nose[2] < KPT_CONF:
        return metrics

    l_ok, r_ok = l_eye[2] >= KPT_CONF, r_eye[2] >= KPT_CONF
    if l_ok and r_ok:
        eye_dist = abs(l_eye[0] - r_eye[0])
        if eye_dist < 8:
            return metrics
        eye_mid_x = (l_eye[0] + r_eye[0]) / 2
        eye_mid_y = (l_eye[1] + r_eye[1]) / 2
        metrics["yaw"] = float((nose[0] - eye_mid_x) / eye_dist)
        metrics["down"] = float((nose[1] - eye_mid_y) / eye_dist)
    elif l_ok != r_ok:
        eye = l_eye if l_ok else r_eye
        metrics["yaw"] = 1.0 if nose[0] > eye[0] else -1.0
    return metrics

def body_lean_metrics(kpts):
    """
    Estimate sideways body lean using shoulders and hips.
    COCO Pose keypoints:
    left shoulder = 5, right shoulder = 6,
    left hip = 11, right hip = 12.
    """
    metrics = {
        "body_lean": None,
        "shoulders_visible": False
    }

    left_shoulder = kpts[5]
    right_shoulder = kpts[6]

    if (
        left_shoulder[2] < KPT_CONF
        or right_shoulder[2] < KPT_CONF
    ):
        return metrics

    # Calculate the center of the shoulders.
    shoulder_x = (
        float(left_shoulder[0]) +
        float(right_shoulder[0])
    ) / 2

    shoulder_y = (
        float(left_shoulder[1]) +
        float(right_shoulder[1])
    ) / 2

    metrics["shoulders_visible"] = True

    # Check whether both hips are visible.
    left_hip = kpts[11]
    right_hip = kpts[12]

    if (
        left_hip[2] >= KPT_CONF
        and right_hip[2] >= KPT_CONF
    ):
        hip_x = (
            float(left_hip[0]) +
            float(right_hip[0])
        ) / 2

        hip_y = (
            float(left_hip[1]) +
            float(right_hip[1])
        ) / 2

        # Normalize horizontal displacement by body height.
        body_height = abs(hip_y - shoulder_y)

        if body_height >= 10:
            metrics["body_lean"] = (
                (shoulder_x - hip_x) / body_height
            )

    return metrics


def extract_face_crop(frame, box, kpts):
    """Crop an enlarged upper-face region using pose keypoints where possible."""
    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_w, x2), min(frame_h, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 2 or bh < 2:
        return None

    # Face keypoints are nose=0, left eye=1, right eye=2 in COCO Pose.
    nose, left_eye, right_eye = kpts[0], kpts[1], kpts[2]
    visible_eyes = [p for p in (left_eye, right_eye) if p[2] >= KPT_CONF]
    if nose[2] >= KPT_CONF and visible_eyes:
        eye_xs = [float(p[0]) for p in visible_eyes]
        eye_ys = [float(p[1]) for p in visible_eyes]
        center_x = float(np.mean(eye_xs))
        center_y = float(np.mean(eye_ys))
        if len(visible_eyes) == 2:
            eye_dist = max(8.0, abs(float(left_eye[0] - right_eye[0])))
        else:
            eye_dist = max(12.0, bw * 0.12)
        crop_w = int(max(3.8 * eye_dist, bw * 0.32))
        crop_h = int(max(4.4 * eye_dist, bh * 0.22))
        cx1 = int(center_x - crop_w / 2)
        cx2 = int(center_x + crop_w / 2)
        cy1 = int(center_y - 0.85 * eye_dist)
        cy2 = int(max(center_y + 2.8 * eye_dist, float(nose[1]) + 1.2 * eye_dist))
        # Make the crop tall enough to include forehead, eyes, nose and face outline.
        if cy2 - cy1 < crop_h:
            cy2 = cy1 + crop_h
    else:
        # Fallback: upper portion of the detected person's box.
        crop_w = int(max(0.40 * bw, 24))
        crop_h = int(max(0.32 * bh, 24))
        center_x = (x1 + x2) / 2
        cx1, cx2 = int(center_x - crop_w / 2), int(center_x + crop_w / 2)
        cy1, cy2 = y1, y1 + crop_h

    cx1, cy1 = max(0, cx1), max(0, cy1)
    cx2, cy2 = min(frame_w, cx2), min(frame_h, cy2)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    # Enlarge the crop so small but visible faces have a better chance of detection.
    return cv2.resize(crop, (256, 256), interpolation=cv2.INTER_CUBIC)


def face_metrics(face_mesh, frame, box, kpts):
    """Return approximate normalized iris position and head roll from Face Mesh.

    This is an approximate gaze cue, not precise eye tracking. Results depend on
    face size, lighting, pose, glasses and camera angle.
    """
    metrics = {"gaze_x": None, "roll": None, "face_landmarks_visible": False}
    if face_mesh is None:
        return metrics

    crop = extract_face_crop(frame, box, kpts)
    if crop is None or crop.shape[0] < 20 or crop.shape[1] < 20:
        return metrics

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    try:
        result = face_mesh.process(rgb)
    except Exception as exc:
        print("Face landmark processing failed:", exc)
        return metrics
    if not result.multi_face_landmarks:
        return metrics

    lm = result.multi_face_landmarks[0].landmark

    def xy(index):
        return np.array([lm[index].x, lm[index].y], dtype=float)

    # Face Mesh eye corners and iris points (refine_landmarks=True).
    iris_groups = ((468, 469, 470, 471, 472), (473, 474, 475, 476, 477))
    eye_corners = ((33, 133), (362, 263))
    iris_ratios = []
    eye_centers = []

    try:
        for iris_ids, corner_ids in zip(iris_groups, eye_corners):
            iris_x = float(np.mean([lm[idx].x for idx in iris_ids]))
            c1, c2 = xy(corner_ids[0]), xy(corner_ids[1])
            lo, hi = sorted((float(c1[0]), float(c2[0])))
            eye_width = hi - lo
            if eye_width < 0.008:
                continue
            ratio = (iris_x - lo) / eye_width
            # Avoid extreme ratios caused by partial occlusion / poor landmarks.
            if -0.25 <= ratio <= 1.25:
                iris_ratios.append(float(ratio))
            eye_centers.append((c1 + c2) / 2)
    except (IndexError, ValueError):
        return metrics

    if iris_ratios:
        metrics["gaze_x"] = float(np.mean(iris_ratios))

    # Roll: slope of the line between the two eye centers, sorted left-to-right
    # in image coordinates so the sign roughly corresponds to image-side tilt.
    if len(eye_centers) == 2:
        eye_centers.sort(key=lambda p: p[0])
        left_center, right_center = eye_centers
        dx = float(right_center[0] - left_center[0])
        dy = float(right_center[1] - left_center[1])
        if abs(dx) > 1e-4:
            metrics["roll"] = float(np.degrees(np.arctan2(dy, dx)))

    metrics["face_landmarks_visible"] = bool(iris_ratios)
    return metrics


def neighbor_in_direction(box, other_boxes, direction):
    """Check for a nearby tracked student in the requested image direction."""
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    center_x = (x1 + x2) / 2
    for ox1, oy1, ox2, oy2 in other_boxes:
        dx = ((ox1 + ox2) / 2 - center_x) * direction
        vertical_overlap = min(y2, oy2) - max(y1, oy1)
        other_h = oy2 - oy1
        if 0.15 * width < dx < 3.0 * width and vertical_overlap > 0.15 * min(height, other_h):
            return True
    return False


def head_conf(value, threshold):
    threshold = max(abs(float(threshold)), 1e-6)
    return float(min(0.99, 0.50 + 0.5 * abs(float(value)) / threshold * 0.25))



def update_calibration(cal, yaw, gaze_x, roll, down, body_lean, dt):
    """Calibrate each student's normal face, head and body pose."""
    if cal["baseline"] is not None:
        return True

    cal["t"] += dt

    features = {
        "yaw": yaw,
        "gaze_x": gaze_x,
        "roll": roll,
        "down": down,
        "body_lean": body_lean,
    }

    for key, value in features.items():
        if value is not None and np.isfinite(value):
            cal["samples"].setdefault(key, []).append(float(value))

    if cal["t"] >= CALIBRATION_SECONDS:
        cal["baseline"] = {
            key: (
                safe_median(values)
                if len(values) >= CALIBRATION_MIN_SAMPLES
                else None
            )
            for key, values in cal["samples"].items()
        }

        print(
            f"Calibration complete for track {cal.get('track_id', '?')}: "
            f"baseline={cal['baseline']}"
        )
        return True

    return False



def decide_reason(pose_m, gaze_delta, yaw_delta, roll_delta, down_delta,
                  box, other_boxes, shoulders_visible, body_lean_delta):
    """Compare current eye/head features with the student's calibrated normal pose."""
    # Eye movement is the main side-looking cue; require a nearby tracked student.
    if gaze_delta is not None and abs(gaze_delta) >= GAZE_THRESHOLD:
        direction = 1 if gaze_delta > 0 else -1
        conf = min(0.95, 0.58 + (abs(gaze_delta) - GAZE_THRESHOLD) * 1.8)
        if neighbor_in_direction(box, other_boxes, direction):
            return "Peeking at Neighbor", conf
        if not ALERT_ONLY_WHEN_NEIGHBOR:
            return "Looking Away", conf

    # Head turn is measured relative to this student's calibrated normal pose.
    if yaw_delta is not None and abs(yaw_delta) >= YAW_THRESHOLD:
        direction = 1 if yaw_delta > 0 else -1
        conf = min(0.90, 0.55 + abs(yaw_delta) * 0.5)
        if neighbor_in_direction(box, other_boxes, direction):
            return "Peeking at Neighbor", conf
        if not ALERT_ONLY_WHEN_NEIGHBOR:
            return "Looking Away", conf

    # A head lean alone is not enough; it must agree with gaze or yaw direction.
    if roll_delta is not None and abs(roll_delta) >= ROLL_THRESHOLD:
        roll_direction = 1 if roll_delta > 0 else -1
        same_direction_support = (
            (gaze_delta is not None and abs(gaze_delta) >= GAZE_THRESHOLD * 0.65
             and (1 if gaze_delta > 0 else -1) == roll_direction)
            or (yaw_delta is not None and abs(yaw_delta) >= YAW_THRESHOLD * 0.65
                and (1 if yaw_delta > 0 else -1) == roll_direction)
        )
        if same_direction_support and neighbor_in_direction(box, other_boxes, roll_direction):
            return "Peeking at Neighbor", 0.65

    # Downward movement is relative to the calibrated normal face pose, not an
    # absolute threshold that changes with the camera angle.
    if down_delta is not None and down_delta >= DOWN_DELTA_THRESHOLD:
        return "Looking Down", min(0.90, 0.55 + abs(down_delta) * 0.5)

    if HEAD_DOWN_WHEN_FACE_HIDDEN and not pose_m["face_visible"] and shoulders_visible:
        return "Head Down", 0.60
        # Body lean alone is not proof; require a nearby student
    # in the same direction and a sustained change.
    if (
        body_lean_delta is not None
        and abs(body_lean_delta) >= BODY_LEAN_THRESHOLD
    ):
        direction = 1 if body_lean_delta > 0 else -1

        if neighbor_in_direction(box, other_boxes, direction):
            return "Peeking at Neighbor", 0.65
    return None, 0.0


# ================= Application =================
class ExamMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Exam Proctoring System")
        self.root.geometry("1200x780")
        self.root.config(bg="#f5f6fa")
        self._set_icon()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.video_path = None
        self.is_running = False
        self.thread = None
        self.row_paths = {}
        self.face_mesh = None

        print("Loading AI models...")
        try:
            self.pose_model = YOLO(POSE_MODEL)
            self.object_model = YOLO(OBJECT_MODEL)
        except Exception as exc:
            messagebox.showerror("Model loading error", f"Could not load YOLO models:\n{exc}")
            raise

        if MEDIAPIPE_AVAILABLE:
            try:
                self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                )
                gaze_status = "Eye landmarks: ON"
            except Exception as exc:
                print("Could not initialize MediaPipe Face Mesh:", exc)
                self.face_mesh = None
                gaze_status = "Eye landmarks: OFF"
        else:
            gaze_status = "Eye landmarks: OFF (install mediapipe)"

        print("Models loaded. Behavior model best.pt is not used.")
        self.firebase_on = init_firebase()

        # ---- Control panel ----
        panel = tk.Frame(root, bg="#2f3640", width=300)
        panel.pack(side=tk.LEFT, fill=tk.Y)
        panel.pack_propagate(False)

        tk.Label(panel, text="Control Panel", font=("Arial", 16, "bold"),
                 bg="#2f3640", fg="white").pack(pady=20)

        self.btn_upload = self._button(panel, "📁 Upload Exam Video", "#3498db", self.upload_video)
        self.btn_start = self._button(panel, "▶ Start Monitoring", "#2ecc71", self.start_monitoring)
        self.btn_stop = self._button(panel, "⏹ Stop Monitoring", "#e74c3c", self.stop_monitoring)
        self.btn_folder = self._button(
            panel, "📂 Open Screenshots Folder", "#9b59b6",
            lambda: open_path(os.path.abspath(SCREENSHOT_DIR)), pady=(30, 10)
        )

        self.status_label = tk.Label(panel, text="No video selected", bg="#2f3640",
                                     fg="#dcdde1", wraplength=260, justify="center")
        self.status_label.pack(pady=20)

        firebase_status = "Firebase: ON" if self.firebase_on else "Firebase: OFF (local DB only)"
        tk.Label(panel, text=f"Pose + objects: ON\n{gaze_status}\n{firebase_status}",
                 bg="#2f3640", fg="#7f8fa6", font=("Arial", 9), justify="center").pack(
                     side=tk.BOTTOM, pady=15
                 )

        # ---- Video area ----
        self.video_frame = tk.Frame(root, bg="black")
        self.video_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.pack(expand=True, fill=tk.BOTH)

        # ---- Alert history ----
        table_frame = tk.Frame(root)
        table_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        tk.Label(table_frame, text="Alert History (double-click a row to open its screenshot)",
                 font=("Arial", 10, "bold")).pack(anchor="w")

        columns = ("Student", "Behavior", "Confidence", "Video Time", "Time", "Screenshot")
        self.alert_table = ttk.Treeview(table_frame, columns=columns, show="headings", height=6)
        widths = {"Student": 90, "Behavior": 170, "Confidence": 80,
                  "Video Time": 80, "Time": 170, "Screenshot": 330}
        for col in columns:
            self.alert_table.heading(col, text=col)
            self.alert_table.column(col, width=widths[col], anchor=tk.CENTER)
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.alert_table.yview)
        self.alert_table.configure(yscrollcommand=scrollbar.set)
        self.alert_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.alert_table.bind("<Double-1>", self.open_selected_screenshot)
        self.load_existing_alerts()

    # ---------- UI helpers ----------
    def _set_icon(self):
        try:
            if sys.platform.startswith("win"):
                self.root.iconbitmap(ICON_PATH)
            else:
                self._icon_img = ImageTk.PhotoImage(Image.open(ICON_PATH))
                self.root.iconphoto(True, self._icon_img)
        except Exception as exc:
            print("Icon not loaded:", exc)

    @staticmethod
    def _button(parent, text, color, command, pady=10):
        button = tk.Button(parent, text=text, bg=color, fg="white", font=("Arial", 11, "bold"),
                           command=command, relief=tk.FLAT)
        button.pack(pady=pady, padx=20, fill=tk.X)
        return button

    def set_status(self, text):
        try:
            self.root.after(0, lambda: self.status_label.config(text=text))
        except tk.TclError:
            pass

    # ---------- Actions ----------
    def upload_video(self):
        if self.is_running:
            return
        path = filedialog.askopenfilename(
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if path:
            self.video_path = path
            self.set_status(f"Video: {os.path.basename(path)}")
            messagebox.showinfo("Success", "Video uploaded successfully!")

    def start_monitoring(self):
        if not self.video_path:
            messagebox.showerror("Error", "Please upload an exam video first!")
            return
        if self.is_running or (self.thread is not None and self.thread.is_alive()):
            return
        self.is_running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_upload.config(state=tk.DISABLED)
        self.thread = threading.Thread(target=self.process_video, daemon=True)
        self.thread.start()

    def stop_monitoring(self):
        self.is_running = False
        self.set_status("Stopping monitoring...")

    def on_close(self):
        self.is_running = False
        try:
            if self.face_mesh is not None:
                self.face_mesh.close()
                self.face_mesh = None
        except Exception:
            pass
        self.root.destroy()

    def load_existing_alerts(self):
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT student_id, behavior, confidence, video_time, timestamp, image_path "
            "FROM alerts ORDER BY id DESC"
        ).fetchall()
        conn.close()
        for sid, behavior, conf, vtime, ts, path in rows:
            self.add_alert_row(sid, behavior, conf, vtime or "-", ts, path, at_end=True)

    def open_selected_screenshot(self, event=None):
        selected = self.alert_table.selection()
        if not selected:
            return
        path = self.row_paths.get(selected[0])
        if path and os.path.exists(path):
            open_path(os.path.abspath(path))
        else:
            messagebox.showwarning("Not found", "Screenshot file was not found on disk.")

    # ---------- Main-thread UI updates ----------
    def update_frame(self, img_pil):
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.imgtk = img_tk
        self.video_label.config(image=img_tk)

    def add_alert_row(self, student_id, behavior, conf, vtime, ts, path, at_end=False):
        iid = self.alert_table.insert(
            "", tk.END if at_end else 0,
            values=(f"Student #{student_id}", behavior, conf, vtime, ts, path)
        )
        self.row_paths[iid] = path

    def show_notification(self, behavior, ts):
        try:
            win = tk.Toplevel(self.root)
            win.title("Proctoring Alert")
            win.config(bg="#c0392b")
            win.attributes("-topmost", True)
            win.geometry("340x150+60+60")
            tk.Label(win, text="⚠ Suspicious Behavior Detected", bg="#c0392b", fg="white",
                     font=("Arial", 12, "bold")).pack(pady=(15, 5))
            tk.Label(win, text=f"Behavior: {behavior}\nTime: {ts}\nScreenshot saved if available.",
                     bg="#c0392b", fg="white", font=("Arial", 10), justify="left").pack()
            win.after(4000, win.destroy)
        except tk.TclError:
            pass

    def on_video_finished(self):
        self.btn_start.config(state=tk.NORMAL)
        self.btn_upload.config(state=tk.NORMAL)
        self.set_status("Monitoring finished")

    # ---------- AI engine (worker thread) ----------
    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.is_running = False
            self.set_status("Could not open the video file")
            try:
                self.root.after(0, self.on_video_finished)
            except tk.TclError:
                pass
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or not np.isfinite(fps) or fps < 1:
            fps = 25.0
        dt = PROCESS_EVERY_N / fps
        self.pose_model.predictor = None
        self.set_status("Monitoring...")

        state = {}
        yaw_hist = {}
        gaze_hist = {}
        roll_hist = {}
        calib = {}
        frame_idx = proc_idx = 0
        obj_boxes = np.zeros((0, 4))
        obj_cls = np.zeros(0, dtype=int)
        obj_conf = np.zeros(0)

        try:
            while self.is_running and cap.isOpened():
                if not cap.grab():
                    break
                frame_idx += 1
                if frame_idx % PROCESS_EVERY_N != 0:
                    continue
                ret, frame = cap.retrieve()
                if not ret:
                    break
                proc_idx += 1
                video_time = fmt_video_time(frame_idx / fps)

                height, width = frame.shape[:2]
                scale = min(DISPLAY_W / width, DISPLAY_H / height)
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

                # --- Object detection (phone / book) ---
                if proc_idx % OBJECT_EVERY_N == 1 or OBJECT_EVERY_N == 1:
                    result_obj = self.object_model(
                        frame, classes=[PHONE_CLASS, BOOK_CLASS], conf=OBJ_CONF, verbose=False
                    )[0]
                    if result_obj.boxes is not None and len(result_obj.boxes) > 0:
                        obj_boxes = result_obj.boxes.xyxy.cpu().numpy()
                        obj_cls = result_obj.boxes.cls.cpu().numpy().astype(int)
                        obj_conf = result_obj.boxes.conf.cpu().numpy()
                    else:
                        obj_boxes = np.zeros((0, 4))
                        obj_cls = np.zeros(0, dtype=int)
                        obj_conf = np.zeros(0)

                # --- Student tracking / pose ---
                result_pose = self.pose_model.track(
                    source=frame, persist=True, tracker=TRACKER, verbose=False
                )[0]
                pending = []

                if (result_pose.boxes is not None and result_pose.boxes.id is not None
                        and result_pose.keypoints is not None):
                    track_ids = result_pose.boxes.id.int().cpu().tolist()
                    boxes = result_pose.boxes.xyxy.cpu().numpy()
                    keypoints = result_pose.keypoints.data.cpu().numpy()

                    for i, (tid, box, kpts) in enumerate(zip(track_ids, boxes, keypoints)):
                        x1, y1, x2, y2 = map(int, box)
                        others = [boxes[j] for j in range(len(boxes)) if j != i]

                        # Pose features and smoothed head yaw.
                        pose_m = pose_metrics(kpts)
                        body_m = body_lean_metrics(kpts)
                        body_lean = body_m["body_lean"]
                        yaw_queue = yaw_hist.setdefault(tid, deque(maxlen=SMOOTH_WINDOW))
                        if pose_m["yaw"] is not None:
                            yaw_queue.append(pose_m["yaw"])
                        yaw_s = float(np.median(yaw_queue)) if yaw_queue else None

                        # Face landmarks: approximate iris direction and head roll.
                        fm = face_metrics(self.face_mesh, frame, box, kpts)
                        gaze_queue = gaze_hist.setdefault(tid, deque(maxlen=SMOOTH_WINDOW))
                        roll_queue = roll_hist.setdefault(tid, deque(maxlen=SMOOTH_WINDOW))
                        if fm["gaze_x"] is not None:
                            gaze_queue.append(fm["gaze_x"])
                        if fm["roll"] is not None:
                            roll_queue.append(fm["roll"])
                        gaze_s = float(np.median(gaze_queue)) if gaze_queue else None
                        roll_s = float(np.median(roll_queue)) if roll_queue else None

                        shoulders_visible = bool(kpts[5][2] > 0.5 and kpts[6][2] > 0.5)
                        cal = calib.setdefault(tid, {
                            "track_id": tid,
                            "t": 0.0,
                           "samples": { "yaw": [], "gaze_x": [], "roll": [],"down": [],"body_lean": []},"baseline": None,
                        })

                        if update_calibration( cal, yaw_s, gaze_s, roll_s,pose_m["down"], body_lean, dt
):
                            baseline = cal["baseline"] or {}
                            yaw_base = baseline.get("yaw")
                            gaze_base = baseline.get("gaze_x")
                            roll_base = baseline.get("roll")
                            down_base = baseline.get("down")
                            yaw_delta = None if yaw_s is None or yaw_base is None else yaw_s - yaw_base
                            gaze_delta = None if gaze_s is None or gaze_base is None else gaze_s - gaze_base
                            roll_delta = None if roll_s is None or roll_base is None else roll_s - roll_base
                            down_delta = (None if pose_m["down"] is None or down_base is None
                                          else pose_m["down"] - down_base)
                            body_baseline = baseline.get("body_lean")
                            body_lean_delta = (
                            None
                            if body_lean is None or body_baseline is None
                            else body_lean - body_baseline
                              )

                            reason, confidence = decide_reason(
                            pose_m, gaze_delta, yaw_delta, roll_delta, down_delta,
                            box, others, shoulders_visible, body_lean_delta

                            
                            )
                        else:
                            yaw_delta = gaze_delta = roll_delta = down_delta = None
                            reason, confidence = None, 0.0

                        # --- Phone / book detection inside the student's box ---
                        for obj_box, class_id, obj_confidence in zip(obj_boxes, obj_cls, obj_conf):
                            obj_cx = (obj_box[0] + obj_box[2]) / 2
                            obj_cy = (obj_box[1] + obj_box[3]) / 2
                            if x1 <= obj_cx <= x2 and y1 <= obj_cy <= y2:
                                # Object detections can persist for a few frames to improve speed.
                                object_reason = "Phone Detected" if class_id == PHONE_CLASS else "Book Detected"
                                reason = object_reason
                                confidence = float(obj_confidence)
                                cv2.rectangle(frame, (int(obj_box[0]), int(obj_box[1])),
                                              (int(obj_box[2]), int(obj_box[3])), (0, 0, 255), 2)

                        # --- Persistent behavior timer per tracked student ---
                        st = state.setdefault(tid, {
                            "reason": None, "t": 0.0, "sent": False, "conf": 0.0
                        })
                        if reason:
                            if reason == st["reason"]:
                                st["t"] += dt
                                st["conf"] = max(st["conf"], confidence)
                            else:
                                st["reason"], st["t"], st["sent"], st["conf"] = (
                                    reason, dt, False, confidence
                                )
                        else:
                            st["t"] = max(0.0, st["t"] - dt)
                            if st["t"] == 0.0:
                                st["reason"], st["sent"], st["conf"] = None, False, 0.0

                        required = SECONDS_REQUIRED.get(st["reason"], DEFAULT_SECONDS)
                        confirmed = bool(reason) and st["reason"] is not None and st["t"] >= required

                        if confirmed and st["reason"] in WARNING_ONLY:
                            color = (0, 165, 255)
                            label = f"Student #{tid} - {st['reason']}"
                        elif confirmed:
                            # This means suspicious behavior detected, not proof of cheating.
                            color = (0, 0, 255)
                            label = f"Student #{tid} - {st['reason']}"
                            if not st["sent"]:
                                st["sent"] = True
                                pending.append((tid, st["reason"], st["conf"]))
                        elif reason:
                            color = (0, 255, 255)
                            label = f"Student #{tid} - Checking..."
                        else:
                            color = (0, 255, 0)
                            label = f"Student #{tid}"
                            if cal["baseline"] is None:
                                label += " (calibrating)"

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, max(15, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        if DEBUG:
                            gaze_txt = "-" if gaze_s is None else f"{gaze_s:.2f}"
                            gaze_d_txt = "-" if gaze_delta is None else f"{gaze_delta:+.2f}"
                            yaw_d_txt = "-" if yaw_delta is None else f"{yaw_delta:+.2f}"
                            roll_d_txt = "-" if roll_delta is None else f"{roll_delta:+.1f}"
                            down_d_txt = "-" if down_delta is None else f"{down_delta:+.2f}"
                            base_state = "calibrating" if cal["baseline"] is None else "baseline set"
                            debug_text = (
                                f"gaze={gaze_txt} dg={gaze_d_txt} "
                                f"dyaw={yaw_d_txt} droll={roll_d_txt} "
                                f"ddown={down_d_txt} {base_state} t={st['t']:.1f}s"
                            )
                            cv2.putText(frame, debug_text,
                                        (x1, min(frame.shape[0] - 5, y2 + 15)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 0), 1)

                # --- Save screenshots and database alerts ---
                if pending:
                    now = time.time()
                    timestamp = time.strftime("%Y-%m-%d %I:%M:%S %p")
                    tids = "_".join(str(item[0]) for item in pending)
                    image_path = os.path.join(
                        SCREENSHOT_DIR, f"alert_{int(now * 1000)}_student_{tids}.jpg"
                    )
                    saved = save_image(image_path, frame)
                    for tid, behavior, confidence in pending:
                        confidence_text = f"{confidence * 100:.0f}%"
                        alert_id = save_alert(
                            tid, behavior, confidence_text, timestamp, video_time,
                            image_path if saved else ""
                        )
                        if self.firebase_on and saved:
                            record = {
                                "student_id": str(tid), "behavior": behavior,
                                "confidence": confidence_text, "timestamp": timestamp,
                                "video_time": video_time,
                            }
                            threading.Thread(
                                target=upload_to_firebase,
                                args=(alert_id, record, image_path), daemon=True
                            ).start()
                        try:
                            self.root.after(
                                0, self.add_alert_row, tid, behavior, confidence_text,
                                video_time, timestamp, image_path if saved else ""
                            )
                            self.root.after(0, self.show_notification, behavior, timestamp)
                        except tk.TclError:
                            pass

                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                try:
                    self.root.after(0, self.update_frame, pil_image)
                except tk.TclError:
                    break
        except Exception as exc:
            print("Error while processing video:", exc)
            self.set_status(f"Processing error: {exc}")
        finally:
            cap.release()
            self.is_running = False
            try:
                self.root.after(0, self.on_video_finished)
            except tk.TclError:
                pass


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("exam.proctoring.app")
        except Exception:
            pass

    init_database()
    root = tk.Tk()
    app = ExamMonitorApp(root)
    root.mainloop()
