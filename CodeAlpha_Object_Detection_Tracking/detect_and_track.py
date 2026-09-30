"""
Real-time Object Detection and Tracking
---------------------------------------
Detector : YOLOv8 (Ultralytics, pre-trained on COCO)
Tracker  : SORT (Kalman filter + Hungarian assignment), implemented below
Input    : webcam or video file (OpenCV)

Install:
    pip install ultralytics opencv-python numpy scipy

Run:
    python detect_and_track.py                       # webcam 0
    python detect_and_track.py --source video.mp4    # video file
    python detect_and_track.py --source video.mp4 --classes 0 2 --save out.mp4
Press 'q' or ESC to quit.
"""

import argparse
import time
from collections import defaultdict, deque

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO


# --------------------------------------------------------------------------- #
# SORT implementation
# --------------------------------------------------------------------------- #
def iou_batch(a, b):
    """IoU between every box in a (N,4) and b (M,4); boxes are [x1,y1,x2,y2]."""
    a = np.expand_dims(a, 1)
    b = np.expand_dims(b, 0)
    xx1 = np.maximum(a[..., 0], b[..., 0])
    yy1 = np.maximum(a[..., 1], b[..., 1])
    xx2 = np.minimum(a[..., 2], b[..., 2])
    yy2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area_a + area_b - inter + 1e-9)


def bbox_to_z(bbox):
    """[x1,y1,x2,y2] -> [cx, cy, area, aspect_ratio]"""
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return np.array([bbox[0] + w / 2, bbox[1] + h / 2, w * h, w / (h + 1e-9)]).reshape(4, 1)


def x_to_bbox(x):
    """State vector -> [x1,y1,x2,y2]"""
    w = np.sqrt(max(x[2, 0] * x[3, 0], 0))
    h = x[2, 0] / (w + 1e-9)
    return np.array([x[0, 0] - w / 2, x[1, 0] - h / 2,
                     x[0, 0] + w / 2, x[1, 0] + h / 2])


class KalmanBoxTracker:
    """Constant-velocity Kalman filter for one tracked box.
    State: [cx, cy, area, ratio, vx, vy, v_area]."""
    count = 0

    def __init__(self, bbox, cls_id, conf):
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0
        self.H = np.eye(4, 7)
        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0
        self.P = np.eye(7)
        self.P[4:, 4:] *= 1000.0
        self.P *= 10.0
        self.Q = np.eye(7)
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01
        self.x = np.zeros((7, 1))
        self.x[:4] = bbox_to_z(bbox)

        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.cls_id, self.conf = cls_id, conf
        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def predict(self):
        if self.x[6, 0] + self.x[2, 0] <= 0:      # keep area positive
            self.x[6, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return x_to_bbox(self.x)

    def update(self, bbox, cls_id, conf):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.cls_id, self.conf = cls_id, conf
        z = bbox_to_z(bbox)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    def get_state(self):
        return x_to_bbox(self.x)


class Sort:
    def __init__(self, max_age=20, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets):
        """dets: (N,6) array [x1,y1,x2,y2,conf,cls].
        Returns (K,7) array [x1,y1,x2,y2,id,cls,conf]."""
        self.frame_count += 1

        # 1) predict new locations of existing tracks
        preds, alive = [], []
        for t in self.trackers:
            p = t.predict()
            if not np.any(np.isnan(p)):
                preds.append(p)
                alive.append(t)
        self.trackers = alive
        preds = np.array(preds).reshape(-1, 4)

        # 2) associate detections to tracks (Hungarian on IoU)
        matched, unmatched_dets = [], list(range(len(dets)))
        if len(preds) and len(dets):
            iou = iou_batch(dets[:, :4], preds)
            rows, cols = linear_sum_assignment(-iou)
            unmatched_dets = [d for d in range(len(dets)) if d not in rows]
            for d, t in zip(rows, cols):
                if iou[d, t] < self.iou_threshold:
                    unmatched_dets.append(d)
                else:
                    matched.append((d, t))

        # 3) update matched tracks, spawn tracks for unmatched detections
        for d, t in matched:
            self.trackers[t].update(dets[d, :4], int(dets[d, 5]), float(dets[d, 4]))
        for d in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[d, :4], int(dets[d, 5]), float(dets[d, 4])))

        # 4) collect output, drop stale tracks
        out = []
        for t in self.trackers:
            if t.time_since_update < 1 and (t.hit_streak >= self.min_hits
                                            or self.frame_count <= self.min_hits):
                out.append(np.concatenate([t.get_state(), [t.id, t.cls_id, t.conf]]))
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        return np.array(out) if out else np.empty((0, 7))


# --------------------------------------------------------------------------- #
# Visualisation helpers
# --------------------------------------------------------------------------- #
def id_color(track_id):
    rng = np.random.RandomState(int(track_id) * 7919)
    return tuple(int(c) for c in rng.randint(60, 255, 3))


def draw_track(frame, box, track_id, label, conf, color):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"ID {track_id} | {label} {conf:.2f}"
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    top = max(y1 - th - bl - 4, 0)
    cv2.rectangle(frame, (x1, top), (x1 + tw + 6, top + th + bl + 4), color, -1)
    cv2.putText(frame, text, (x1 + 3, top + th + 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="webcam index or video path")
    ap.add_argument("--model", default="yolov8n.pt", help="YOLO weights (n/s/m/l/x)")
    ap.add_argument("--conf", type=float, default=0.4, help="detection confidence")
    ap.add_argument("--classes", type=int, nargs="*", default=None,
                    help="COCO class ids to keep, e.g. 0=person 2=car")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--save", default=None, help="optional output video path")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    model = YOLO(args.model)
    names = model.names
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    trails = defaultdict(lambda: deque(maxlen=30))

    writer = None
    if args.save:
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, size)

    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # --- detection ---
        res = model.predict(frame, conf=args.conf, classes=args.classes,
                            imgsz=args.imgsz, verbose=False)[0]
        if len(res.boxes):
            dets = np.hstack([
                res.boxes.xyxy.cpu().numpy(),
                res.boxes.conf.cpu().numpy()[:, None],
                res.boxes.cls.cpu().numpy()[:, None],
            ])
        else:
            dets = np.empty((0, 6))

        # --- tracking ---
        tracks = tracker.update(dets)

        # --- drawing ---
        for x1, y1, x2, y2, tid, cid, conf in tracks:
            color = id_color(tid)
            draw_track(frame, (x1, y1, x2, y2), int(tid), names[int(cid)], conf, color)
            trails[int(tid)].append((int((x1 + x2) / 2), int(y2)))
            pts = trails[int(tid)]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], color, 2)

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(frame, f"FPS: {fps:.1f}  Tracks: {len(tracks)}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Object Detection + SORT Tracking", frame)
        if writer:
            writer.write(frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
