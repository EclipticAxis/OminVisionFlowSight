import math

import cv2
import numpy as np


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


_RECTANGLE_PRESETS = {
    "low": {
        "detector": {
            "min_area_ratio": 0.015,
            "max_area_ratio": 0.40,
            "canny_low": 85,
            "canny_high": 210,
            "max_rectangles": 1,
            "score_threshold": 0.64,
            "adaptive_enabled": False,
            "motion_edge_enabled": False,
            "min_rect_fill_ratio": 0.68,
        },
        "tracker": {"iou_threshold": 0.18, "max_misses": 5, "min_hits": 1, "smooth_alpha": 0.72},
    },
    "medium": {
        "detector": {
            "min_area_ratio": 0.01,
            "max_area_ratio": 0.45,
            "canny_low": 70,
            "canny_high": 180,
            "max_rectangles": 2,
            "score_threshold": 0.58,
            "adaptive_enabled": False,
            "motion_edge_enabled": False,
            "min_rect_fill_ratio": 0.62,
        },
        "tracker": {"iou_threshold": 0.16, "max_misses": 6, "min_hits": 1, "smooth_alpha": 0.70},
    },
    "high": {
        "detector": {
            "min_area_ratio": 0.006,
            "max_area_ratio": 0.55,
            "canny_low": 50,
            "canny_high": 150,
            "max_rectangles": 3,
            "score_threshold": 0.52,
            "adaptive_enabled": True,
            "motion_edge_enabled": True,
            "min_rect_fill_ratio": 0.55,
        },
        "tracker": {"iou_threshold": 0.14, "max_misses": 8, "min_hits": 1, "smooth_alpha": 0.68},
    },
}


def normalize_rectangle_sensitivity(level: str) -> str:
    normalized = str(level or "low").lower()
    return normalized if normalized in _RECTANGLE_PRESETS else "low"


class RectangleDetector:

    def __init__(
        self,
        *,
        min_area_ratio: float = 0.01,
        max_area_ratio: float = 0.45,
        epsilon_ratio: float = 0.035,
        canny_low: int = 70,
        canny_high: int = 180,
        max_rectangles: int = 3,
        score_threshold: float = 0.58,
        adaptive_enabled: bool = False,
        motion_edge_enabled: bool = False,
        min_rect_fill_ratio: float = 0.68,
    ):
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.epsilon_ratio = epsilon_ratio
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.max_rectangles = max_rectangles
        self.score_threshold = score_threshold
        self.adaptive_enabled = adaptive_enabled
        self.motion_edge_enabled = motion_edge_enabled
        self.min_rect_fill_ratio = min_rect_fill_ratio

    @classmethod
    def from_preset(cls, level: str) -> "RectangleDetector":
        preset = _RECTANGLE_PRESETS[normalize_rectangle_sensitivity(level)]["detector"]
        return cls(**preset)

    def set_max_rectangles(self, count: int) -> None:
        self.max_rectangles = max(1, min(10, int(count)))

    def detect(self, frame: np.ndarray) -> list[dict]:
        if frame is None or frame.size == 0:
            return []

        frame_h, frame_w = frame.shape[:2]
        if frame_w <= 0 or frame_h <= 0:
            return []

        frame_area = float(frame_w * frame_h)
        min_area = frame_area * self.min_area_ratio
        max_area = frame_area * self.max_area_ratio

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        masks = self._build_edge_masks(blurred)
        contours = []
        for mask in masks:
            found, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours.extend(found)

        detections: list[dict] = []
        seen_boxes: list[tuple[float, float, float, float]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            peri = cv2.arcLength(contour, True)
            if peri <= 0:
                continue

            points = self._contour_to_quad(contour, peri)
            if points is None:
                continue

            ordered = self._order_points(points)
            if not self._has_valid_geometry(ordered, frame_w, frame_h):
                continue

            score = self._geometry_score(ordered, area, blurred)
            if score < self.score_threshold:
                continue

            x1 = float(np.min(ordered[:, 0]) / frame_w)
            y1 = float(np.min(ordered[:, 1]) / frame_h)
            x2 = float(np.max(ordered[:, 0]) / frame_w)
            y2 = float(np.max(ordered[:, 1]) / frame_h)
            box = (_clamp01(x1), _clamp01(y1), _clamp01(x2), _clamp01(y2))
            if any(_box_iou(box, existing) > 0.28 for existing in seen_boxes):
                continue
            seen_boxes.append(box)

            detections.append(
                {
                    "x1": box[0],
                    "y1": box[1],
                    "x2": box[2],
                    "y2": box[3],
                    "confidence": float(score),
                    "label": "rectangle",
                    "keypoints": [],
                    "polygon": [
                        {"x": _clamp01(float(x) / frame_w), "y": _clamp01(float(y) / frame_h)}
                        for x, y in ordered
                    ],
                }
            )

        detections.sort(key=lambda det: det["confidence"], reverse=True)
        return detections[: max(self.max_rectangles, 6)]

    def _build_edge_masks(self, gray: np.ndarray) -> list[np.ndarray]:
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        masks = [closed_edges]

        if self.motion_edge_enabled:
            motion_edges = cv2.Canny(gray, 30, 100)
            motion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            motion_edges = cv2.dilate(motion_edges, motion_kernel, iterations=1)
            motion_edges = cv2.morphologyEx(motion_edges, cv2.MORPH_CLOSE, kernel, iterations=2)
            masks.append(motion_edges)

        if not self.adaptive_enabled:
            return masks

        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            3,
        )
        adaptive_inv = cv2.bitwise_not(adaptive)
        adaptive_edges = cv2.Canny(adaptive_inv, 30, 120)
        adaptive_edges = cv2.morphologyEx(adaptive_edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        masks.append(adaptive_edges)
        return masks

    def _contour_to_quad(self, contour: np.ndarray, peri: float) -> np.ndarray | None:
        for eps in (self.epsilon_ratio, self.epsilon_ratio * 1.55, self.epsilon_ratio * 2.1):
            approx = cv2.approxPolyDP(contour, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return approx.reshape(4, 2).astype(np.float32)

        rect = cv2.minAreaRect(contour)
        (cx, cy), (w, h), _ = rect
        if w <= 1 or h <= 1:
            return None
        contour_area = cv2.contourArea(contour)
        rect_area = float(w * h)
        if rect_area <= 0 or contour_area / rect_area < self.min_rect_fill_ratio:
            return None
        return cv2.boxPoints(rect).astype(np.float32)

    @staticmethod
    def _order_points(points: np.ndarray) -> np.ndarray:
        sums = points.sum(axis=1)
        diffs = np.diff(points, axis=1).reshape(-1)
        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = points[np.argmin(sums)]
        ordered[2] = points[np.argmax(sums)]
        ordered[1] = points[np.argmin(diffs)]
        ordered[3] = points[np.argmax(diffs)]
        return ordered

    def _has_valid_geometry(self, points: np.ndarray, frame_w: int, frame_h: int) -> bool:
        side_lengths = []
        for index in range(4):
            p1 = points[index]
            p2 = points[(index + 1) % 4]
            side_lengths.append(float(np.linalg.norm(p2 - p1)))

        if min(side_lengths) < max(18.0, min(frame_w, frame_h) * 0.05):
            return False

        width = max(side_lengths[0], side_lengths[2])
        height = max(side_lengths[1], side_lengths[3])
        aspect = width / max(height, 1.0)
        if aspect < 0.18 or aspect > 5.5:
            return False

        x1 = float(np.min(points[:, 0]))
        y1 = float(np.min(points[:, 1]))
        x2 = float(np.max(points[:, 0]))
        y2 = float(np.max(points[:, 1]))
        bbox_area_ratio = ((x2 - x1) * (y2 - y1)) / max(float(frame_w * frame_h), 1.0)
        touches_border = x1 <= 2.0 or y1 <= 2.0 or x2 >= frame_w - 2.0 or y2 >= frame_h - 2.0
        if touches_border and bbox_area_ratio > 0.35:
            return False
        if bbox_area_ratio < self.min_area_ratio or bbox_area_ratio > self.max_area_ratio:
            return False

        return True

    @staticmethod
    def _geometry_score(points: np.ndarray, contour_area: float, gray: np.ndarray) -> float:
        angles = []
        parallel_errors = []
        edge_samples = []
        for index in range(4):
            prev_pt = points[(index - 1) % 4]
            cur_pt = points[index]
            next_pt = points[(index + 1) % 4]
            v1 = prev_pt - cur_pt
            v2 = next_pt - cur_pt
            denom = max(float(np.linalg.norm(v1) * np.linalg.norm(v2)), 1e-6)
            cos_angle = float(np.dot(v1, v2) / denom)
            cos_angle = max(-1.0, min(1.0, cos_angle))
            angles.append(abs(math.degrees(math.acos(cos_angle)) - 90.0))

            opposite = points[(index + 2) % 4] - points[(index + 1) % 4]
            side = next_pt - cur_pt
            side_norm = max(float(np.linalg.norm(side)), 1e-6)
            opposite_norm = max(float(np.linalg.norm(opposite)), 1e-6)
            cross_value = float(side[0] * opposite[1] - side[1] * opposite[0])
            parallel_errors.append(abs(cross_value / (side_norm * opposite_norm)))

            p1 = cur_pt.astype(int)
            p2 = next_pt.astype(int)
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.line(mask, tuple(p1), tuple(p2), 255, 2)
            values = gray[mask > 0]
            if values.size:
                edge_samples.append(float(np.std(values)) / 64.0)

        angle_score = max(0.0, 1.0 - (sum(angles) / len(angles)) / 35.0)
        rect_area = cv2.contourArea(points.astype(np.float32))
        fill_score = max(0.0, min(1.0, float(contour_area) / max(rect_area, 1.0)))
        parallel_score = max(0.0, 1.0 - (sum(parallel_errors) / len(parallel_errors)))
        edge_score = max(0.0, min(1.0, sum(edge_samples) / max(len(edge_samples), 1)))
        score = 0.46 * angle_score + 0.24 * fill_score + 0.18 * parallel_score + 0.12 * edge_score
        return max(0.0, min(0.99, score))


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 1e-8 else 0.0


class _RectangleTrack:

    def __init__(self, track_id: int, det: dict):
        self.track_id = track_id
        self.det = _clone_detection(det)
        self.kalman = _RectangleKalman(_det_box(det))
        self.hits = 1
        self.misses = 0
        self.age = 1


class _RectangleKalman:

    def __init__(self, box: tuple[float, float, float, float]):
        cx, cy, w, h = _box_to_measurement(box)
        self.x = np.array([[cx], [cy], [w], [h], [0.0], [0.0], [0.0], [0.0]], dtype=np.float64)
        self.p = np.eye(8, dtype=np.float64) * 0.02
        self.f = np.eye(8, dtype=np.float64)
        for i in range(4):
            self.f[i, i + 4] = 1.0
        self.h = np.zeros((4, 8), dtype=np.float64)
        self.h[0, 0] = 1.0
        self.h[1, 1] = 1.0
        self.h[2, 2] = 1.0
        self.h[3, 3] = 1.0
        self.q = np.eye(8, dtype=np.float64) * 0.0015
        self.r = np.eye(4, dtype=np.float64) * 0.012

    def predict(self) -> tuple[float, float, float, float]:
        self.x = self.f @ self.x
        self.p = self.f @ self.p @ self.f.T + self.q
        return self.to_box()

    def correct(self, box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        z = np.array(_box_to_measurement(box), dtype=np.float64).reshape(4, 1)
        innovation = z - self.h @ self.x
        s = self.h @ self.p @ self.h.T + self.r
        k = self.p @ self.h.T @ np.linalg.inv(s)
        self.x = self.x + k @ innovation
        self.p = (np.eye(8, dtype=np.float64) - k @ self.h) @ self.p
        return self.to_box()

    def to_box(self) -> tuple[float, float, float, float]:
        cx = float(self.x[0, 0])
        cy = float(self.x[1, 0])
        w = max(1e-5, float(self.x[2, 0]))
        h = max(1e-5, float(self.x[3, 0]))
        return (_clamp01(cx - w * 0.5), _clamp01(cy - h * 0.5), _clamp01(cx + w * 0.5), _clamp01(cy + h * 0.5))


class RectangleTracker:

    def __init__(self, iou_threshold: float = 0.25, max_misses: int = 4, min_hits: int = 2, smooth_alpha: float = 0.65):
        self._iou_threshold = iou_threshold
        self._max_misses = max_misses
        self._min_hits = min_hits
        self._smooth_alpha = smooth_alpha
        self._next_id = 1
        self._tracks: dict[int, _RectangleTrack] = {}
        self._last_stats = {"matched": 0, "predicted_only": 0, "new": 0, "deleted": 0}

    @classmethod
    def from_preset(cls, level: str) -> "RectangleTracker":
        preset = _RECTANGLE_PRESETS[normalize_rectangle_sensitivity(level)]["tracker"]
        return cls(**preset)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def update(self, detections: list[dict]) -> list[dict]:
        self._last_stats = {"matched": 0, "predicted_only": 0, "new": 0, "deleted": 0}
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        for track in self._tracks.values():
            track.age += 1
            predicted_box = track.kalman.predict()
            self._set_track_box(track, predicted_box)

        for track_id, track in list(self._tracks.items()):
            best_index = None
            best_score = -1.0
            for index, det in enumerate(detections):
                if index in matched_dets:
                    continue
                score, accepted = self._match_score(track.det, det)
                if not accepted:
                    continue
                if score > best_score:
                    best_score = score
                    best_index = index
            if best_index is not None:
                self._update_track(track, detections[best_index])
                matched_tracks.add(track_id)
                matched_dets.add(best_index)
                self._last_stats["matched"] += 1

        for index, det in enumerate(detections):
            if index in matched_dets:
                continue
            track_id = self._next_id
            self._next_id += 1
            self._tracks[track_id] = _RectangleTrack(track_id, det)
            matched_tracks.add(track_id)
            self._last_stats["new"] += 1

        for track_id, track in list(self._tracks.items()):
            if track_id not in matched_tracks:
                track.misses += 1
                self._last_stats["predicted_only"] += 1
            if track.misses > self._max_misses:
                del self._tracks[track_id]
                self._last_stats["deleted"] += 1

        results = []
        for track in self._tracks.values():
            if track.hits < self._min_hits:
                continue
            det = _clone_detection(track.det)
            det["track_id"] = track.track_id
            det["track_hits"] = track.hits
            det["track_misses"] = track.misses
            det["predicted_only"] = bool(track.misses > 0)
            det["confidence"] = float(max(0.1, det.get("confidence", 0.0) * (1.0 - track.misses / (self._max_misses + 1))))
            results.append(det)
        return _nms(results, 0.45)

    def _update_track(self, track: _RectangleTrack, det: dict) -> None:
        alpha = self._smooth_alpha
        old = track.det
        merged = _clone_detection(det)
        corrected_box = track.kalman.correct(_det_box(det))
        for key in ("x1", "y1", "x2", "y2"):
            index = ("x1", "y1", "x2", "y2").index(key)
            merged[key] = float(alpha * corrected_box[index] + (1.0 - alpha) * old[key])

        old_poly = old.get("polygon") or []
        new_poly = det.get("polygon") or []
        if len(old_poly) == 4 and len(new_poly) == 4:
            merged["polygon"] = [
                {
                    "x": float(alpha * new_point["x"] + (1.0 - alpha) * old_point["x"]),
                    "y": float(alpha * new_point["y"] + (1.0 - alpha) * old_point["y"]),
                }
                for old_point, new_point in zip(old_poly, new_poly)
            ]

        merged["confidence"] = float(0.75 * det.get("confidence", 0.0) + 0.25 * old.get("confidence", 0.0))
        track.det = merged
        track.hits += 1
        track.misses = 0

    def _set_track_box(self, track: _RectangleTrack, box: tuple[float, float, float, float]) -> None:
        det = _clone_detection(track.det)
        old_box = _det_box(track.det)
        old_w = max(1e-6, old_box[2] - old_box[0])
        old_h = max(1e-6, old_box[3] - old_box[1])
        new_w = max(1e-6, box[2] - box[0])
        new_h = max(1e-6, box[3] - box[1])
        dx = box[0] - old_box[0]
        dy = box[1] - old_box[1]
        for index, key in enumerate(("x1", "y1", "x2", "y2")):
            det[key] = float(box[index])
        polygon = det.get("polygon") or []
        if len(polygon) == 4:
            scaled = []
            for point in polygon:
                rel_x = (float(point["x"]) - old_box[0]) / old_w
                rel_y = (float(point["y"]) - old_box[1]) / old_h
                scaled.append({"x": _clamp01(box[0] + rel_x * new_w), "y": _clamp01(box[1] + rel_y * new_h)})
            det["polygon"] = scaled
        track.det = det

    def _match_score(self, a: dict, b: dict) -> tuple[float, bool]:
        iou = _box_iou(_det_box(a), _det_box(b))
        center = self._center_score(a, b)
        size = self._size_score(a, b)
        accepted = iou >= self._iou_threshold or (center >= 0.72 and size >= 0.58) or (iou >= 0.04 and center >= 0.58 and size >= 0.45)
        score = iou * 0.62 + center * 0.26 + size * 0.12
        return score, accepted

    @staticmethod
    def _center_score(a: dict, b: dict) -> float:
        ax = (float(a["x1"]) + float(a["x2"])) * 0.5
        ay = (float(a["y1"]) + float(a["y2"])) * 0.5
        bx = (float(b["x1"]) + float(b["x2"])) * 0.5
        by = (float(b["y1"]) + float(b["y2"])) * 0.5
        dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
        avg_size = (
            (float(a["x2"]) - float(a["x1"])) + (float(a["y2"]) - float(a["y1"]))
            + (float(b["x2"]) - float(b["x1"])) + (float(b["y2"]) - float(b["y1"]))
        ) * 0.25
        return max(0.0, 1.0 - dist / max(avg_size * 1.8, 0.08))

    @staticmethod
    def _size_score(a: dict, b: dict) -> float:
        aw = max(1e-6, float(a["x2"]) - float(a["x1"]))
        ah = max(1e-6, float(a["y2"]) - float(a["y1"]))
        bw = max(1e-6, float(b["x2"]) - float(b["x1"]))
        bh = max(1e-6, float(b["y2"]) - float(b["y1"]))
        width_score = min(aw, bw) / max(aw, bw)
        height_score = min(ah, bh) / max(ah, bh)
        return float((width_score + height_score) * 0.5)


def _det_box(det: dict) -> tuple[float, float, float, float]:
    return (float(det["x1"]), float(det["y1"]), float(det["x2"]), float(det["y2"]))


def _box_to_measurement(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    w = max(1e-5, float(box[2]) - float(box[0]))
    h = max(1e-5, float(box[3]) - float(box[1]))
    cx = float(box[0]) + w * 0.5
    cy = float(box[1]) + h * 0.5
    return (cx, cy, w, h)


def _clone_detection(det: dict) -> dict:
    cloned = dict(det)
    cloned["polygon"] = [dict(point) for point in det.get("polygon", [])]
    cloned["keypoints"] = [dict(point) for point in det.get("keypoints", [])]
    return cloned


def _nms(detections: list[dict], iou_threshold: float) -> list[dict]:
    ordered = sorted(
        detections,
        key=lambda det: (
            int(det.get("track_hits", 0)),
            -int(det.get("track_misses", 0)),
            float(det.get("confidence", 0.0)),
        ),
        reverse=True,
    )
    kept: list[dict] = []
    for det in ordered:
        if any(_box_iou(_det_box(det), _det_box(existing)) > iou_threshold for existing in kept):
            continue
        kept.append(det)
    return kept
