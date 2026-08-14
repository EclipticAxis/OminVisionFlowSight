from __future__ import annotations

import logging

import numpy as np


_WRIST = 0
_THUMB_TIP = 4
_INDEX_MCP = 5
_INDEX_PIP = 6
_INDEX_TIP = 8
_MIDDLE_MCP = 9
_MIDDLE_PIP = 10
_MIDDLE_TIP = 12
_RING_MCP = 13
_RING_PIP = 14
_RING_TIP = 16
_PINKY_MCP = 17
_PINKY_PIP = 18
_PINKY_TIP = 20


def normalize_gesture_mode(mode: str) -> str:
    normalized = str(mode or "off").lower()
    return normalized if normalized in {"off", "body", "hand", "all"} else "off"


class HandGestureRecognizer:

    def __init__(self, max_num_hands: int = 2):
        self._hands = None
        self._max_num_hands = max_num_hands

    def detect(self, frame: np.ndarray) -> list[dict]:
        hands = self._ensure_hands()
        if hands is None or frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        if w <= 0 or h <= 0:
            return []

        try:
            frame.flags.writeable = False
            results = hands.process(frame)
            frame.flags.writeable = True
        except Exception as err:
            logging.error("MediaPipe hand detection failed: %s", err)
            try:
                frame.flags.writeable = True
            except Exception:
                pass
            return []

        if not results.multi_hand_landmarks:
            return []

        handedness = results.multi_handedness or []
        detections = []
        for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            landmarks = [
                {
                    "x": float(point.x),
                    "y": float(point.y),
                    "z": float(point.z),
                    "conf": 1.0,
                }
                for point in hand_landmarks.landmark
            ]
            if len(landmarks) != 21:
                continue

            hand_label = self._hand_label(handedness, index)
            gesture = self.classify(landmarks, hand_label)
            x_values = [point["x"] for point in landmarks]
            y_values = [point["y"] for point in landmarks]
            pad = 0.035
            detections.append(
                {
                    "x1": _clamp01(min(x_values) - pad),
                    "y1": _clamp01(min(y_values) - pad),
                    "x2": _clamp01(max(x_values) + pad),
                    "y2": _clamp01(max(y_values) + pad),
                    "confidence": float(gesture["confidence"]),
                    "label": "hand_gesture",
                    "gesture_name": gesture["name"],
                    "gesture_code": gesture["code"],
                    "hand": hand_label,
                    "keypoints": [],
                    "hand_landmarks": landmarks,
                }
            )
        return detections

    def classify(self, landmarks: list[dict], hand_label: str = "unknown") -> dict:
        fingers = self._finger_states(landmarks, hand_label)
        extended = {name for name, value in fingers.items() if value}
        palm_size = self._palm_size(landmarks)
        thumb_index_close = self._dist(landmarks[_THUMB_TIP], landmarks[_INDEX_TIP]) < palm_size * 0.42

        if thumb_index_close and fingers["middle"] and fingers["ring"]:
            return self._gesture("ok", "OK", 0.86)
        if extended == {"thumb"} and self._thumb_points_up(landmarks):
            return self._gesture("thumbs_up", "点赞", 0.84)
        if len(extended) >= 5:
            return self._gesture("five", "数字5", 0.82)
        if extended == {"index"}:
            return self._gesture("one", "数字1", 0.80)
        if extended == {"index", "middle"}:
            return self._gesture("two", "数字2", 0.80)
        if extended == {"index", "middle", "ring"}:
            return self._gesture("three", "数字3", 0.78)
        if not any(fingers.values()):
            return self._gesture("fist", "握拳", 0.78)
        if len(extended) >= 4:
            return self._gesture("open_palm", "张掌", 0.76)
        return self._gesture("hand", "手部", 0.55)

    def close(self) -> None:
        if self._hands is not None:
            try:
                self._hands.close()
            except Exception:
                pass
            self._hands = None

    def detect_in_rois(self, frame: np.ndarray, rois: list[tuple[float, float, float, float]]) -> list[dict]:
        if frame is None or frame.size == 0:
            return []
        frame_h, frame_w = frame.shape[:2]
        detections: list[dict] = []
        for roi in rois:
            x1 = max(0, min(frame_w, int(round(roi[0] * frame_w))))
            y1 = max(0, min(frame_h, int(round(roi[1] * frame_h))))
            x2 = max(0, min(frame_w, int(round(roi[2] * frame_w))))
            y2 = max(0, min(frame_h, int(round(roi[3] * frame_h))))
            if x2 - x1 < 16 or y2 - y1 < 16:
                continue
            roi_frame = frame[y1:y2, x1:x2]
            for det in self.detect(roi_frame):
                detections.append(_map_roi_detection(det, x1, y1, x2 - x1, y2 - y1, frame_w, frame_h))
        return detections

    def _ensure_hands(self):
        if self._hands is not None:
            return self._hands
        try:
            import mediapipe as mp

            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=self._max_num_hands,
                model_complexity=1,
                min_detection_confidence=0.35,
                min_tracking_confidence=0.35,
            )
        except Exception as err:
            logging.error("MediaPipe Hands unavailable: %s", err)
            self._hands = None
        return self._hands

    @staticmethod
    def _hand_label(handedness, index: int) -> str:
        try:
            label = handedness[index].classification[0].label.lower()
            return "left" if label == "left" else "right" if label == "right" else "unknown"
        except Exception:
            return "unknown"

    @staticmethod
    def _finger_states(landmarks: list[dict], hand_label: str) -> dict:
        return {
            "thumb": HandGestureRecognizer._thumb_extended(landmarks, hand_label),
            "index": HandGestureRecognizer._finger_extended(landmarks, _INDEX_MCP, _INDEX_PIP, _INDEX_TIP),
            "middle": HandGestureRecognizer._finger_extended(landmarks, _MIDDLE_MCP, _MIDDLE_PIP, _MIDDLE_TIP),
            "ring": HandGestureRecognizer._finger_extended(landmarks, _RING_MCP, _RING_PIP, _RING_TIP),
            "pinky": HandGestureRecognizer._finger_extended(landmarks, _PINKY_MCP, _PINKY_PIP, _PINKY_TIP),
        }

    @staticmethod
    def _finger_extended(landmarks: list[dict], mcp: int, pip: int, tip: int) -> bool:
        return landmarks[tip]["y"] < landmarks[pip]["y"] < landmarks[mcp]["y"]

    @staticmethod
    def _thumb_extended(landmarks: list[dict], hand_label: str) -> bool:
        if hand_label == "left":
            return landmarks[_THUMB_TIP]["x"] > landmarks[_INDEX_MCP]["x"]
        if hand_label == "right":
            return landmarks[_THUMB_TIP]["x"] < landmarks[_INDEX_MCP]["x"]
        return abs(landmarks[_THUMB_TIP]["x"] - landmarks[_WRIST]["x"]) > abs(landmarks[_INDEX_MCP]["x"] - landmarks[_WRIST]["x"]) * 0.55

    @staticmethod
    def _thumb_points_up(landmarks: list[dict]) -> bool:
        return landmarks[_THUMB_TIP]["y"] < landmarks[_INDEX_MCP]["y"] and landmarks[_THUMB_TIP]["y"] < landmarks[_WRIST]["y"]

    @staticmethod
    def _palm_size(landmarks: list[dict]) -> float:
        return max(HandGestureRecognizer._dist(landmarks[_WRIST], landmarks[_MIDDLE_MCP]), 1e-6)

    @staticmethod
    def _dist(a: dict, b: dict) -> float:
        return ((float(a["x"]) - float(b["x"])) ** 2 + (float(a["y"]) - float(b["y"])) ** 2) ** 0.5

    @staticmethod
    def _gesture(code: str, name: str, confidence: float) -> dict:
        return {"code": code, "name": name, "confidence": confidence}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _map_roi_detection(det: dict, roi_x: int, roi_y: int, roi_w: int, roi_h: int, frame_w: int, frame_h: int) -> dict:
    mapped = dict(det)
    mapped["x1"] = _clamp01((roi_x + float(det["x1"]) * roi_w) / frame_w)
    mapped["y1"] = _clamp01((roi_y + float(det["y1"]) * roi_h) / frame_h)
    mapped["x2"] = _clamp01((roi_x + float(det["x2"]) * roi_w) / frame_w)
    mapped["y2"] = _clamp01((roi_y + float(det["y2"]) * roi_h) / frame_h)
    mapped["hand_landmarks"] = [
        {
            "x": _clamp01((roi_x + float(point["x"]) * roi_w) / frame_w),
            "y": _clamp01((roi_y + float(point["y"]) * roi_h) / frame_h),
            "z": float(point.get("z", 0.0)),
            "conf": float(point.get("conf", 1.0)),
        }
        for point in det.get("hand_landmarks", [])
    ]
    return mapped
