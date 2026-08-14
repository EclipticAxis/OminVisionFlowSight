from __future__ import annotations

from math import atan2, degrees


_NOSE = 0
_LEFT_SHOULDER = 5
_RIGHT_SHOULDER = 6
_LEFT_ELBOW = 7
_RIGHT_ELBOW = 8
_LEFT_WRIST = 9
_RIGHT_WRIST = 10
_LEFT_HIP = 11
_RIGHT_HIP = 12
_LEFT_KNEE = 13
_RIGHT_KNEE = 14
_LEFT_ANKLE = 15
_RIGHT_ANKLE = 16


class GestureRecognizer:

    def __init__(self, conf_threshold: float = 0.30):
        self._conf_threshold = conf_threshold

    def detect(self, det: dict) -> list[dict]:
        kpts = det.get("keypoints") or []
        if det.get("label") != "person" or len(kpts) != 17:
            return []

        torso = self._torso_size(kpts)
        if torso <= 0.0:
            return []

        gestures: list[dict] = []
        left_up = self._is_hand_up(kpts, "left", torso)
        right_up = self._is_hand_up(kpts, "right", torso)

        if left_up and right_up:
            gestures.append(self._gesture("hands_up", "双手举起", "both", 0.86, 70))
        elif left_up:
            gestures.append(self._gesture("left_hand_up", "左手举起", "left", 0.78, 62))
        elif right_up:
            gestures.append(self._gesture("right_hand_up", "右手举起", "right", 0.78, 62))

        if self._is_fallen(kpts, det):
            gestures.append(self._gesture("fallen", "摔倒", "body", 0.76, 100))
        elif self._is_squatting(kpts, torso):
            gestures.append(self._gesture("squat", "蹲下", "body", 0.70, 92))

        if self._is_arms_spread(kpts, torso):
            gestures.append(self._gesture("arms_spread", "双臂展开", "both", 0.74, 58))
        elif self._is_arms_crossed(kpts, torso):
            gestures.append(self._gesture("arms_crossed", "手臂交叉", "both", 0.68, 56))

        near_head = self._near_head_side(kpts, torso)
        if near_head is not None and not (left_up or right_up):
            gestures.append(self._gesture("hand_near_head", "手靠近头部", near_head, 0.66, 48))

        if self._is_sideways(kpts):
            gestures.append(self._gesture("sideways", "侧身", "body", 0.62, 36))

        gestures.sort(key=lambda item: (item["priority"], item["confidence"]), reverse=True)
        return self._dedupe(gestures)[:2]

    def _is_hand_up(self, kpts: list[dict], side: str, torso: float) -> bool:
        wrist_idx, shoulder_idx = (
            (_LEFT_WRIST, _LEFT_SHOULDER) if side == "left" else (_RIGHT_WRIST, _RIGHT_SHOULDER)
        )
        if not self._valid(kpts, wrist_idx) or not self._valid(kpts, shoulder_idx):
            return False
        return kpts[wrist_idx]["y"] < kpts[shoulder_idx]["y"] - torso * 0.16

    def _is_arms_spread(self, kpts: list[dict], torso: float) -> bool:
        required = (_LEFT_WRIST, _RIGHT_WRIST, _LEFT_SHOULDER, _RIGHT_SHOULDER)
        if not all(self._valid(kpts, idx) for idx in required):
            return False
        shoulder_y = (kpts[_LEFT_SHOULDER]["y"] + kpts[_RIGHT_SHOULDER]["y"]) * 0.5
        left_out = kpts[_LEFT_WRIST]["x"] < kpts[_LEFT_SHOULDER]["x"] - torso * 0.18
        right_out = kpts[_RIGHT_WRIST]["x"] > kpts[_RIGHT_SHOULDER]["x"] + torso * 0.18
        level = abs(kpts[_LEFT_WRIST]["y"] - shoulder_y) < torso * 0.32 and abs(kpts[_RIGHT_WRIST]["y"] - shoulder_y) < torso * 0.32
        return left_out and right_out and level

    def _is_arms_crossed(self, kpts: list[dict], torso: float) -> bool:
        required = (_LEFT_WRIST, _RIGHT_WRIST, _LEFT_SHOULDER, _RIGHT_SHOULDER, _LEFT_HIP, _RIGHT_HIP)
        if not all(self._valid(kpts, idx, 0.35) for idx in required):
            return False
        body_center_x = (kpts[_LEFT_SHOULDER]["x"] + kpts[_RIGHT_SHOULDER]["x"] + kpts[_LEFT_HIP]["x"] + kpts[_RIGHT_HIP]["x"]) * 0.25
        upper_y = min(kpts[_LEFT_SHOULDER]["y"], kpts[_RIGHT_SHOULDER]["y"]) - torso * 0.10
        lower_y = max(kpts[_LEFT_HIP]["y"], kpts[_RIGHT_HIP]["y"]) + torso * 0.08
        crossed_x = kpts[_LEFT_WRIST]["x"] > body_center_x and kpts[_RIGHT_WRIST]["x"] < body_center_x
        in_torso_band = upper_y < kpts[_LEFT_WRIST]["y"] < lower_y and upper_y < kpts[_RIGHT_WRIST]["y"] < lower_y
        close_wrists = self._dist(kpts[_LEFT_WRIST], kpts[_RIGHT_WRIST]) < torso * 0.75
        return crossed_x and in_torso_band and close_wrists

    def _near_head_side(self, kpts: list[dict], torso: float) -> str | None:
        if not self._valid(kpts, _NOSE, 0.35):
            return None
        sides = []
        for name, wrist_idx in (("left", _LEFT_WRIST), ("right", _RIGHT_WRIST)):
            if self._valid(kpts, wrist_idx, 0.35) and self._dist(kpts[wrist_idx], kpts[_NOSE]) < torso * 0.52:
                sides.append(name)
        if len(sides) == 2:
            return "both"
        return sides[0] if sides else None

    def _is_squatting(self, kpts: list[dict], torso: float) -> bool:
        required = (_LEFT_HIP, _RIGHT_HIP, _LEFT_KNEE, _RIGHT_KNEE, _LEFT_SHOULDER, _RIGHT_SHOULDER)
        if not all(self._valid(kpts, idx, 0.32) for idx in required):
            return False
        hip_y = (kpts[_LEFT_HIP]["y"] + kpts[_RIGHT_HIP]["y"]) * 0.5
        knee_y = (kpts[_LEFT_KNEE]["y"] + kpts[_RIGHT_KNEE]["y"]) * 0.5
        shoulder_y = (kpts[_LEFT_SHOULDER]["y"] + kpts[_RIGHT_SHOULDER]["y"]) * 0.5
        hips_near_knees = abs(knee_y - hip_y) < torso * 0.34
        compressed_torso = abs(hip_y - shoulder_y) < torso * 1.08
        return hips_near_knees and compressed_torso

    def _is_fallen(self, kpts: list[dict], det: dict) -> bool:
        required = (_LEFT_SHOULDER, _RIGHT_SHOULDER, _LEFT_HIP, _RIGHT_HIP)
        if not all(self._valid(kpts, idx, 0.32) for idx in required):
            return False
        bbox_w = max(0.0, float(det.get("x2", 0.0)) - float(det.get("x1", 0.0)))
        bbox_h = max(0.0, float(det.get("y2", 0.0)) - float(det.get("y1", 0.0)))
        if bbox_h <= 1e-6 or bbox_w / bbox_h < 1.15:
            return False
        shoulder_center = self._center(kpts[_LEFT_SHOULDER], kpts[_RIGHT_SHOULDER])
        hip_center = self._center(kpts[_LEFT_HIP], kpts[_RIGHT_HIP])
        dy = hip_center["y"] - shoulder_center["y"]
        dx = hip_center["x"] - shoulder_center["x"]
        angle = abs(degrees(atan2(dy, dx)))
        return angle < 35.0 or angle > 145.0

    def _is_sideways(self, kpts: list[dict]) -> bool:
        required = (_LEFT_SHOULDER, _RIGHT_SHOULDER, _LEFT_HIP, _RIGHT_HIP)
        if not all(self._valid(kpts, idx, 0.25) for idx in required):
            return False
        shoulder_width = abs(kpts[_LEFT_SHOULDER]["x"] - kpts[_RIGHT_SHOULDER]["x"])
        hip_width = abs(kpts[_LEFT_HIP]["x"] - kpts[_RIGHT_HIP]["x"])
        body_height = abs(((kpts[_LEFT_HIP]["y"] + kpts[_RIGHT_HIP]["y"]) * 0.5) - ((kpts[_LEFT_SHOULDER]["y"] + kpts[_RIGHT_SHOULDER]["y"]) * 0.5))
        if body_height <= 1e-6:
            return False
        narrow_body = shoulder_width < body_height * 0.36 and hip_width < body_height * 0.34
        conf_gap = abs(kpts[_LEFT_SHOULDER].get("conf", 0.0) - kpts[_RIGHT_SHOULDER].get("conf", 0.0)) > 0.35
        return narrow_body or conf_gap

    def _torso_size(self, kpts: list[dict]) -> float:
        if not all(self._valid(kpts, idx, 0.25) for idx in (_LEFT_SHOULDER, _RIGHT_SHOULDER, _LEFT_HIP, _RIGHT_HIP)):
            return 0.0
        shoulder_center = self._center(kpts[_LEFT_SHOULDER], kpts[_RIGHT_SHOULDER])
        hip_center = self._center(kpts[_LEFT_HIP], kpts[_RIGHT_HIP])
        return max(self._dist(shoulder_center, hip_center), 1e-6)

    def _valid(self, kpts: list[dict], idx: int, threshold: float | None = None) -> bool:
        if idx >= len(kpts):
            return False
        return float(kpts[idx].get("conf", 0.0)) >= (self._conf_threshold if threshold is None else threshold)

    @staticmethod
    def _dist(a: dict, b: dict) -> float:
        return ((float(a["x"]) - float(b["x"])) ** 2 + (float(a["y"]) - float(b["y"])) ** 2) ** 0.5

    @staticmethod
    def _center(a: dict, b: dict) -> dict:
        return {"x": (float(a["x"]) + float(b["x"])) * 0.5, "y": (float(a["y"]) + float(b["y"])) * 0.5}

    @staticmethod
    def _gesture(code: str, name: str, side: str, confidence: float, priority: int) -> dict:
        return {"code": code, "name": name, "side": side, "confidence": confidence, "priority": priority}

    @staticmethod
    def _dedupe(gestures: list[dict]) -> list[dict]:
        seen = set()
        results = []
        for gesture in gestures:
            code = gesture["code"]
            if code in seen:
                continue
            seen.add(code)
            results.append(gesture)
        return results
