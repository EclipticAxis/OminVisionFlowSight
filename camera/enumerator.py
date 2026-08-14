from __future__ import annotations

import logging
import platform
from dataclasses import dataclass
from typing import Optional

import av
import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    from PyQt6.QtMultimedia import QMediaDevices
except Exception:  # pragma: no cover
    QMediaDevices = None  # type: ignore


@dataclass(frozen=True)
class CameraInfo:
    """
    统一的摄像头描述对象。

    index:
        逻辑索引，优先与 GUI 下拉框对应。
    name:
        展示名称。
    device_spec:
        供拉流端直接使用的设备描述符：
        - Windows / DirectShow: "video=xxx" 或纯索引字符串 "0"
        - 其他平台可按需扩展
    backend:
        标注来源，便于调试。
    """

    index: int
    name: str
    device_spec: Optional[str] = None
    backend: str = "unknown"


def _log_and_return_false(message: str, *args) -> bool:
    logging.warning(message, *args)
    return False


def _qt_device_names() -> list[str]:
    """
    使用现代 Qt6 Multimedia 枚举友好名称。

    任何导入失败、插件缺失或返回空列表都不抛异常，交给后续 fallback。
    """
    if QMediaDevices is None:
        logging.info("PyQt6.QtMultimedia is unavailable; skipping QMediaDevices scan")
        return []

    try:
        devices = QMediaDevices.videoInputs()
    except Exception as exc:  # pragma: no cover
        logging.warning("QMediaDevices.videoInputs() failed: %s: %s", type(exc).__name__, exc)
        return []

    names: list[str] = []
    for device in devices:
        try:
            name = device.description().strip()
        except Exception:
            name = ""
        if name:
            names.append(name)

    # 去重，保留原始顺序
    return list(dict.fromkeys(names))


def _probe_with_pyav(device_spec: str, *, max_frames: int = 2) -> bool:
    """
    尝试用 PyAV + DirectShow 拉取最多两帧并做亮度验证。

    兼容两类传参：
    - "video=Device Name"
    - "Device Name"
    """
    container = None
    try:
        container = av.open(
            device_spec,
            format="dshow",
            options={
                "rtbufsize": "256M",
                "video_size": "640x480",
                "framerate": "30",
            },
        )

        # 若设备未产出视频流，直接判定失败
        if not container.streams.video:
            return _log_and_return_false("PyAV validation failed for '%s': no video stream", device_spec)

        stream = container.streams.video[0]
        decoded = 0
        brightness_sum = 0.0

        for frame in container.decode(stream):
            rgb_array = frame.to_ndarray(format="rgb24")
            brightness_sum += float(np.mean(rgb_array))
            decoded += 1
            if decoded >= max_frames:
                break

        if decoded == 0:
            return _log_and_return_false("PyAV validation failed for '%s': decoded 0 frames", device_spec)

        average_brightness = brightness_sum / decoded
        if average_brightness <= 1.0:
            return _log_and_return_false(
                "PyAV validation failed for '%s': average brightness %.3f <= 1.0",
                device_spec,
                average_brightness,
            )

        return True

    except Exception as exc:
        logging.info(
            "PyAV probe failed for '%s': %s: %s",
            device_spec,
            type(exc).__name__,
            exc,
        )
        return False
    finally:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass


def _probe_with_opencv(index: int) -> bool:
    """
    使用 OpenCV + CAP_DSHOW 回退验证。

    这一步主要解决：
    1) QtMultimedia 插件缺失
    2) PyAV dshow 不可用
    3) Qt 枚举返回空，但实际设备可用
    """
    if cv2 is None:
        logging.info("OpenCV is unavailable; cannot fallback-probe camera index %d", index)
        return False

    if platform.system().lower() != "windows":
        backend = getattr(cv2, "CAP_ANY", 0)
    else:
        backend = getattr(cv2, "CAP_DSHOW", getattr(cv2, "CAP_ANY", 0))

    cap = None
    try:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            return False

        # 先丢掉一帧，等待自动曝光/自动白平衡稳定一点
        for _ in range(2):
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            mean_value = float(np.mean(frame))
            if mean_value > 1.0:
                return True

        return False

    except Exception as exc:  # pragma: no cover
        logging.info("OpenCV probe failed for index %d: %s: %s", index, type(exc).__name__, exc)
        return False
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


def _qt_candidates() -> list[CameraInfo]:
    """
    尽量使用 Qt6 的现代设备枚举结果作为首选候选。
    """
    candidates: list[CameraInfo] = []
    for idx, name in enumerate(_qt_device_names()):
        candidates.append(
            CameraInfo(
                index=idx,
                name=name,
                device_spec=f"video={name}",
                backend="qt",
            )
        )
    return candidates


def _opencv_candidates(max_index: int = 16) -> list[CameraInfo]:
    """
    当 Qt 枚举失效时，使用 DirectShow/VideoCapture 扫描设备索引。
    这一步不保证拿到真实友好名称，但能保证找到“可用设备”。
    """
    result: list[CameraInfo] = []
    for idx in range(max_index):
        if _probe_with_opencv(idx):
            result.append(
                CameraInfo(
                    index=idx,
                    name=f"Camera {idx}",
                    device_spec=str(idx),
                    backend="opencv_dshow" if platform.system().lower() == "windows" else "opencv",
                )
            )
    return result


def scan_cameras() -> list[CameraInfo]:
    """
    扫描并验证可用摄像头。

    策略：
    1. 优先使用 QMediaDevices.videoInputs() 获取现代 Qt 设备拓扑；
    2. 若 QtMultimedia 插件缺失、返回空列表或 PyAV 无法直连，则进行 OpenCV 索引回退扫描；
    3. 每个候选都要做真实拉流验证，尽可能排除 OBS / 占位虚拟设备。

    返回值保证不抛出异常，失败时返回空列表。
    """
    validated: list[CameraInfo] = []

    qt_candidates = _qt_candidates()
    if qt_candidates:
        logging.info("Qt enumerated %d video input candidate(s)", len(qt_candidates))
        for candidate in qt_candidates:
            # 先尝试 Qt 给出的友好名称映射到 DirectShow
            if candidate.device_spec and _probe_with_pyav(candidate.device_spec):
                validated.append(candidate)
                continue

            # 某些驱动在 PyAV 中需要直接使用原始名称，而不是 video= 前缀
            if _probe_with_pyav(candidate.name):
                validated.append(
                    CameraInfo(
                        index=candidate.index,
                        name=candidate.name,
                        device_spec=candidate.name,
                        backend="qt_raw",
                    )
                )
                continue

            # 兜底：如果 Qt 设备有名字但 PyAV 拉不通，再试试 OpenCV 的索引打开
            if platform.system().lower() == "windows" and _probe_with_opencv(candidate.index):
                validated.append(
                    CameraInfo(
                        index=candidate.index,
                        name=candidate.name,
                        device_spec=str(candidate.index),
                        backend="opencv_fallback",
                    )
                )
    else:
        logging.info("Qt enumerator returned 0 devices or is unavailable")

    # 若 Qt 枚举失效 / 全部验证失败，再走 OpenCV 索引扫描
    if not validated:
        fallback = _opencv_candidates()
        validated.extend(fallback)

    # 二次去重：避免 Qt 和 OpenCV fallback 重复记录
    deduped: list[CameraInfo] = []
    seen_keys: set[tuple[int, str]] = set()
    for item in validated:
        key = (item.index, item.name)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(item)

    logging.info("scan_cameras() found %d usable camera(s)", len(deduped))
    return deduped
