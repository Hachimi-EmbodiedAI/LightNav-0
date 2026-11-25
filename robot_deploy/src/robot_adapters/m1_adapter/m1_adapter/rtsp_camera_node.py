"""Standalone low-latency RTSP -> ROS 2 CompressedImage publisher for Genisom M1.

Runs independently of ``zsm_console`` so the LightNav stack can consume the
M1's RTSP camera stream (``rtsp://192.168.168.168:8554/front``) directly on
``/camera/front/image_raw/compressed`` with ``BEST_EFFORT`` sensor QoS.

Low-latency design:
  * Requests FFmpeg ``nobuffer`` / ``low_delay`` and ``CAP_PROP_BUFFERSIZE=1``.
  * A dedicated reader thread continuously drains ``cap.read()`` and overwrites
    a single-slot ``LatestFrameBuffer`` stamped at read time.
  * The publisher loop only encodes and publishes the newest frame at ``fps``,
    so JPEG encoding never causes FFmpeg frames to queue up behind stale data.
"""

from __future__ import annotations

import os
import signal
import threading
import time
from array import array
from typing import Any, Optional

import cv2
import numpy as np

LOW_LATENCY_FFMPEG_OPTIONS = (
    "rtsp_transport;tcp|stimeout;5000000|fflags;nobuffer|flags;low_delay|max_delay;200000"
)


class LatestFrameBuffer:
    """Thread-safe single-slot buffer that always retains only the newest frame."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seq = 0
        self._frame: Optional[np.ndarray] = None
        self._stamp: Any = None

    def put(self, frame: np.ndarray, stamp: Any) -> int:
        with self._lock:
            self._seq += 1
            self._frame = frame
            self._stamp = stamp
            return self._seq

    def take_if_newer(self, last_seq: int) -> tuple[int, Optional[np.ndarray], Any]:
        with self._lock:
            if self._seq == last_seq or self._frame is None:
                return last_seq, None, None
            return self._seq, self._frame, self._stamp


def encode_frame_jpeg(
    frame: np.ndarray,
    *,
    output_width: int = 640,
    jpeg_quality: int = 80,
) -> Optional[bytes]:
    """Optionally resize a BGR frame and encode it as JPEG bytes."""
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return None
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        return None
    image = frame
    if output_width > 0 and width != output_width:
        target_height = max(1, round(output_width * height / width))
        image = cv2.resize(frame, (output_width, target_height))
    ok, encoded = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    )
    if not ok:
        return None
    return encoded.tobytes()


def main(args: Optional[list[str]] = None) -> None:
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", LOW_LATENCY_FFMPEG_OPTIONS)
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage

    rclpy.init(args=args)
    node = rclpy.create_node("m1_rtsp_camera")
    node.declare_parameter("rtsp_url", "rtsp://192.168.168.168:8554/front")
    node.declare_parameter("image_topic", "/camera/front/image_raw/compressed")
    node.declare_parameter("frame_id", "front_camera_optical")
    node.declare_parameter("fps", 10.0)
    node.declare_parameter("output_width", 640)
    node.declare_parameter("jpeg_quality", 80)
    node.declare_parameter("reconnect_delay_s", 2.0)

    rtsp_url = str(node.get_parameter("rtsp_url").value).strip()
    image_topic = str(node.get_parameter("image_topic").value).strip()
    frame_id = str(node.get_parameter("frame_id").value).strip()
    fps = float(node.get_parameter("fps").value)
    output_width = int(node.get_parameter("output_width").value)
    jpeg_quality = int(node.get_parameter("jpeg_quality").value)
    reconnect_delay_s = float(node.get_parameter("reconnect_delay_s").value)

    if (
        not rtsp_url
        or not image_topic
        or fps <= 0.0
        or output_width < 0
        or not 1 <= jpeg_quality <= 100
        or reconnect_delay_s <= 0.0
    ):
        node.destroy_node()
        rclpy.shutdown()
        raise ValueError("invalid m1_rtsp_camera parameter")

    pub = node.create_publisher(CompressedImage, image_topic, qos_profile_sensor_data)
    min_period_s = 1.0 / fps
    stop_event = threading.Event()
    latest = LatestFrameBuffer()

    def _stop(*_args) -> None:
        stop_event.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    def _reader_loop() -> None:
        while not stop_event.is_set() and rclpy.ok():
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                node.get_logger().warn(
                    f"RTSP open failed ({rtsp_url}); retrying in {reconnect_delay_s:.1f}s"
                )
                cap.release()
                stop_event.wait(reconnect_delay_s)
                continue
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            try:
                while not stop_event.is_set() and rclpy.ok():
                    ok, frame = cap.read()
                    if not ok:
                        node.get_logger().warn(
                            f"RTSP read failed ({rtsp_url}); reconnecting"
                        )
                        break
                    latest.put(frame, node.get_clock().now().to_msg())
            finally:
                cap.release()

    reader = threading.Thread(target=_reader_loop, name="m1_rtsp_reader", daemon=True)
    reader.start()

    node.get_logger().info(
        f"m1_rtsp_camera starting (low-latency): {rtsp_url} -> {image_topic} "
        f"(fps={fps:.1f}, width={output_width or 'native'}, q={jpeg_quality})"
    )

    last_seq = 0
    next_pub_s = 0.0
    try:
        while not stop_event.is_set() and rclpy.ok():
            now = time.monotonic()
            if now >= next_pub_s:
                seq, frame, stamp = latest.take_if_newer(last_seq)
                if frame is not None:
                    last_seq = seq
                    next_pub_s = now + min_period_s
                    jpeg = encode_frame_jpeg(
                        frame,
                        output_width=output_width,
                        jpeg_quality=jpeg_quality,
                    )
                    if jpeg is not None:
                        msg = CompressedImage()
                        msg.header.stamp = stamp
                        msg.header.frame_id = frame_id
                        msg.format = "jpeg"
                        msg.data = array("B", jpeg)
                        pub.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.01)
    finally:
        stop_event.set()
        reader.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
