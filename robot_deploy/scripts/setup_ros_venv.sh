#!/usr/bin/env bash
# Create the ROS 2 runtime venv for the on-robot stack.
#
# Why a separate venv from `.venv310` / `.venv311`:
#
#   * `.venv311` cannot run ROS at all -- rclpy in ROS 2 Humble is a cp310 C
#     extension (`_rclpy_pybind11.cpython-310-*.so`); Python 3.11 cannot load it.
#   * `.venv310` has numpy 2.2.6, and `cv_bridge` segfaults/raises on numpy 2.x
#     (`AttributeError: _ARRAY_API not found`). That is the real "numpy split"
#     warned about in the workspace AGENTS.md -- tf2 is fine, cv_bridge is not.
#
# This venv is created with `--system-site-packages` so it sees:
#
#   rclpy, cv_bridge, tf2_ros, geometry_msgs, ...   from /opt/ros/humble
#   numpy 1.21.5                                    from /usr/lib/python3/dist-packages
#   cv2, yaml                                       from the system
#
# and installs only what the system lacks:
#
#   casadi   -- vln_mpc's Opti/ipopt solver
#   aiohttp  -- vln_client / vln_web websocket client
#
# `casadi` is installed with `--no-deps` on purpose: its metadata allows any
# numpy, so a plain install upgrades numpy to 2.2.6 and breaks cv_bridge again.
set -eo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(cd -- "${script_dir}/../.." && pwd)"
venv="${workspace_dir}/.venv_ros"

if [[ -x "${venv}/bin/python" ]]; then
    echo "[venv_ros] reusing ${venv}"
else
    echo "[venv_ros] creating ${venv} (system-site-packages, python3.10)"
    uv venv --python /usr/bin/python3 --system-site-packages "${venv}"
fi

# casadi: --no-deps keeps numpy at 1.21.5. aiohttp: pulls no numpy at all.
uv pip install --no-deps --python "${venv}/bin/python" "casadi>=3.7,<4"
uv pip install --python "${venv}/bin/python" aiohttp

echo "[venv_ros] verifying"
"${venv}/bin/python" - <<'PY'
import sys

import numpy

print(f"  python  {sys.version.split()[0]}")
print(f"  numpy   {numpy.__version__}  <- {numpy.__file__}")
if numpy.__version__.split(".")[0] != "1":
    sys.exit(
        "numpy must stay on 1.x for cv_bridge; something upgraded it.\n"
        "Re-run: uv pip install --no-deps --python .venv_ros/bin/python casadi"
    )

# ROS is only importable after sourcing /opt/ros/humble/setup.bash.
try:
    import cv_bridge
    import rclpy  # noqa: F401
except ImportError:
    print("  ROS     not sourced (run: source /opt/ros/humble/setup.bash)")
else:
    import cv_bridge as _cv
    import numpy as np

    bridge = _cv.CvBridge()
    msg = bridge.cv2_to_imgmsg(np.zeros((4, 4, 3), np.uint8), encoding="bgr8")
    assert bridge.imgmsg_to_cv2(msg, "rgb8").shape == (4, 4, 3)
    print("  ROS     cv_bridge round-trip OK")

import casadi

print(f"  casadi  {casadi.__version__}")
print("[venv_ros] OK")
PY
