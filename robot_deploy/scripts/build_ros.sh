#!/usr/bin/env bash
# Build LightNav-0/robot_deploy with .venv_ros/bin/python (Scheme A).
#
# Why this script exists:
#   * Keeps LightNav's ROS 2 packages completely outside hachimi_ws so
#     install_m1_driver, zsm_console (8080) and zsm-1 (8099) are untouched.
#   * Runs colcon via .venv_ros/bin/python so ament_python writes the venv
#     shebang (#!/.../LightNav-0/.venv_ros/bin/python) into every console
#     script, giving vln_client / vln_mpc / vln_web / m1_adapter direct access
#     to casadi + aiohttp while preserving system numpy 1.21.5 + cv_bridge.
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROBOT_DEPLOY_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${ROBOT_DEPLOY_DIR}/.." && pwd)"
VENV_PY="${REPO_ROOT}/.venv_ros/bin/python"
HACHIMI_WS="${HACHIMI_WS:-$HOME/Public/hachimi_ws}"

if [[ ! -x "${VENV_PY}" ]]; then
    echo "[build_ros] missing ${VENV_PY}; run bash robot_deploy/scripts/setup_ros_venv.sh first" >&2
    exit 1
fi

if [[ ! -f "${HACHIMI_WS}/scripts/m1_driver_env.sh" ]]; then
    echo "[build_ros] missing ${HACHIMI_WS}/scripts/m1_driver_env.sh (set HACHIMI_WS=...)" >&2
    exit 1
fi

# Clean inherited overlay variables so old nav_ws / zsm prefixes cannot leak in.
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH ROS_PACKAGE_PATH PYTHONPATH LD_LIBRARY_PATH

# Load ROS 2 Humble + hachimi_ws active m1_robot_driver/robot_control interfaces.
# shellcheck disable=SC1091
source "${HACHIMI_WS}/scripts/m1_driver_env.sh"

cd "${ROBOT_DEPLOY_DIR}"
echo "[build_ros] building ${ROBOT_DEPLOY_DIR} with ${VENV_PY}"
"${VENV_PY}" -m colcon build \
    --symlink-install \
    --cmake-args "-DPython3_EXECUTABLE=${VENV_PY}" \
    "$@"

echo "[build_ros] verifying console_script shebangs..."
bad=0
for exe in \
    install/vln_client/lib/vln_client/vln_client \
    install/vln_mpc/lib/vln_mpc/vln_mpc \
    install/vln_web/lib/vln_web/vln_web \
    install/m1_adapter/lib/m1_adapter/m1_adapter \
    install/m1_adapter/lib/m1_adapter/m1_rtsp_camera; do
    if [[ ! -f "${exe}" ]]; then
        echo "[build_ros] WARN: missing ${exe}" >&2
        bad=1
        continue
    fi
    shebang="$(head -n 1 "${exe}")"
    if [[ "${shebang}" != "#!${VENV_PY}" ]]; then
        echo "[build_ros] FAIL: ${exe} has shebang '${shebang}', expected '#!${VENV_PY}'" >&2
        bad=1
    else
        echo "  OK  ${exe} -> ${shebang}"
    fi
done

if [[ "${bad}" -ne 0 ]]; then
    exit 1
fi
echo "[build_ros] DONE"
