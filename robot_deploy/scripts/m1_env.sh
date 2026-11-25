#!/usr/bin/env bash
# Source LightNav-0/robot_deploy overlaid cleanly on hachimi_ws/install_m1_driver.
# Compatible with both bash and zsh.
#
# Usage:
#   source /home/jetson/Public/LightNav-0/robot_deploy/scripts/m1_env.sh
#   ros2 launch vln_bringup m1.launch.py
#
# Also exposes `m1_robot_driver` (SDK 0.2.3), so the same script works for
# launching `ros2 launch m1_robot_driver robot_control.launch.py` in zsh/bash.
# Does not modify hachimi_ws, zsm_console (8080), or zsm-1 (8099).

if [ -n "${ZSH_VERSION:-}" ]; then
    _LN_THIS="${(%):-%N}"
    _LN_EXT="zsh"
elif [ -n "${BASH_VERSION:-}" ]; then
    _LN_THIS="${BASH_SOURCE[0]}"
    _LN_EXT="bash"
else
    _LN_THIS="$0"
    _LN_EXT="sh"
fi

_LN_SCRIPT_DIR="$(builtin cd -- "$(dirname -- "${_LN_THIS}")" && pwd)"
_LN_ROBOT_DEPLOY="$(builtin cd -- "${_LN_SCRIPT_DIR}/.." && pwd)"
_LN_REPO_ROOT="$(builtin cd -- "${_LN_ROBOT_DEPLOY}/.." && pwd)"
_LN_HACHIMI_WS="${HACHIMI_WS:-$HOME/Public/hachimi_ws}"

if [ ! -f "${_LN_HACHIMI_WS}/install_m1_driver/local_setup.${_LN_EXT}" ]; then
    echo "[m1_env] missing ${_LN_HACHIMI_WS}/install_m1_driver/local_setup.${_LN_EXT} (build hachimi_ws with ./build.sh or set HACHIMI_WS=...)" >&2
    unset _LN_THIS _LN_EXT _LN_SCRIPT_DIR _LN_ROBOT_DEPLOY _LN_REPO_ROOT _LN_HACHIMI_WS
    return 1 2>/dev/null || exit 1
fi

if [ ! -f "${_LN_ROBOT_DEPLOY}/install/local_setup.${_LN_EXT}" ]; then
    echo "[m1_env] robot_deploy not built yet; run bash ${_LN_ROBOT_DEPLOY}/scripts/build_ros.sh" >&2
    unset _LN_THIS _LN_EXT _LN_SCRIPT_DIR _LN_ROBOT_DEPLOY _LN_REPO_ROOT _LN_HACHIMI_WS
    return 1 2>/dev/null || exit 1
fi

# Start from a clean ROS overlay so legacy nav_ws paths never leak in.
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH ROS_PACKAGE_PATH PYTHONPATH

# 1. ROS 2 Humble
# shellcheck disable=SC1090
source "/opt/ros/humble/setup.${_LN_EXT}"

# 2. Active hachimi_ws SDK 0.2.3 m1_robot_driver & robot_control msgs
export NAV_RUNTIME_ROOT="${_LN_HACHIMI_WS}"
if [ -f "${_LN_HACHIMI_WS}/scripts/env.sh" ]; then
    # shellcheck disable=SC1091
    source "${_LN_HACHIMI_WS}/scripts/env.sh"
fi
if [ -f "${_LN_HACHIMI_WS}/install/local_setup.${_LN_EXT}" ]; then
    # shellcheck disable=SC1090
    source "${_LN_HACHIMI_WS}/install/local_setup.${_LN_EXT}"
fi
# shellcheck disable=SC1090
source "${_LN_HACHIMI_WS}/install_m1_driver/local_setup.${_LN_EXT}"

# 3. LightNav-0 robot_deploy overlay
# shellcheck disable=SC1090
source "${_LN_ROBOT_DEPLOY}/install/local_setup.${_LN_EXT}"

# 4. Expose .venv_ros/bin without touching system python
if [ -d "${_LN_REPO_ROOT}/.venv_ros/bin" ]; then
    case ":${PATH}:" in
        *":${_LN_REPO_ROOT}/.venv_ros/bin:"*) ;;
        *) export PATH="${_LN_REPO_ROOT}/.venv_ros/bin:${PATH}" ;;
    esac
fi

echo "[m1_env] loaded (${_LN_EXT}): hachimi_ws/install_m1_driver + LightNav-0/robot_deploy (.venv_ros)"

unset _LN_THIS _LN_EXT _LN_SCRIPT_DIR _LN_ROBOT_DEPLOY _LN_REPO_ROOT _LN_HACHIMI_WS
