# 真机部署运行环境（`.venv_ros`）

机器人侧的 ROS 2 节点（`vln_client` / `vln_mpc` / `vln_web` / `robot_adapters`）
需要一套和推理侧（`.venv310` / `.venv311`）**不同**的依赖组合。这份文档说明为什么、
怎么建、以及怎么用。

一键建好（幂等，可重复运行）：

```bash
cd /home/jetson/Public/LightNav-0
bash robot_deploy/scripts/setup_ros_venv.sh
```

---

## 三个 venv 的分工

| venv | Python | numpy | 用途 | 能跑 ROS 节点？ |
|---|---|---|---|---|
| `.venv310` | 3.10.12 | **2.2.6** | 推理服务（vLLM 0.19.0 / transformers 4.57） | ✅ 但 `cv_bridge` 会崩 |
| `.venv311` | 3.11.18 | 2.2.6 | 推理服务（自编译 torch / transformers 5.8） | ❌ **不可能** |
| **`.venv_ros`** | 3.10.12 | **1.21.5**（系统） | **机器人侧 ROS 2 节点** | ✅ **含 `cv_bridge`** |

### 为什么 `.venv311` 一定不能跑 ROS

```
$ .venv311/bin/python -c "import rclpy"
The C extension
'/opt/ros/humble/lib/python3.10/site-packages/_rclpy_pybind11.cpython-311-aarch64-linux-gnu.so'
isn't present on the system.
```

ROS 2 Humble 的 `rclpy` 是按 **cp310** 编译的 C 扩展，Python 3.11 无法加载。
除非自己编译整个 ROS 2，否则无解。所以：

- **推理服务** → `.venv310` 或 `.venv311`（都不要 `rclpy`）
- **ROS 节点** → `.venv_ros`

### 为什么不能直接用 `.venv310`

`.venv310` 是 numpy **2.2.6**，而 `cv_bridge` 是按 numpy 1.x 的 C API 编译的：

```
$ .venv310/bin/python -c "import cv_bridge; ..."
AttributeError: _ARRAY_API not found
KeyError: 16
```

这才是工作区 `AGENTS.md` 里「numpy 分裂」警告的**真正对象**。
注意区分：**`tf2_ros` 在 numpy 2.x 下是正常的**（实测可 pub/sub + 广播 TF），
只有 `cv_bridge` 这个直接吃 numpy C API 的包会崩。

---

## `.venv_ros` 的做法：`--system-site-packages`

```bash
uv venv --python /usr/bin/python3 --system-site-packages .venv_ros
```

这样它同时看得见：

| 来源 | 拿到什么 |
|---|---|
| `/opt/ros/humble/lib/python3.10/site-packages` | `rclpy`、`tf2_ros`、各种 `*_msgs` |
| `/opt/ros/humble/local/lib/python3.10/dist-packages` | **`cv_bridge`** |
| `/usr/lib/python3/dist-packages` | **numpy 1.21.5**、`cv2`、`yaml` |

只装系统缺的两个：

```bash
uv pip install --no-deps --python .venv_ros/bin/python "casadi>=3.7,<4"
uv pip install --python .venv_ros/bin/python aiohttp
```

### ⚠️ `casadi` 必须用 `--no-deps`

`casadi` 的元数据允许任意 numpy，普通安装会把 numpy **升级到 2.2.6**，`cv_bridge`
立刻回到崩的状态：

```
$ uv pip install --python .venv_ros/bin/python casadi
 + casadi==3.8.2
 + numpy==2.2.6      ← 灾难
```

`--no-deps` 后干净：

```
 + casadi==3.8.2     ← 只有这一个
```

`aiohttp` 及其依赖（`aiohappyeyeballs`/`aiosignal`/`attrs`/`frozenlist`/`idna`/
`multidict`/`propcache`/`yarl` …）**完全不碰 numpy**，可以正常装。

---

## 实测验证过的能力

| 项 | 结果 |
|---|---|
| `numpy` | 1.21.5（来自系统，非 venv） |
| `rclpy` / `tf2_ros` / `*_msgs` | OK |
| **`cv_bridge` 往返**（`bgr8` → msg → `rgb8`） | ✅ 不再崩溃 |
| `casadi` + **ipopt** 求解 | ✅ `Opti` 正常求解 |
| **真实 `vln_mpc.mpc.MPCController.solve()`** | ✅ 39.5 ms，输出 `v=0.100 w=0.000` |
| `aiohttp` / `cv2` / `yaml` | OK |
| venv 体积 | 仅 ~170 MB（其余全复用系统） |

`vln_mpc.mpc` 的真实调用：

```python
from vln_mpc.mpc import MPCController
c = MPCController(horizon=10, dt_s=0.1, w_max=1.0, a_max_v=1.0, a_max_w=2.0,
                  q_weights=(1.0, 1.0, 1.0), r_weights=(0.1, 0.1))
v, traj = c.solve(pose, reference, previous=(0.0, 0.0), v_max=0.3)
# → v=(0.100, 0.000), traj.shape=(11, 3)
```

---

## 用法

```bash
source /opt/ros/humble/setup.bash          # 必须先，否则找不到 rclpy/cv_bridge
source /home/jetson/Public/LightNav-0/robot_deploy/install/setup.bash

# colcon 生成的 console_scripts 的 shebang 指向系统 python3（numpy 1.21.5），
# 而系统 python3 没有 casadi/aiohttp。所以显式用 .venv_ros 跑模块：
PY=/home/jetson/Public/LightNav-0/.venv_ros/bin/python

$PY -m vln_client.vln_node --ros-args -p image_transport:=compressed
$PY -m vln_mpc.mpc_node    --ros-args -p track_v_max:=0.3
$PY -m vln_web.web_node
```

**不要**依赖 `ros2 run` / `ros2 launch` 直接起这些节点 —— 它们会用 shebang 里的
系统解释器，缺 `casadi`。要么显式 `python -m`，要么把入口脚本的 shebang 改掉
（见下节）。

### 让 `ros2 run` / `ros2 launch` 直接使用 `.venv_ros`

用 `.venv_ros/bin/python` 驱动 `colcon build`（`ament_python` 会按当前解释器写入 shebang），项目内已提供一键脚本，且不触碰 `hachimi_ws`：

```bash
bash /home/jetson/Public/LightNav-0/robot_deploy/scripts/build_ros.sh
```

编译后加载叠加环境即可直接 `ros2 launch`：

```bash
source /home/jetson/Public/LightNav-0/robot_deploy/scripts/m1_env.sh
ros2 launch vln_bringup m1.launch.py
```

---

## 相机：必须走 `compressed`

| `image_transport` | 用的库 | 你的相机（`bgr8`） | 结果 |
|---|---|---|---|
| `raw` | `cv_bridge` + 只接受 `rgb8` | `bgr8` | ❌ 编码不匹配 |
| **`compressed`** | 纯 `cv2.imdecode` | 有 JPEG 话题 | ✅ |

`vln_client` 的 `compressed` 分支完全不 import `cv_bridge`
（`vln_client.py` 里是 `cv2.imdecode(...)`），所以即使 numpy 环境不合适也能跑。
但 `vln_web` 等其它节点可能仍需要 `cv_bridge`，所以 numpy 1.x 这个前提还是要保证。

机器人前视相机话题：

```
/camera/front/image_raw            sensor_msgs/Image  bgr8  1920x1080
/camera/front/image_raw/compressed sensor_msgs/CompressedImage  JPEG
```

---

## 与推理服务的关系

两者是**独立进程**，通过 WebSocket 通信，互不共享 Python 环境：

```
.venv311 (或 .venv310)                    .venv_ros
  lightnav-serve                            vln_client ──┐
  ws://127.0.0.1:8050  ←──── WebSocket ──────  vln_mpc  ──┼─→ /cmd_vel
                                            vln_web    ──┘
```

唯一需要一致的只有 `vln_client` 的 `server_url` 参数。
