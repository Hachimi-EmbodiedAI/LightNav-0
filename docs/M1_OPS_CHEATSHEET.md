# LightNav-0（铜锤 M1）常用运维命令速查

> 相关文档：[复现现状与踩坑说明](M1_DEPLOY_STATUS.md) ｜ [ZSM 集成与后续规划](M1_ROADMAP.md)

## 一、启动前确认（任意终端）

```zsh
nvpmodel -q          # 必须为 MODE_50W (3) 或 MODE_30W (2)；切 50W：sudo nvpmodel -m 3
```
确认：手机 APP 已关闭、`zsm_console`（8080）当前未连接底盘且未开前摄。

---

## 二、标准启动流程（按顺序开 3 个终端）

### 终端 1：启动模型推理服务（后台常驻，端口 `8050`）
执行目录：`/home/jetson/Public/LightNav-0`
```zsh
cd /home/jetson/Public/LightNav-0
MODEL_PATH=$PWD/checkpoints/LightNav-0 TASK=vln bash scripts/serve_orin.sh
```
- 看到 `[serve_orin] READY -> ws://127.0.0.1:8050` 即成功（脚本返回后进程在后台运行）。
- 常用辅助命令：
  ```zsh
  tail -f logs/server_8050.log                      # 查看实时推理延迟与日志
  .venv310/bin/python scripts/smoke_orin.py --steps 4   # 端到端冒烟测延迟
  ```

### 终端 2：启动 M1 底盘驱动（`hachimi_ws` SDK 0.2.3）
执行目录：任意（兼容 `bash` / `zsh`）
```zsh
source /home/jetson/Public/LightNav-0/robot_deploy/scripts/m1_env.sh
ros2 launch m1_robot_driver robot_control.launch.py
```

### 终端 3：启动 LightNav ROS 栈（含 RTSP 相机、VLN、MPC、8088 网页）
执行目录：任意
```zsh
source /home/jetson/Public/LightNav-0/robot_deploy/scripts/m1_env.sh
ros2 launch vln_bringup m1.launch.py
```
- 常用临时覆写参数（接在命令末尾）：
  ```zsh
  rtsp_url:=rtsp://<dog-ip>:8554/front    # 更换机器狗 IP
  track_v_max:=0.5 objnav_v_max:=0.4      # 调整线速度上限 (m/s)
  w_max:=0.4                              # 调整转向角速度上限 (rad/s)
  start_rtsp_camera:=false                # 若外部节点已发布前摄压缩图
  server_url:=ws://<gpu-host>:8050        # 连接局域网远端 GPU 推理机
  ```

---

## 三、网页操作与终端 4 常用检查

浏览器打开 **`http://<orin-ip>:8088`**：
1. 确认顶部 `WEB` / `CAMERA` / `ROBOT` 亮绿灯，右侧显示当前姿态（如 `LIE_DOWN`）与电量。
2. 确认周围安全 → 点右侧 **`Stand`** 站立（姿态变 `WALK`）→ 可先点 **`Take manual control`** 用 `WASD` 低速验证底盘，再点 **`STOP`**。
3. 左侧选择 **`ObjNav`**（找物/导航）或 **`Track`**（跟随），输入英文指令，点 **`Start VLN`**；随时可点红色 **`STOP`** 立即急停释放。

### 终端 4（可选）：状态诊断与底盘快捷服务
```zsh
source /home/jetson/Public/LightNav-0/robot_deploy/scripts/m1_env.sh

ros2 topic hz /camera/front/image_raw/compressed                          # 检查前摄帧率 (~10 Hz)
ros2 topic echo /diagnostics --once                                       # 查控制权/急停/remote 状态
ros2 service call /robot/switch_remote std_srvs/srv/Trigger "{}"          # 若处于 idle，切到 remote
ros2 service call /robot/stand std_srvs/srv/Trigger "{}"                  # 站立
ros2 service call /robot_control/obstacle_avoidance std_srvs/srv/SetBool "{data: true}"  # 开/关底盘停障
```

---

## 四、停止服务

| 组件 | 停止方式 |
|---|---|
| 终端 3（LightNav ROS 栈 + 8088） | 在终端 3 按 `Ctrl+C` |
| 终端 2（`m1_robot_driver` 底盘驱动） | 确认狗已停稳/趴卧后，在终端 2 按 `Ctrl+C` |
| 终端 1（后台推理服务 `8050`） | `cd /home/jetson/Public/LightNav-0 && bash scripts/serve_orin.sh --stop` |

---

## 五、端口与维护速查

| 端口 | 服务 | 备注 |
|---|---|---|
| `8050` | `lightnav-serve`（仅本机 `127.0.0.1`） | VLM WebSocket 推理服务 |
| `8088` | `vln_web` | LightNav 临时调试网页 |
| `8080` | `zsm_console`（常驻生产版） | 切勿与终端 2 同时连接底盘 |
| `8099` | `zsm-1`（开发版，按需启动） | 同上 |
| `8000` | `nav-start.service`（旧厂商服务） | 历史遗留，不使用 |

### 改过 `robot_deploy/` 代码后的重编命令
仅当新增包/修改 `setup.py` 时需要（纯改 `.py` 源码因软链接安装，重启终端 3 即生效）：
```zsh
bash /home/jetson/Public/LightNav-0/robot_deploy/scripts/build_ros.sh
```
