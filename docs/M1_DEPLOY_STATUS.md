# LightNav-0 在铜锤 M1（Jetson AGX Orin）上的复现现状与踩坑说明

> 更新时间：2026-11-25 · 当前分支：`launch-with-venv310`
> 相关文档：[常用运维命令速查](M1_OPS_CHEATSHEET.md) ｜ [ZSM 集成与后续规划](M1_ROADMAP.md) ｜ [Orin 底层环境踩坑](JETSON_ORIN.md)

## 一、阶段性结论：已完成基础复现

已在 **Jetson AGX Orin 64GB + 铜锤 M1（SDK 0.2.3）** 上跑通端到端真机独立闭环：

- **板载推理服务**：`lightnav-serve`（`vllm_local` + `checkpoints/LightNav-0`）在 Orin `MODE_50W` 下稳定运行。
- **独立前视相机**：`m1_rtsp_camera` 直接低延迟拉取机器狗 RTSP 流，不依赖 `zsm_console`。
- **真机控制闭环**：`vln_client` → `vln_mpc` → `m1_adapter` → `hachimi_ws` 新版 `m1_robot_driver`，已打通站立/趴卧、WASD 低速手控、英文指令 `ObjNav` 找物与 `Track` 人物跟随，并可在 8088 通用控制台实时查看画面与遥测。
- **工作区零干扰**：LightNav ROS 栈保留在 `LightNav-0/robot_deploy/`，以叠加层方式复用 `hachimi_ws/install_m1_driver`，未改动 `hachimi_ws`、`zsm_console`（8080）及 `zsm-1`（8099）。

---

## 二、实测性能基线（Orin 板载 `MODE_50W`）

| 指标 | 实测值 | 说明 |
|---|---|---|
| 推理冷启动 | ~90–150 s | 缓存命中 ~70–90 s，加载权重 + CUDA Graph |
| 单步推理延迟 | 初期 ~650 ms → 长时 ~950 ms | 随历史帧累积，端到端频率从 **1.5 Hz 降至 ~1.0 Hz** |
| 显存 / 统一内存 | ~11 GB / 61 GB | 预留充足空间给 ROS 2 与控制栈 |
| 默认运动限速 | 线速 `0.4–0.5 m/s`，角速 `0.4 rad/s` | 针对 ~1 Hz 推理延迟保守标定 |

---

## 三、实际使用中的核心踩坑与限制

### 1. 功耗模式必须锁定 `MODE_50W`（或 `30W`），严禁 `MAXN`
- 本机在 `MAXN`（模式 0）下加载 9.7 GB 权重会**触发整机硬断电重启**（原因尚未结案，见 [JETSON_ORIN.md](JETSON_ORIN.md)）。
- 每次重启设备后，启动推理前务必确认 `nvpmodel -q` 为 `MODE_50W`（模式 3）；若不是，执行 `sudo nvpmodel -m 3`。

### 2. `zsh` 终端下不能直接 source `hachimi_ws/scripts/m1_driver_env.sh`
- `m1_driver_env.sh` 使用了 bash 专有的 `BASH_SOURCE`，在 `zsh` 里会误报「找不到构建产物」。
- **正确做法**：所有终端统一 `source /home/jetson/Public/LightNav-0/robot_deploy/scripts/m1_env.sh`（同时兼容 `bash` / `zsh`，且已包含新版 `m1_robot_driver` 和 LightNav 全部节点）。

### 3. 底盘驱动与前摄严禁双端争抢
- 机器狗 SDK 同一时刻只能有一个客户端持权，**手机 APP 优先级最高**（APP 连着时无法接管，必须关闭 APP）。
- 独立跑 LightNav 时，**不要在 `zsm_console`（8080）或 `zsm-1`（8099）中点击连接底盘或开启前摄拉流**，否则会发生控制权冲突或画面互相冲刷。
- 若需临时在 `zsm_console` 调参数，先停掉 LightNav 的终端 2 和终端 3（**终端 1 的推理服务可保持常驻不用关**），调完断开 `zsm_console` 底盘连接后再重启终端 2/3。

### 4. `Track` 与 `ObjNav` 切换无需重启模型，但指令需用英文
- 当前正式版权重 `LightNav-0` 使用统一提示词，**直接在 8088 网页下拉框切换 `Track` / `ObjNav` 即可生效**，不用重启推理服务。
- 两者区别在于控制侧：`ObjNav` 收到模型 `stop` 会立刻停止并结束任务；`Track` 会忽略 `stop`，人停则等、人走续跟。
- 指令请使用清晰具体的**英文**描述（如 `walk to the chair`、`follow the person in the black shirt`）。

### 5. 推理帧率偏低时的运动特征
- 模型按 4 Hz 训练，Orin 板载实际仅 ~1.0–1.5 Hz。
- 角速度过高（如 1.0 rad/s）会因 1 秒延迟导致**左右过冲震荡甚至甩丢目标**，因此默认将 MPC `w_max` 限制在 `0.4 rad/s`、`r_w = 0.5`；线速度已按实机体感上调至 `0.4 m/s`（ObjNav）/ `0.5 m/s`（Track），可在 8088 页面下方 MPC 面板实时微调。

### 6. 8088 控制台仅为临时通用调试页
- 该页面是开源项目自带的通用机器人前端，仅用于验证 LightNav 最小闭环，并非为 M1 定制；无线/部分硬件状态项无实际意义。长期入口规划见 [M1_ROADMAP.md](M1_ROADMAP.md)。
