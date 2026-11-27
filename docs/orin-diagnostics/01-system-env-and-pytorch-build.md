# Jetson AGX Orin 系统环境体检与自编译 PyTorch 说明

> 适用设备：NVIDIA Jetson AGX Orin Developer Kit（64GB，SoC `tegra234`，`sm_87`）  
> 检查基准：`jtop 7.2.1` 面板截图与本机实际安装包状态（2026-11-26）

---

## 一、`jtop` 面板逐项解读与系统环境诊断

`jtop` 的 `INFO` 页面分为 **Platform（系统平台）**、**Hardware（硬件与底包）**、**Libraries（加速库）**、**CPU（处理器能力）** 四块。下面逐项解释它们代表什么，并标出哪些是真正的问题、哪些只是虚惊一场。

### 1. Platform & Hardware（系统与硬件基础）

| 条目 | 本机显示值 | 通俗含义 | 状态评估 |
|---|---|---|---|
| `Machine` | `aarch64` | 64 位 ARM 架构（与普通台式机/服务器的 `x86_64` 指令集不同） | 正常，但也是绝大多数 PyPI 二进制包不能直接用的根源 |
| `Distribution` | `Ubuntu 22.04` | 操作系统版本，ROS 2 Humble 与 JetPack 6 的标准底座 | ✅ 合适 |
| `Release` | `5.15.199-tegra` | NVIDIA 为 Jetson 定制的 Linux 内核（含 GPU/电源管理驱动） | ✅ 合适 |
| `Python` | `3.10.12` | Ubuntu 22.04 默认系统 Python（ROS 2 Humble 强依赖它） | ✅ 合适 |
| `Model / Module` | `AGX Orin (64GB ram)` | 官方 64GB 开发套件（`p3701-0005`），CPU 与 GPU 共享 64GB 统一内存 | ✅ 硬件充裕 |
| `SoC` | `tegra234` | Orin 芯片内部代号（T234），对应 apt 软件源中的 `t234` 分支 | 正常 |
| `CUDA Arch BIN` | **`8.7`** | **GPU 计算能力 `sm_87`**（Ampere 架构，Jetson Orin 独占代号） | ⚠️ 最关键的兼容约束，见第二部分 |
| `L4T` | **`36.5.2`** | Linux for Tegra 底层固件与驱动版本（驱动 540.5.0） | ✅ 正常 |
| `Jetpack` | **`6.2.3`** | NVIDIA 官方嵌入式 AI 套件总版本号（升级至 `jetson-stats 7.3.0` 后正常识别） | ✅ 正常 |
| `Interfaces` | `wlP1p1s0` / `eno1` | 无线网卡（`192.168.0.102`）与有线网口（`192.168.168.123`，接机器狗） | 正常 |

### 2. Libraries（底层加速库体检）

| 条目 | 本机显示值 | 通俗含义 | 状态与是否需要补装 |
|---|---|---|---|
| `CUDA` | `12.6.68` | GPU 通用计算平台与编译器（`nvcc`） | ✅ **合适**（L4T 36.5.2 驱动能支持的最高版本，已通过 JetPack apt 补齐 runtime 与 dev 包） |
| `cuDNN` | `9.3.0` | 深度神经网络加速库（卷积、Attention 等底层算子） | ✅ **合适**（JetPack 6.2 官方配套，PyTorch / vLLM 正常调用） |
| **`TensorRT`** | **`MISSING`** | NVIDIA 专用推理优化引擎（将 ONNX 编译为极致加速引擎） | 🟡 **确实未装，但当前不需要**。LightNav-0 走 PyTorch + vLLM 路线，不依赖它；未来跑 YOLO 等 TensorRT 模型时再 `apt` 补装即可 |
| **`VPI`** | **`MISSING`** | Jetson 专用视觉算子库（调用 PVA/VIC 硬件做光流、去畸变等） | 🟡 **确实未装，当前不需要**。LightNav-0 与 ZSM/ROS 栈均未使用 |
| `Vulkan` | `1.3.204` | 底层图形/计算 API | ✅ 正常 |
| **`OpenCV`** | **`4.5.4 with CUDA: NO`** | 计算机视觉基础库，当前为 Ubuntu 官方 apt 自带的 CPU 版 | 🟢 **保持现状最佳**。ROS 2 Humble 的 `cv_bridge` 强绑定系统 OpenCV 4.5.4，擅自重编 CUDA 版反而易破坏 ROS 节点；LightNav-0 仅用它做轻量图像缩放，并非瓶颈 |

### 3. CPU（处理器参数简读）

Orin 搭载 **12 核 Arm Cortex-A78AE**（`Armv8.2-A` 架构，3 个 4 核簇）。面板里剩余条目均为芯片自带的硬件指令集与片上缓存规格，无需人工配置：
- `NEON` / `FP16` / `INT8 Dot Product`：CPU 侧的向量并行与半精度/整型矩阵加速指令；
- `Arm LSE` / `PAC` / `AES, SHA, CRC32`：多核原子锁、指针安全认证与硬件加解密指令；
- `L1 / L2 / L3 Cache`：各级片上高速缓存容量（以字节为单位显示）。

### 4. 系统环境体检总结：到底有没有缺陷？

1. **本机曾经真正存在的系统缺陷（已修复）**：  
   刷机后 `/etc/apt/sources.list.d/nvidia-l4t-apt-source.list` 是 0 字节空文件，且只装了精简版 L4T，没有完整安装 JetPack SDK（缺少 CUDA 与 cuDNN）。此前部署 LightNav-0 时已重建 apt 源并补装了 `cuda-*-12-6` 与 `libcudnn9-cuda-12`。
2. **当前保留的小注意点**：
   - `cuda-cupti-12-6` 的库文件名带日期版本号（`libcupti.so.2024.3.1`），Linux 的 `ldconfig` 不会自动索引它，因此 `scripts/serve_orin.sh` 启动时通过 `LD_LIBRARY_PATH` 显式引入 `/usr/local/cuda/targets/aarch64-linux/lib`。
   - `TensorRT` 与 `VPI` 缺失属于“精简按需安装”的正常结果，**不建议为了消除红色 `MISSING` 而盲目安装整套 `nvidia-jetpack`**，以免占用数 GB 空间或扰动现有依赖。

---

## 二、为什么找不到公开现成环境？我们何时才不得不自编译 PyTorch？

### 1. 先厘清一个容易记混的事实：本机有两套推理环境

查阅 git 历史与本机环境现状，当前 `LightNav-0` 下有两套可工作的推理虚拟环境：

| 环境 | Python | PyTorch 来源 | vLLM 来源 | transformers | 当前定位 |
|---|---|---|---|---|---|
| **`.venv310`** | 3.10.12 | **`jetson-ai-lab` 社区预编译** `2.10.0` | `jetson-ai-lab` 预编译 `0.19.0+cu126` | `4.57.1`（+3 个轻量兼容层） | **当前默认主力环境**（`serve_orin.sh` 默认使用） |
| **`.venv311`** | 3.11.18 | **`jetson-pytorch-builder` 本机自编译** `2.10.0` | 将 `cp310` 轮子重打 `cp311` 标签（内部全为 `abi3`） | **`5.8.0`（与上游训练一致，零兼容层）** | 实验对照环境（实测速度与输出和 `.venv310` 完全一致） |

也就是说：**跑通项目的默认主线 `.venv310` 其实用的是社区预编译包；真正动用 `jetson-pytorch-builder` 自编译 PyTorch，是为了打通与上游完全对齐的 Python 3.11 + `transformers 5.8.0` 路线（`.venv311`）。**

### 2. 确认：当前组合是否真的独特到“全网找不到现成轮子”？

**结论：是的。四重硬约束叠加，几乎封死了所有常规公开下载源：**

1. **架构约束（`aarch64` + `sm_87` 独占）**：  
   PyPI 与 `pytorch.org` 发布的 `aarch64 + cu126` 预编译轮子，是面向数据中心 ARM 服务器（Grace Hopper/Blackwell）编译的，里面只包含 **`sm_80`（A100）和 `sm_90`（H100）** 的 GPU 机器码，且没有可实时翻译的 PTX 中间码。  
   CUDA 同一代架构的不同小版本之间**机器码并不通用**（`sm_80` 不能在 Orin 的 `sm_87` 上执行），导致官方轮子虽然能 `import torch` 且 `cuda.is_available()` 返回 `True`，一跑计算就立刻报错 `no kernel image is available`。同样地，PyPI 的 `nvidia-cublas-cu12` 等运行库也缺 `sm_87`，必须改用 JetPack 原生 CUDA 库。
2. **驱动上限约束（CUDA 12.6 / PTX 8.5）**：  
   L4T 36.5.2 的 GPU 驱动版本是 540.5.0，最高只支持 CUDA 12.6（PTX 8.5）。PyPI 上最新的 `vllm==0.19.1` aarch64 轮子是用 CUDA 12.9（PTX 8.8）编的，一运行就被驱动拒载。
3. **项目版本栈极新（`torch>=2.9` + `vllm 0.19.x` + `transformers==5.8.0`）**：  
   NVIDIA 官方 Jetson 软件源提供的 PyTorch 版本明显滞后，覆盖不到 `torch 2.10`。
4. **社区唯一可用源只提供 Python 3.10**：  
   全网唯一提前编好 `sm_87 + cu126 + torch 2.10 + vllm 0.19.0` 的公开源是 `jetson-ai-lab`，但它**只提供了 `cp310`（Python 3.10）轮子**，没有 Python 3.11 版本，且其 `vllm 0.19.0` 元数据锁死了 `transformers<5`。

因此：
- 走 **Python 3.10**：可以直接借用 `jetson-ai-lab` 预编译轮子，代价是在代码里加 3 个极薄的 `transformers 4.57 ↔ 5.8` 兼容适配层；
- 走 **Python 3.11（原汁原味对齐上游 `transformers 5.8`）**：公开渠道彻底无解，**必须自己从源码编译 PyTorch**。

---

## 三、我们自己编译 PyTorch 到底完成了什么事？

可以把 PyTorch 看作“上层 Python 接口 + 底层大量 C++/CUDA 算子库”。从官网下载的 `.whl` 安装包本质上是别人提前翻译好的成品机器码；当别人的成品里没有针对你这块芯片的翻译版本时，就必须自己拿源码重新编译一遍。

使用 `/home/jetson/Public/jetson-pytorch-builder` 编译 `torch-2.10.0-cp311-linux_aarch64.whl` 时，实际完成了 **4 件关键工作**：

1. **定向生成 `sm_87` 原生 GPU 机器码（解决核心死结）**  
   通过指定 `TORCH_CUDA_ARCH_LIST=8.7`，让 NVIDIA 的 `nvcc` 编译器把 PyTorch 内置的数千个 CUDA 算子（矩阵乘、归一化、激活函数等）全部编译成 Orin GPU 能直接执行的 `sm_87` SASS 指令，彻底消除 `no kernel image is available`。
2. **绑定本机 JetPack 原生 CUDA 12.6 / cuDNN 9.3（实现环境自包含）**  
   直接链接系统的 `/usr/local/cuda-12.6` 与 `libcudnn 9.3`，跳过数据中心版 pip CUDA 库；同时不再像 `jetson-ai-lab` 预编译轮子那样强依赖系统里没有的 `libcudss.so.0`，所以 `.venv311` 甚至不配 `LD_LIBRARY_PATH` 也能直接 `import torch`。
3. **对齐 Python 3.11 二进制接口（`cp311`）**  
   编译出适配 Python 3.11 解释器的 `torch._C` C++ 扩展，补上社区源缺少 `cp311` 轮子的空白。
4. **定制 Jetson + vLLM 所需的编译开关与头文件补丁**  
   - **开启 `USE_DISTRIBUTED=1`**：vLLM 启动时会无条件导入 `torch.distributed.distributed_c10d`。构建工具默认关闭分布式（曾编出无法启动 vLLM 的 `nodist-backup`），后来显式打开重编才通过。
   - **关闭 `USE_NCCL=0`**：Jetson 单卡环境不提供标准二进制 NCCL，关闭后避免链接失败。
   - **补齐 JetPack 缺失头文件**：JetPack 6.2 的 `cuda-cudart-dev-12-6` 虽在 `.so` 里导出了 Profiler 函数，却漏装了 `cuda_profiler_api.h`，因此通过 `cuda-shim/cuda_profiler_api.h` 补齐声明后才顺利过编。

> **实用建议**：日常部署继续使用 `.venv310` 即可（无需维护自编译产物，性能与精度已验证与 `.venv311` 完全一致）；保留 `/home/jetson/Public/jetson-pytorch-builder/wheels/py311/` 中的 `.whl` 作为备份，无需再次耗时数小时重编。
