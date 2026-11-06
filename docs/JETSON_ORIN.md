# 在 NVIDIA Jetson AGX Orin 上板载部署 LightNav-0

本文记录在 **AGX Orin (SM 8.7, L4T R36.5 / JetPack 6.2, CUDA 12.6)** 上直接跑
`lightnav-serve` 的可行路径与实测踩坑。与 `JETSON_THOR.md` 是互补关系：
Thor 是 SM 11.0，Orin 是 SM 8.7，两者可用轮子的情况**完全不同**。

## 结论速览

| 组件 | Orin 上唯一可行来源 | 原因 |
|---|---|---|
| **功耗模式** | **`MODE_50W`**（`sudo nvpmodel -m 3`） | **MAXN 在加载 9.7 GB 权重时整机断电**，见第 5 节 |
| Python | **3.10** | 唯一有 sm_87 kernel 的 vLLM 轮子是 `cp310` |
| torch | **jetson-ai-lab**（非 pytorch.org，非 PyPI） | pypi/pytorch.org 的 aarch64 轮子只编 `sm_80`/`sm_90` |
| CUDA 运行库 | **JetPack 原生**（`cuda-cudart-12-6` 等 runtime 包 + `libcudnn9`） | pip 的 `nvidia-*-cu12` 是数据中心构建，**缺 sm_87** |
| vLLM | **jetson-ai-lab `vllm-0.19.0+cu126`** | PyPI 的 aarch64 轮子 PTX 是 8.8（CUDA 12.9），且无 sm_87 |
| transformers | **4.57.1** | vllm 0.19.0 元数据钉 `<5`；由 `rope_compat` / `tokenizer_compat` 抹平 API 差异 |

实测结果（`MODE_50W`）：服务首次启动 **150 s**，单步 **642 ms**（ViT 132 + LLM 523），
端到端 **1.50–1.56 Hz**，峰值功耗 **25.4 W**。真机须降速到 `v_max ≈ 0.3 m/s`。

## 为什么 pip 的 CUDA 库在 Orin 上不可用

这是最容易踩的坑，因为**它们能装、能 import，只在真正 launch kernel 时才失败**。

实测（`nvidia-*` pip 轮子 / PyPI，2026-11）：

```
libcublas.so.12      sm=['90']                 PTX .target sm_90   → 无 sm_87，JIT 也无法降级
libcublasLt.so.12    sm=['52','89','90']       PTX sm_90
libcurand.so.10      sm=['90']                 PTX sm_90
libcusolver.so.11    sm=['80','86','90']       PTX sm_90
libcusparse.so.12    sm=['52','80','90']       PTX sm_80
libcudnn_ops.so.9    sm=['100','120']          PTX sm_120
libcudnn_adv.so.9    sm=['100','120']
```

只有少数库带 sm_87 SASS：`libnvrtc`、`libcufft`、`libnccl`、`libcusparseLt`、
`libcudnn_engines_precompiled`。**关键的 cuBLAS / cuRAND / cuDNN-ops 都没有。**

⚠️ **别用「so 里搜 `sm_87` 字节串」来判断。** 这个判据对 NVIDIA 官方库不成立：

```
libcublas.so.12.6.1.4 (Jetson)   sm_87=False  sm=['90']   ← 但实测 matmul 正常
libcusparse.so.12                sm_87=True   sm=['52','80','87','90']
libcudnn.so.9.3.0                sm_87=False  sm=[]        ← dispatch 库，只含 stub
```

Jetson 版 `libcublas` 的 kernel 打包结构与 pip 版不同，`sm_XX` 不以明文出现。
**唯一可信的判据是真的跑一次 `a@a`**。实测：Jetson `libcublas 12.6.1.4` 在
sm_87 上工作正常；pip 的 `12.6.4.1` 不行。

报错形态很有误导性 —— `torch.zeros()`、`add`、`relu`、`softmax` **全都成功**
（这些是 torch 自己的 kernel，来自含 sm_87 的 torch 轮子），只有第一次
cuBLAS 调用（如 `@`）失败：

```
RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`
```

**不要把它误判成内存不足或 workspace 配置问题**：
`CUBLAS_WORKSPACE_CONFIG` 和 `CUBLAS_LT_DISABLE` 都无效。用下面这段快速判别：

```bash
python -c "
import torch
torch.zeros(4, device='cuda'); torch.cuda.synchronize(); print('torch kernel OK')
torch.randn(64,64,device='cuda') @ torch.randn(64,64,device='cuda'); print('cublas OK')
"
```

第一行过、第二行失败 ⇒ CUDA 库缺 sm_87，与 torch/vLLM 无关。

## 为什么 pytorch.org 的 cu126 aarch64 torch 也不行

```
torch-2.10.0+cu126-cp311-manylinux_2_28_aarch64.whl
  libtorch_cuda.so: -gencode arch=compute_80,code=sm_80;-gencode arch=compute_90,code=sm_90
  嵌入 PTX 段: 0 个
```

名字里的 `cu126` 只表示**编译工具链是 CUDA 12.6**，架构列表仍是数据中心的
sm_80/sm_90。结果是 `torch.cuda.is_available()` 返回 `True`、
`get_device_capability()` 返回 `(8, 7)`（这些只查驱动属性），但
`torch.zeros(4, device='cuda')` 立刻 `no kernel image is available`。

⚠️ **Ampere 家族内 minor 版本之间 cubin 不兼容**：sm_80 的 cubin 不能在
sm_87 上执行。没有可 JIT 的 PTX 时无解。

判别方法（不要只看 `is_available()`）：

```bash
python -c "
import torch, os
p = os.path.join(os.path.dirname(torch.__file__),'lib','libtorch_cuda.so')
d = open(p,'rb').read()
i = d.find(b'-gencode')
print(d[i:i+120].split(b'\x00')[0].decode())
print('PTX blocks:', d.count(b'.visible .entry'))
"
```

## 安装步骤

### 1. 让 JetPack 源可用

本机（R36.5）的 `/etc/apt/sources.list.d/nvidia-l4t-apt-source.list` **是 0 字节空文件**，
需重建。源可达性已验证（`repo.download.nvidia.com/jetson/{common,t234}/dists/r36.5`）。

```bash
# 备份
sudo cp /etc/apt/sources.list.d/nvidia-l4t-apt-source.list /tmp/l4t.list.bak 2>/dev/null || true

# 导入 key
curl -s https://repo.download.nvidia.com/jetson/jetson-ota-public.asc \
  | sudo tee /etc/apt/trusted.gpg.d/jetson-ota-public.asc > /dev/null

# 建源
sudo tee /etc/apt/sources.list.d/nvidia-l4t-apt-source.list > /dev/null <<'EOF'
deb https://repo.download.nvidia.com/jetson/common r36.5 main
deb https://repo.download.nvidia.com/jetson/t234 r36.5 main
EOF

sudo apt update
```

### 2. 装 CUDA 运行库（不改 torch，不装 nvcc）

```bash
sudo apt install -y \
  cuda-cudart-12-6 cuda-nvrtc-12-6 cuda-cupti-12-6 \
  libcublas-12-6 libcufft-12-6 libcurand-12-6 \
  libcusolver-12-6 libcusparse-12-6 libnvjitlink-12-6 \
  libcufile-12-6 libnpp-12-6 \
  libcudnn9-cuda-12
```

这就是 `.venv310` 运行**全部**需要的。实测 `libc10_cuda.so` 的 NEEDED 全部由这些包提供：
`libcudart` `libcublas(-Lt)` `libcudnn` `libcufft` `libcufile` `libcurand`
`libcusolver` `libcusparse` `libcupti` `libnvJitLink`（唯一的例外是 `libcudss`，见 2b）。

> **如果另外还想自编译 torch**（本文的实测路径**不需要**，直接用 jetson 轮子），
> 才需要 `cuda-nvcc-12-6`、`cuda-nvml-dev-12-6` 和各种 `-dev` 包（头文件 / `.so`
> 符号链接）。参见「未走通的两条路」第 2 条。

包名注意事项（实测核对过 R36.5 的 `Packages` 索引）：

- 运行时库是 `cuda-nvrtc-12-6`，**没有** `libnvrtc-12-6` 这个包。
- **只装 runtime，不要 `cuda-toolkit-12-6`**（它拉 nvcc + 数 GB，而且会创建
  `/usr/local/cuda*`，我们不需要编译任何东西）。
- `libcudss` **不在 Jetson 源里**。torch 2.10 的 `libtorch_cuda.so` 依赖它，但只在
  LU/Cholesky 等分解路径上调用，本模型不走。若 import 时报缺，单独补 pip 的
  `nvidia-cudss-cu12==0.7.1.4`（其它 pip CUDA 包不要装）。
- `nccl` 没有独立包，R36 的 NCCL 内嵌在 `nvidia-l4t-*` 里；单卡不需要。

这些包**只往 `/usr/lib/aarch64-linux-gnu/` 放 `.so`**，不碰任何 Python 包、
不碰 numpy、不创建 `/usr/local/cuda`。对同机其它 ROS 工作区（如 `hachimi_ws`）
无影响 —— 前提是那些进程本来就不加载 CUDA 库（用 `ldd` 可以确认）。

装完验证：

```bash
ldconfig -p | grep -E "libcublas.so.12|libcudnn.so.9|libcurand.so.10"
```

### 2b. 两个必知的链接期坑

**（一）`libcupti.so.12` 不会被 ldconfig 索引。**
`cuda-cupti-12-6` 把库装成 `libcupti.so.12 -> libcupti.so.2024.3.1`。ldconfig 只给
`libX.so.N[.N[.N]]` 形式的真实文件建缓存，而 `2024.3.1` 这种版本段它不认，所以
`ldconfig -p | grep cupti` **是空的**，`import torch` 报

```
ImportError: libcupti.so.12: cannot open shared object file
```

`libtorch_cpu.so` 把 cupti 列在 NEEDED 里，即使这里从不开 profiler。两个解法任选：

```bash
# A) 给个 ldconfig 认得的名字（推荐，一次性）
sudo ln -sf /usr/local/cuda-12.6/targets/aarch64-linux/lib/libcupti.so.2024.3.1 \
            /usr/local/cuda-12.6/targets/aarch64-linux/lib/libcupti.so.12.0.0
sudo ldconfig

# B) 运行时把 CUDA 目录放进搜索路径（不需要 sudo）
export LD_LIBRARY_PATH=/usr/local/cuda/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}
```

`scripts/serve_orin.sh` 用 B，因此不依赖系统的 `ldconfig` 状态。

**（二）`libcudss.so.0` 必须保留 pip 版。**
它在 JetPack apt 源里**不存在**，但 `libtorch_cuda.so` 的 NEEDED 里有它。它只在
LU/Cholesky 等分解路径上被调用，本模型不走，所以它的 datacenter-only kernel 永不
执行 —— 只需要能被解析到。因此那一个 pip 包要留着：

```bash
uv pip install "nvidia-cudss-cu12==0.7.1.4"
```

`scripts/serve_orin.sh` 会把 `site-packages/nvidia/cu12/lib` 加进
`LD_LIBRARY_PATH`（**只有这一个目录**，不含其它 pip CUDA 库），其余全部来自
JetPack。

### 3. Python 3.10 venv + jetson torch

```bash
cd /path/to/LightNav-0
uv venv --python 3.10 .venv310
source .venv310/bin/activate

uv pip install \
  --index https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/ \
  --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match \
  "torch==2.10.0" "torchvision==0.25.0"

# jetson 版 torch 的 METADATA 没有任何 nvidia-* 依赖，必须补上 torch 需要、
# 但 JetPack 源里没有的部分（cudss）。其余交给系统的 JetPack 库。
uv pip install "nvidia-cudss-cu12==0.7.1.4" "cuda-bindings==12.9.4" "nvidia-nvjitlink-cu12==12.6.85"
```

**关键：不要再把 pip 的 `nvidia/*/lib` 放进 `LD_LIBRARY_PATH`。**
那正是污染来源。让系统库优先，只在 torch 报缺 cudss 时单独补那一个。

### 4. vLLM（原生 sm_87）

```bash
uv pip install --no-deps \
  "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/6f4/3a27efa50bd2d/vllm-0.19.0+cu126-cp310-cp310-linux_aarch64.whl"

uv pip install "transformers==4.57.1" "tokenizers==0.22.1" \
  "flashinfer-python==0.6.6" "flashinfer-cubin==0.6.6" \
  "nvidia-cutlass-dsl==4.5.2" "quack-kernels==0.5.0" \
  "nvidia-cudnn-frontend>=1.13.0,<1.19.0" "compressed-tensors==0.14.0.1" \
  "numba==0.61.2" "outlines_core==0.2.11" "xgrammar>=0.1.32,<1.0.0" \
  "llguidance>=1.3.0,<1.4.0"

uv pip install --no-deps -e .
uv pip install "pillow>=11.0" "pyyaml>=6.0" "websockets>=12.0" "numpy>=2.0" accelerate
```

`nvidia-cutlass-dsl==4.5.2` 是**硬性 pin**：vLLM 只要求 `>=4.4.0.dev1`，
新版本删掉了 `cutlass.cute.core.ThrMma`，而 quack-kernels 仍在引用它。

`quack-kernels==0.5.0` 也必须钉住，方向和上面相反：它自己的 cutlass 下限随时间上涨，
不 pin 的话解析器会选一个要求 `>=4.7` 的 quack，与 4.5.2 直接冲突而装不上。
0.5.0 是最后一个接受 `>=4.5.2` 的版本。quack 只在 `vllm/vllm_flash_attn/cute/`
（SM90/SM100 的 CUTLASS DSL attention kernel）里被 import，SM8x 走的是预编译的
`_vllm_fa2_C.abi3.so` SASS，所以只要它能 import 成功即可。

### 5. 功耗模式：**用 `MODE_50W`**，MAXN 在本机会硬断电

这是 Orin 上最容易误判成软件故障的问题，**先读这一节再启动**。

实测（2026-11-06，AGX Orin 64GB，`--backend vllm_local`）：

| 功耗模式 | GPU 上限 | 加载 9.7 GB 权重 | 端到端吞吐 | 峰值功耗 |
|---|---|---|---|---|
| `MAXN` (0) | 1300 MHz | ❌ **整机断电重启** | — | — |
| `MODE_50W` (3) | 816 MHz | ✅ 成功 | **1.50–1.56 Hz** | 25.4 W |
| `MODE_30W` (2) | 612 MHz | ✅ 成功 | **1.28 Hz** | 12.5 W |
| `MODE_15W` (1) | 408 MHz | （未实测，预期可用） | — | — |

**推荐 `MODE_50W`**：比 30W 快 22%（LLM decode 640 → 523 ms），且连续 20 步验证稳定。

```bash
sudo nvpmodel -m 3     # MODE_50W
nvpmodel -q            # 确认
```

现象（MAXN）：`lightnav-serve` 进入 vLLM 初始化后**十几秒内整机断电**。不是 kernel
panic（`panic=0` 会让 panic 挂住而不是重启），没有 `pstore` 记录，温度 41–45 °C。

⚠️ **原因尚未确定。** 早期认为「瞬时电流尖峰触发输入保护」，但带功耗监控后
抓到的断电前读数是 **2.1–3.8 W、GPU 306 MHz（未升频）**，这个假设已不成立。
50W（816 MHz）稳定而 MAXN（1300 MHz）必死，说明与**频率档位**相关，而不是简单的
总功率。剩下的测量盲区是 INA3221 的 `samples=512` 使读数为 512 ms 平均，
毫秒级尖峰不可见。完整的调查过程、已排除的假设与下一步见
[**`JETSON_ORIN_POWER.md`**](JETSON_ORIN_POWER.md) §3。

**工程上的做法：用 50W（或 30W）。** 两者都能稳定加载，50W 快 22%。

```bash
sudo nvpmodel -m 3     # MODE_50W（推荐）
# sudo nvpmodel -m 2   # MODE_30W（更保守，功耗减半）
nvpmodel -q            # 确认
```

排查时带功耗监控（读数落到文件，断电后重启还能看）：

```bash
cd /path/to/LightNav-0
bash scripts/monitor_power.sh /tmp/power.log &   # 每 0.5 s 落盘 VDD_IN/I/P
# ...跑会断电的测试...
tail -5 /tmp/power.log
```

`scripts/serve_orin.sh` 在 MAXN 下**只警告不拦截**（3 秒后继续），`MAXN_WARN=0`
可静默。

### 6. 启动

```bash
# 前置：确认功耗模式（见上一节）
nvpmodel -q                      # MODE_50W（推荐）或 MODE_30W；MAXN 会断电

cd /path/to/LightNav-0
MODEL_PATH=$PWD/checkpoints/LightNav-0 TASK=vln bash scripts/serve_orin.sh
```

首次启动约 **150 s**（加载 9.7 GB 权重 + warm-up + CUDA graph capture）。
之后若 torch.compile / inductor 缓存仍有效，实测可降到 **~70 s**；
清过 `~/.cache/vllm/torch_compile_cache` 或换代码后会回到 150 s 量级。
就绪信号是 `.servers_ready/port8050.ready` 出现，**不是端口能 connect** ——
服务在绑定端口前会跑一次合成 warm-up 推理。

另开一个终端验证端到端：

```bash
cd /path/to/LightNav-0
.venv310/bin/python scripts/smoke_orin.py --steps 8
```

期望输出（MODE_30W 实测）：

```
login  rc=0 ok
reset  rc=0 ok
  step 0:   1088 ms  stop=False  rows=10  a0=[1.08e-05, 3.69e-06, 0.314]  raw=<apos_1297><opos_0><act_l0_41>...
  ...
latency min=760 med=776 max=1088 ms  (n=5)
  vit_ms  med=130
  llm_ms  med=640
throughput = 1.29 Hz  (end-to-end, single session)
```

停止服务：

```bash
pkill -f "lightnav.serving.ws[_]server"    # 方括号避免匹配到自己的命令行
```

`scripts/serve_orin.sh` 的 preflight 会先检查 torch CUDA 可用性、cuBLAS matmul、
cutlass `ThrMma`、SM 版本、pip CUDA 库污染，失败即退出并给出修复命令。

## 实测性能（单会话）

| 项 | `MODE_50W` | `MODE_30W` |
|---|---|---|
| 服务启动 | 150 s 首次 / ~70 s（缓存有效时） | 同左 |
| 单步延迟中位 | **642–666 ms** | 776 ms |
| ├ ViT（视觉塔） | 129–132 ms | 130 ms |
| └ LLM decode（5 个 token） | **523–532 ms** | 640 ms |
| 端到端吞吐 | **1.50–1.56 Hz** | 1.28 Hz |
| 峰值功耗 | 25.4 W | 12.5 W |
| 显存占用 | ~11 GB / 61 GB 统一内存 | 同左 |

（50W 数据来自连续 20 步两轮测量；30W 为早期基线。）

**对真机部署的含义**：1.5 Hz 意味着每 ~650 ms 更新一次目标。`vln_mpc` 的
控制环是 10 Hz，即每 ~6 个周期换一次参考轨迹。**仍需降速**：
`track_v_max 0.3`（默认 1.5）、`objnav_v_max 0.25`（默认 0.8）、`a_max_v 1.0`。
若能解决 MAXN 断电（或换电源）到 2+ Hz，可放宽到 0.4–0.5 m/s。

## 与 transformers 版本的关系

项目默认面向 transformers 5.8，但 jetson vllm 0.19.0 钉 `<5`。两者唯一冲突的
私有 API 是 `Qwen3VLModel.get_rope_index`：

```
4.57:  get_rope_index(self, input_ids, image_grid_thw, video_grid_thw, attention_mask)
5.8 :  get_rope_index(self, input_ids, mm_token_type_ids, image_grid_thw, video_grid_thw, ...)
```

`lightnav.inference.rope_compat` 在运行时用 `inspect.signature` 判断并丢掉
4.x 不认识的 `mm_token_type_ids`，所以同一份代码在两种版本下都能跑。
查看当前走哪条分支：

```bash
python -c "from lightnav.inference.rope_compat import describe_rope_index_compat as d; print(d())"
```

## 硬件前置

- **供电（最重要）**：`nvpmodel -q` 应为 **MODE_50W**（推荐）或 `MODE_30W`。MAXN 会在加载权重时
  整机断电 —— 机制与诊断见上文「5. 供电」一节。
- **USB 相机链路**：`lsusb -t` 确认相机枚举在 `5000M`。USB-2 会让相机静默掉到
  10 fps，推理率减半、转弯失稳 —— 这个影响压倒任何模型侧优化。
- **统一内存**：`--gpu_memory_utilization` 按系统其余部分的需要预算
  （模型 + 机器人栈 + 可选共存服务），`scripts/serve_orin.sh` 默认 `0.45`。
- **无鉴权**：`lightnav-serve` 自身没有认证。板载时绑 `127.0.0.1`，
  跨机访问时限定私有网段。

## 未走通的一条路，和一条被误判为「走不通」的路

留档以免重复劳动：

1. ❌ **`pytorch.org` 的 `torch-2.10.0+cu126` aarch64 轮子** —— `cu126` 只表示工具链，
   架构仍是 `sm_80`/`sm_90` 且不含 PTX，Orin 上第一个 CUDA kernel 就
   `no kernel image is available`。已在上面「为什么 pytorch.org 也不行」详述。
2. ❌ **cp311 + PyPI 的 `vllm==0.19.1`** —— **PyPI 的 aarch64 vLLM 轮子 PTX 是
   8.8（CUDA 12.9），而 L4T R36.5 驱动只到 12.6（PTX 8.5）**，运行时报
   `CUDA error: the provided PTX was compiled with an unsupported toolchain`。
   这一条只针对 **PyPI** 的轮子。
3. ✅ **cp311 + jetson 的 vLLM 0.19.0（重打包标签）—— 可行**。原文曾误以为
   「cp311 整条路线被堵死」，其实只是 PyPI 轮子的问题：jetson 那个
   `cp310` 轮子内部全是 `abi3` 扩展、不引用任何 Python C-API 符号，重打成
   `cp311` 标签后可以独立安装，实测 **1.27 Hz、输出与 3.10 逐 token 一致**。
   步骤见 [`JETSON_ORIN_POWER.md`](JETSON_ORIN_POWER.md) §1。

> 自编译 torch 的前提（若你要重做）：`TORCH_CUDA_ARCH_LIST=8.7`，且
> **`USE_DISTRIBUTED=1` 必须打开** —— vLLM 无条件 import
> `torch.distributed._functional_collectives` → `torch.distributed.distributed_c10d`，
> 少了它 vLLM 连 import 都过不去。

## 脚本清单

本目录针对 Orin 新增了以下脚本。**只有前两个是日常使用的**：

| 脚本 | 作用 | 何时用 |
|---|---|---|
| **`serve_orin.sh`** | 启动 `lightnav-serve`（preflight + LD_LIBRARY_PATH + 功耗检查 + READY 等待） | **主入口** |
| **`smoke_orin.py`** | 经 WebSocket 测端到端延迟，打印 Hz 与建议 `v_max` | 每次启动后验证 |
| `monitor_power.sh` | 每 0.5 s 落盘 `VDD_IN` 电压/电流/功率 | 排查断电（断电后重启看最后读数） |
| `load_engine.py` | 不经 WS 直接建引擎并跑 `--bench N` | 隔离「引擎慢」还是「网络/协议慢」 |
| `test_hf_load.py` | 用纯 transformers 加载（不经过 vLLM） | **供电诊断的对照组**：它能在 MAXN 下跑完而 vLLM 不能 |
| `verify_tokenizer_parity.py` | 对比 transformers 4.57 与 5.8 的 tokenizer 输出 | 换 transformers 版本后证明 tokenization 未漂移 |
| `repackage_vllm_cp311.py` | 把 jetson 的 `cp310` vLLM 树重打成 `cp311` wheel | 需要给 Python 3.11 装原生 sm_87 vLLM（见 `JETSON_ORIN_POWER.md`） |

`serve_orin.sh` 常用旋钮：

```bash
MODEL_PATH=... TASK=vln|tracking PORT=8050 GPU_MEM_UTIL=0.45 \
  bash scripts/serve_orin.sh

# 之前启动失败过、旧进程还没退出时（会拒绝启动第二份，避免抢统一内存）
bash scripts/serve_orin.sh --stop
```

两个环境变量用于特殊情况：

- `MAXN_WARN=0` —— 静默 MAXN 下的断电警告（检查仍然会跑，不再拦截）。
- `NV_LIBS=1` —— 把 pip 的 `nvidia/*/lib` 前置（**仅调试**，在 Orin 上会拉进
  缺 sm_87 的数据中心库）。
