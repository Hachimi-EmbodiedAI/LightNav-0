# Jetson AGX Orin 环境与功耗问题专题文档

本目录针对 **Jetson AGX Orin 64GB（L4T R36.5.2 / CUDA 12.6）** 在运行 LightNav-0 时遇到的两个核心底层问题进行梳理与诊断，面向初学者，力求条理清晰、结论明确。

| 文档 | 核心主题 | 一句话结论 |
|---|---|---|
| [01-system-env-and-pytorch-build.md](01-system-env-and-pytorch-build.md) | `jtop` 系统体检、环境缺陷分析与自编译 PyTorch 原理 | 当前基础环境（JetPack 6.2.3 / CUDA 12.6 / cuDNN 9.3）完整健康；主环境 `.venv310` 用社区预编译轮子，而 `.venv311` 因「`aarch64 + sm_87 + CUDA 12.6 + cp311 + torch 2.10`」全网无现成包才需要自编译 PyTorch。 |
| [02-power-limit-shutdown-diagnosis.md](02-power-limit-shutdown-diagnosis.md) | `MAXN` 无限制功耗整机断电原因诊断与优化方案 | **并非 LightNav-0 代码 Bug**；实测发现本机走的是 ~12V 高内阻供电链路（仅 25W 负载电压就暴跌近 1V），且旧监控脚本漏测了 CPU 供电通道，导致 `MAXN` 下启动瞬态电流触发硬件掉电/过流保护。 |

> 日常启动、运维与真机状态请继续参考上级目录文档：[JETSON_ORIN.md](../JETSON_ORIN.md)、[M1_DEPLOY_STATUS.md](../M1_DEPLOY_STATUS.md)、[M1_OPS_CHEATSHEET.md](../M1_OPS_CHEATSHEET.md)。
