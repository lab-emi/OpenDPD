# OpenDPD 2.4：MATLAB Toolbox 与实验互操作

日期：2026-09-28；开发进展更新：2026-09-29。状态：演进提案与首版开发记录。当前可用范围见[Toolbox 开发预览](../tutorials/matlab-toolbox.md)，其余里程碑仍是设计目标。

## 1. 建议的版本方向

**把 2.4 的主线定为：让 MATLAB、Python 和 Studio 使用同一套 OpenDPD 实验与模型。**

用户应能在 MATLAB 中导入 I/Q、训练 PA/DPD、取得预失真波形，并在 Studio 中查看同一次实验的曲线、配置和结果。这样可以把 OpenDPD 接入实验室已有的信号生成、仪器控制和分析流程。

建议优先级：

| 优先级 | 方向 | 用户得到什么 |
| --- | --- | --- |
| P0 | 稳定 Python SDK + MATLAB Toolbox | 从 MATLAB 完成一条可复现的 PA → DPD → 波形导出流程 |
| P0 | I/Q、模型和结果的数据契约 | 交换数据时明确采样率、幅度、状态和指标含义 |
| P1 | MATLAB 指标交叉验证 | 对同一波形核对 EVM/ACLR，补上现有验证记录 |
| P1 | 实验室测量闭环 | 串起导出、播放、采集、回导和实测评分，保存实验来源 |
| P2 | Simulink 与部署互操作 | 先验证少量模型的流式推理，再扩展执行后端 |

MathWorks 已有神经网络 DPD、在线训练及残差 RNN 对比示例。因此，OpenDPD 的产品重点可以放在可复现的模型比较、PyTorch 自定义模型、统一实验记录，以及训练到实测的衔接上。[MathWorks DPD 与 PA 工作流](https://www.mathworks.com/help/comm/dpd-and-pa-modeling.html)

## 2. 可以成为 MATLAB Toolbox 吗？

**可以先做由 OpenDPD 维护的第三方工具箱，交付 `OpenDPD.mltbx`。** MATLAB 支持把代码、示例和文档打包为 `.mltbx`；可直接分发，也可提交到 MATLAB Central File Exchange。[工具箱打包与分享](https://www.mathworks.com/help/matlab/matlab_prog/interactively-create-and-share-packages.html)

建议产品名称为 **OpenDPD Toolbox for MATLAB**。安装后提供 `opendpd.*` 函数和入门示例。进入 MathWorks 官方产品目录需要另行讨论合作；本提案的交付目标是第三方 Add-On，尚无官方收录或背书安排。

工具箱可以只负责 MATLAB 接口，训练和推理继续调用 Python/PyTorch。MATLAB 原生支持用 `py.` 调用 Python 模块，这条技术路径已有官方接口支持。[从 MATLAB 调用 Python](https://www.mathworks.com/help/matlab/matlab_external/ways-to-call-python-from-matlab.html)

## 3. 当前代码提供了什么基础

以下以分支起点 `7418ba35a480366f6578e20c22f4a36bacb5dbe9` 为准。

| 已有基础 | 对 2.4 的意义 / 需要补齐的部分 |
| --- | --- |
| [Python API](https://github.com/lab-emi/OpenDPD/blob/7418ba35a480366f6578e20c22f4a36bacb5dbe9/opendpd/api.py)：`train_pa`、`train_dpd`、`run_dpd` 等 | 可以做联通实验；训练入口仍调用 `Project`/旧训练步骤，返回路径字典，需要补充面向 workspace/run 的稳定接口 |
| [服务层](https://github.com/lab-emi/OpenDPD/blob/7418ba35a480366f6578e20c22f4a36bacb5dbe9/opendpd/services/experiments.py)、[任务运行时](../architecture/runtime.md) | 已有配置、队列、取消、状态、日志和产物，可作为 SDK 的共用实现 |
| [Studio HTTP API](../architecture/api.md) | 已有 `/api/v1`、OpenAPI、非浏览器会话引导和结果读取，可支撑另一种客户端连接方式 |
| [数据导入](https://github.com/lab-emi/OpenDPD/blob/7418ba35a480366f6578e20c22f4a36bacb5dbe9/opendpd/services/datasets.py) | 当前支持 CSV、NPY、NPZ 和 split 目录；MAT 文件适配需要新增 |
| [MATLAB 仪器脚本](https://github.com/lab-emi/OpenDPD/blob/7418ba35a480366f6578e20c22f4a36bacb5dbe9/Matlab/RS_SMW200A_DPD_Test.m)与[采集脚本](https://github.com/lab-emi/OpenDPD/blob/7418ba35a480366f6578e20c22f4a36bacb5dbe9/Matlab/N9042B_IQdownload.m) | 有 SMW200A/N9042B 工作流经验；需将地址、仪器配置和会话管理参数化 |
| [MATLAB EVM 脚本](https://github.com/lab-emi/OpenDPD/blob/7418ba35a480366f6578e20c22f4a36bacb5dbe9/Matlab/calculate_200MHz_256QAM_evm.m) | 与特定 200 MHz/256QAM 波形绑定，可作为历史信号参考；通用指标仍需独立验证 |
| [波形验证协议](../protocols/waveform-profiles.md) | 已规定 MATLAB 交叉验证方法与误差预算，记录目前为未运行 |
| [流式契约](../architecture/streaming.md)、[C99 定点导出](../tutorials/deployment-export.md) | 已有 GRU/GMP 流式基础和单层 GRU 定点参考，可用于后续 Simulink/部署验证；支持范围须按模型注册表声明 |

## 4. 联动路径与技术取舍

| 路径 | 建议阶段 | 主要价值 | 工程约束 |
| --- | --- | --- | --- |
| MATLAB → `py.opendpd.sdk` → OpenDPD | 2.4.0 首选 | 直接接入当前 Python 生态，便于交换数组和调用自定义模型 | 需固定兼容的 MATLAB/Python/PyTorch 组合，测量数据复制与调用开销 |
| MATLAB → 本地 HTTP API → OpenDPD | 备用连接方式；远程 GPU 在后续扩展 | 复用任务队列，MATLAB 进程无需加载训练依赖 | 需处理会话 cookie、CSRF、上传和任务轮询；远程使用先采用 SSH 隧道 |
| OpenDPD → MATLAB Engine → MATLAB | 交叉验证或复用 MATLAB 脚本时选用 | Python 侧可调用已有 MATLAB 信号处理函数 | MATLAB Engine 方向是 Python 调 MATLAB，需要安装 MATLAB；仅装 Runtime 不够 |
| PyTorch 模型 → MATLAB 原生网络 / Simulink | 后续逐模型试验 | 将已验证的推理接入 MATLAB/Simulink 系统仿真 | 自定义算子、递归状态、量化和前视缓冲均需逐项验证 |

Engine 的方向和安装要求见 [MATLAB Engine 文档](https://www.mathworks.com/help/matlab/matlab-engine-for-python.html)。原生模型导入遇到不支持的算子可能生成占位函数，因此发布时应给出逐模型支持清单。[PyTorch 网络导入](https://www.mathworks.com/help/deeplearning/ref/importnetworkfrompytorch.html)

原型阶段建议先用 `pyenv(..., ExecutionMode="OutOfProcess")` 调试。它便于重启 Python、隔离库冲突，但存在跨进程开销，单次变量传递还有 2 GB 限制。正式默认模式应由实测决定；MathWorks 通常推荐性能更好的 `InProcess`。[Python 进程模式](https://www.mathworks.com/help/matlab/matlab_external/out-of-process-execution-of-python-functionality.html)

### SDK 和任务生命周期

新增 `opendpd.sdk`，对外提供 workspace、dataset、job、model 和 result 的稳定句柄，内部复用 services、schemas 与 runtime。

- 训练调用返回 job/run ID，随后可查询、等待、取消和读取结果。
- 同一 workspace 只由一个 supervisor 管理。Studio 已运行时连接现有实例；独立使用时启动持有同一 workspace 锁的无界面服务。
- MATLAB 客户端关闭或 Python 桥接进程重启后，可凭 run ID 重连。服务退出时沿用现有取消与恢复规则，明确记录作业结果。
- 大数组经过文件暂存或分块传递；HTTP 的 JSON 请求体仅放配置与引用，复用已有文件上传路径。
- 返回 MATLAB 的对象以 struct、table 和数值数组为主；错误包含可操作的信息和 run ID。

### 目标使用体验

以下基本接口已有开发实现。示例的 `x`/`y` 为同次采集的 PA 输入/输出，`xTest` 为独立测试波形。当前 `apply` 支持 `gru`、`tres_gru`、`gmp`、`mp_ls`、`gmp_ls` 的 CPU 离线分段推理（与评估器逐样本一致），`gru` 与 `gmp` 另有有状态流式执行；默认 `Execution="auto"`：有流式变体的模型走流式，其余走离线分段（依据 `docs/performance/matlab-apply-semantics.md` 的预注册测量）。

```matlab
opendpd.setup(PythonExecutable="/path/to/python");
p = opendpd.openProject("lab-pa");
ds = opendpd.importIQ(p, x, y, SampleRate=fs, Bandwidth=bw, SegmentSamples=2048);  % 无默认值

pa = opendpd.wait(opendpd.trainPA(p, ds, Model="gru"));
dpd = opendpd.wait(opendpd.trainDPD(p, ds, PA=pa, Model="gru"));
[u, info] = opendpd.apply(dpd, xTest, Execution="offline_segmented");

opendpd.openStudio(p);
```

`setup` 配置用户选定的 Python 环境；`doctor` 检查版本、模块、设备和服务连接。入门示例必须说明测试波形的来源、模型有效范围及输出尺度。

## 5. 优先固定数据与数值行为

### I/Q 交换

首版建议接受 MATLAB `N×1` 复数向量，并将行向量显式转为列向量；内部转换为 OpenDPD 的 `N×2` I/Q 数组。

契约至少包含：

- schema 版本、采样率 Hz、信号带宽和 I/Q 列顺序。
- `single`/`double` 与 `float32`/`float64` 映射；训练需要转换精度时记录转换。
- 幅度尺度、归一化规则、目标增益；物理功率 dBm 单独记录。
- x/u/y 的角色、数据和模型哈希、连续 train/val/test 分割与保护间隔。
- 离线分段或流式执行、分段长度、状态重置、预热、前视样本和输出有效范围。
- 评估 profile 及版本，结果的仿真/实测来源。

MAT 文件由 MATLAB 自己读取（`whos`/`load`），因此任意 MAT 版本（含 v7.3/HDF5）都支持，并明确变量映射；此前基于 SciPy `loadmat` 的方案不能读取 v7.3。[SciPy MAT 文件支持范围](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html)

### 模型与结果

模型接口应携带结构配置、checkpoint 哈希、预处理信息、幅度尺度及执行语义。MATLAB 和 Studio 读取同一套 run/result/artifact 记录，避免分别推测配置。

数值验收分两层：

1. **桥接一致性：**固定数据、checkpoint、设备、精度和执行语义，比较 Python 直接推理与 MATLAB 桥接的样本及指标；先定义容差，再运行验收。
2. **独立算法核对：**使用 MATLAB 指标实现与 OpenDPD profile 对同一信号评分，记录同步、均衡、滤波、参考功率和积分带宽。沿用[既有波形协议](../protocols/waveform-profiles.md)的误差预算。

EVM、ACLR 和 Arena AER 保留各自的定义与字段。跨语言调用成功之后，还要完成这些数值验证，才能宣称结果一致。

## 6. Toolbox 组织、安装与兼容性

建议延续仓库已有 `Matlab/` 大小写，新增以下目录：

```text
Matlab/toolbox/
  +opendpd/          MATLAB 公共接口与 private 适配代码
  examples/         入门、数据交换、指标对比示例
  tests/            MATLAB 接口与数值一致性测试
  docs/             安装、依赖、支持矩阵、故障诊断
  buildfile.m       可复现的 .mltbx 打包入口
opendpd/sdk/        Python 公共接口
```

`.mltbx` 包含 MATLAB 源码、文档和小型示例数据。Python 环境由明确的安装步骤配置，并提供经过测试的依赖版本记录。打包使用固定工具箱标识和 `matlab.addons.toolbox.ToolboxOptions`；该编程接口自 R2023a 起可用。[编程打包接口](https://www.mathworks.com/help/matlab/ref/matlab.addons.toolbox.packagetoolbox.html)

| 使用场景 | 计划依赖 |
| --- | --- |
| 基础桥接、训练、波形导出 | MATLAB + 兼容 CPython + OpenDPD/PyTorch |
| MATLAB 原生神经网络导入 | 按所用导入功能增加 Deep Learning Toolbox 及相应支持包 |
| LTE/5G 波形与独立指标参考 | 按示例声明 Communications、LTE 或 5G Toolbox |
| MATLAB 仪器控制 | 按适配器声明 Instrument Control Toolbox、VISA 和驱动 |
| Simulink 系统仿真 | Simulink，以及示例实际使用的其他工具箱 |

候选首批验证组合是 MATLAB R2024b、R2025b、R2026b 与 Python 3.11，优先 Windows/Linux CPU，再验证 Linux CUDA。Python 3.11 位于这些 MATLAB 版本的官方兼容范围内；OpenDPD 全部依赖能否在具体平台安装并运行，还需要实际验证。[官方 Python 兼容矩阵](https://www.mathworks.com/support/requirements/python-compatibility.html)

目前 `pyproject.toml` 默认包含桌面界面依赖。2.4 可评估独立的 headless 安装方式，降低 MATLAB/计算节点的安装负担，同时为现有安装入口提供迁移兼容性。

## 7. 分阶段交付与验收

| 阶段 | 交付 | 完成依据 |
| --- | --- | --- |
| M0：连通原型 | 一份 MATLAB 脚本调用 Python，交换复数 I/Q，运行固定 GRU 推理 | 在实际 MATLAB 中记录环境、样本误差、耗时和内存；据此确定默认连接模式 |
| M1：2.4.0 alpha | SDK、MAT v7 适配、训练 job、结果读取、任意受支持测试波形的 apply 接口 | 从合成数据走完 PA → DPD → u；取消、重连和 Studio 读取同一次 run 均通过 |
| M2：2.4.0 | 可安装的 `.mltbx`、帮助、示例和支持矩阵 | 干净环境安装/卸载、路径含空格与中文、CPU 数值一致性及安装包运行通过 |
| M3：2.4.x | MATLAB 指标交叉验证、首个实验室链路的完整记录 | 保存 MATLAB/工具箱版本、参考波形、误差报告；实测记录绑定导出与采集文件 |
| M4：后续预览 | 先为因果 GRU/GMP 评估 Simulink 流式接口，再评估受支持 GRU 的 C/MEX 路径 | 验证状态跨块延续、reset、不同分块和数值误差；代码生成能力单独验收 |

首个 `.mltbx` 的合成数据示例应只需基础 MATLAB 和 Python 环境。指标验证和真实仪器示例按依赖单独提供，便于用户按自己的实验条件逐步启用。

实测闭环可以先由 MATLAB 操作仪器、OpenDPD 接收带来源记录的采集数据。开发自动适配器时复用[现有仪器会话契约](../architecture/instruments.md)，并在对应实验室链路完成验证。Simulink 第一阶段定位为系统仿真；实时吞吐、代码生成和 FPGA 部署分别建立支持与测试记录。

## 8. 当前工作记录与下一步

- 已从 `7418ba3` 建立 `codex/studio-2.4` 和独立 worktree。
- 2.3 工作目录的未提交 Arena 修改未进入此起点；在 2.3 收尾提交稳定后，再合入相关提交并复核受影响的 API 与协议。
- 已实现 `opendpd.sdk` 与 `Matlab/toolbox`：环境诊断、共享服务、I/Q/MAT v7 导入、训练与任务管理、GRU 推理、标准 DPD 导出和示例。
- Python 包版本暂沿用 2.3 基线；SDK 协议版本为 1，Toolbox 版本随发布，当前为 `2.4.0`（未发布）。发布前统一调整 2.4 元数据；使用者目前必须安装此工作树的 Python 代码。
- `apply` 经 `trained_model` 重建与评估器相同的网络；五个模型的 PA/DPD 离线推理、分段独立性、`gru`/`gmp` 流式契约与评估器的一致性由 `tests/integration/test_apply_parity.py` 逐项验证。最新的验证记录见 `docs/releases/2.4.0.md`。
- MATLAB R2026a（Linux）上的真实测试、`.mltbx` 打包、在新会话中的安装/示例/卸载均已执行，详细数字见 `docs/releases/2.4.0.md`，不在此重复。
- MATLAB CI 现在随涉及工具箱的 PR 触发（Linux 为门禁，Windows 先只报告）。其他 MATLAB 版本、Windows/macOS 和 CUDA 桥接仍需对应环境的验证。

**下一步：扩展兼容性验证。** 以已通过的 R2026a/Linux/CPU 为基础，验证 R2024b/Python 3.11、Windows 和 CUDA，再扩展模型支持。
