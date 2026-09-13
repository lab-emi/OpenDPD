# OpenDPD Studio 下一步改进计划：从四位 RF 研究者的近五年论文出发

日期：2026-09-13

对象：Leo de Vreede、Morteza Alavi、Masoud Babaie、Marco Spirito 及相近研究工作流的用户

产出性质：文献驱动的产品分析与实施建议；尚未经过四位本人访谈或确认。

## 1. 建议先做什么

**下一阶段应把现有 Studio 的结果页和比较页，发展成能支持 RF 实验讨论、方案取舍和论文出图的工作台。** 第一优先级是让用户快速回答：这个结果在什么条件下得到，指标具体怎么算，比较是否公平，代价是多少。

四条研究线索共同支持这个方向，但侧重点不同：

| 研究者 | 最值得优先支持的研究判断 | 对 Studio 的直接要求 |
|---|---|---|
| Leo de Vreede | 满足线性度要求后，能否提高平均输出功率、效率、带宽或降低实现成本？ | 同条件的线性度—功率—效率比较；跨条件表现；硬件成本与性能的取舍图 |
| Morteza Alavi | DPD 是否适配真实数字发射机的 I/Q 或极坐标结构、工作模式和非理想因素？ | I/Q 与相位诊断、逐载波表现、简单 DPD 基线、波形播放与实测闭环 |
| Masoud Babaie | 限制来自算法，还是时序、相位、校准、供电及电路实现？ | 延迟与相位误差诊断、校准消融、实时实现预算，以及测量接收链的信息 |
| Marco Spirito | 改善幅度是否可信，校准、参考面、仪器与重复测量能否支撑结论？ | 测量过程可追溯、原始与处理后数据、重复性与不确定度、可复现图表 |

这里的“要求”是从论文研究问题推导的产品假设，**不是四位本人提出的功能需求，也不能据此断言个人鼠标、快捷键或界面语言习惯**。研究方向与文献依据见第 3、4 节。

建议启动一个 **8 周迭代，首个 2 周只交付 RF 结果审阅与比较的最小闭环**。利用已有的证据标记、信号链、比较规则、实测导入、条件评估、流式执行和定点导出，先把这些能力组织成研究者能直接使用的界面。复杂仪器自动化、完整 OTA 工作流和新模型探索放到后续。

## 2. 检索范围、阅读深度与判断边界

检索重点为 **2021-09-13 至 2026-09-13** 的成果，以正式会议日期或期刊卷期日期归入时间窗口。R01 的期刊卷期为 2021 年 12 月，但 online-first 早于窗口起点；它作为明确标注的边界样本保留。论文版本按同一工作合并，避免把 arXiv、会议版和对应期刊记录重复计数。

采用 TU Delft 教师论文列表和机构知识库交叉检索，再追踪 DOI、作者公开稿及 arXiv 全文。作者身份核对入口：[de Vreede](https://microelectronics.tudelft.nl/People/bio.php?id=26)、[Alavi](https://microelectronics.tudelft.nl/People/bio.php?id=480)、[Babaie](https://microelectronics.tudelft.nl/People/bio.php?id=277)、[Spirito](https://microelectronics.tudelft.nl/People/bio.php?id=264)。注意 L. C. N. de Vreede 等姓名变体；Masoud Babaie 与 Masoud Pashaeifar 是不同作者。

本报告选取 **20 篇代表性论文，其中 14 篇获取全文并阅读相关方法、实验、结果和关键图表，6 篇依据摘要与出版信息分析**。长篇论文按与 OpenDPD 相关章节精读；不是逐页翻译，也不声称完整穷尽四人五年内全部发表。直接涉及 DPD、发射机、实现成本和测量可信度的论文权重较高；接收机、低温器件和振荡器论文只作为邻近研究方向的辅助证据。

下文区分三种内容：

- **论文事实**：有来源支持的方法、实验条件或作者报告的结果。
- **产品推断**：这些研究任务可能需要的 Studio 能力；注明适用范围。
- **交互假设**：有待真实用户任务测试验证的界面组织方式。

## 3. 文献证据：哪些细节真正影响产品设计

阅读标记：**F** = 获取全文并阅读相关章节及关键图表；**A** = 摘要和出版信息。表中“关联作者”仅列本次研究的四位，不是完整作者名单。标题链接指向 DOI 或 arXiv，另列机构知识库公开稿入口，便于复核。方法和数值以本次获取的公开稿为依据，出版年份以正式记录为准。

### 3.1 数字发射机、功率与负载条件

| ID / 年份 | 论文与关联作者 | 阅读 | 与 OpenDPD 最相关的证据及边界 |
|---|---|---|---|
| R01 / 2021 | [A Wideband Four-Way Doherty Bits-In RF-Out CMOS Transmitter](https://doi.org/10.1109/JSSC.2021.3105542)，JSSC；de Vreede、Alavi | A | 数字 I/Q 发射机采用低复杂度的 2×1D DPD；论文分别讨论漏极效率和系统效率，并给出调制、PAPR、采样与功率条件。支持提供轻量基线和明确效率边界。正式卷期为 12 月，online-first 早于检索窗口。 |
| R02 / 2022 | [A Four-Way Series Doherty Digital Polar Transmitter at mm-Wave Frequencies](https://doi.org/10.1109/JSSC.2021.3133861)，JSSC；de Vreede、Spirito、Babaie | F · [公开稿](https://pure.tudelft.nl/ws/portalfiles/portal/146481335/A_Four_Way_Series_Doherty_Digital_Polar_Transmitter_at_mm_Wave_Frequencies.pdf) | Fig. 18 展示幅度与相位路径的延迟失配如何影响 EVM/ACLR，Fig. 19 展示功耗分解。300 MHz OFDM 实验中平均漏极效率与系统效率分别为 18% 和 8%，说明仅展示 PA 效率会遗漏大量代价。论文的延迟容限对应特定实验，不能直接作为通用门限。 |
| R04 / 2023 | [A Low-Complexity Digital Predistortion Technique for Digital I/Q Transmitters](https://doi.org/10.1109/IMS37964.2023.10187914)，IMS；de Vreede、Alavi | F · [公开稿](https://repository.tudelft.nl/record/uuid%3A4952bd91-34e9-4690-a89a-22acce3343dd) | 通过正交 I/Q 扫描获得 code-to-AM、code-to-PM 特性；Doherty 支路切换、时间/相位对齐及循环播放的边界连续性会影响校正。包含非连续多载波实验和各通道 EVM，支持架构相关诊断，而非只看合并频谱。 |
| R05 / 2023 | [An Inverted Doherty Power Amplifier Insensitive to Load Variation With an Embedded Impedance Sensor in Its Output Power-Combining Network](https://doi.org/10.1109/TMTT.2023.3277081)，TMTT；Alavi、de Vreede | A | 负载变化下利用传感器及硬件调节维持表现。支持记录负载、供电、驱动和校正状态；不能推导为“只靠 DPD 就能修复所有负载变化”。 |
| R08 / 2024 | [The Efficiency and Power Utilization of Current-Scaling Digital Transmitters](https://doi.org/10.1109/TMTT.2023.3336984)，TMTT；Alavi、de Vreede | F · [公开稿](https://repository.tudelft.nl/record/uuid%3A2c19d97f-7073-4220-8df8-79958ed2819c) | 比较极坐标、笛卡尔及多相架构，讨论占空比、PAPR 和功率利用率；区分理论假设、漏极效率与系统效率。支持以调制平均效率和系统边界评价方案，不能只按峰值效率排序。 |
| R12 / 2025 | [A 4×Two-Way mm-Wave Doherty CMOS PA](https://doi.org/10.1109/TMTT.2025.3569155)，TMTT；de Vreede、Alavi | F · [公开稿](https://repository.tudelft.nl/record/uuid%3A13d8ea6f-e9f5-4b6b-9007-c0a7a8ff9106) | Fig. 16–21 涉及工作模式、AM/AM、AM/PM、功率/效率、无 DPD 与 AI-DPD，以及冻结 DPD 后的 VSWR/角度变化。**2 GHz 是无 DPD 调制实验；AI-DPD 示例为 400 MHz、1 GHz**，不能宣传成已经完成 2 GHz DPD。 |

### 3.2 OpenDPD、神经网络与硬件实现

| ID / 年份 | 论文与关联作者 | 阅读 | 与 OpenDPD 最相关的证据及边界 |
|---|---|---|---|
| R06 / 2024 | [OpenDPD: An Open-Source End-to-End Learning & Benchmarking Framework for Wideband Power Amplifier Modeling and Digital Pre-Distortion](https://arxiv.org/abs/2401.08318)，[ISCAS](https://doi.org/10.1109/ISCAS58744.2024.10558162)；de Vreede、Alavi | F | 固定 PA 代理模型，标准化数据和划分协议；Table I 汇报多个种子的统计结果，部分图展示最佳种子；仿真和实测结果分开。支持成组实验、统一基线和清楚的统计口径。 |
| R07 / 2024 | [MP-DPD: Low-Complexity Mixed-Precision Neural Networks for Energy-Efficient Digital Predistortion of Wideband Power Amplifiers](https://arxiv.org/abs/2404.15364)，[MWTL](https://doi.org/10.1109/LMWT.2024.3386330)；de Vreede、Alavi | F | 混合精度要同时看权重、激活、乘加和存储访问；特征提取仍使用 FP32。文中的能耗收益基于工艺和操作成本模型，**不是芯片实测功耗**。支持分模块精度与成本账本。 |
| R09 / 2025 | [DeltaDPD: Exploiting Dynamic Temporal Sparsity in Recurrent Neural Networks for Energy-Efficient Wideband Digital Predistortion](https://arxiv.org/abs/2505.06250)，[MWTL](https://doi.org/10.1109/LMWT.2025.3565004)；de Vreede | F | 动态阈值影响时域稀疏性、有效计算量和线性度；跳过计算是否转化为节能依赖实现。**Table I 的 EVM 以输入波形和实测输出比较，而非理想符号网格**，且波形包含轻度 CFR；此指标不能直接等同标准解调 EVM。 |
| R10 / 2025 | [TCN-DPD: Parameter-Efficient Temporal Convolutional Networks for Wideband Digital Predistortion](https://arxiv.org/abs/2506.12165)，[IMS](https://doi.org/10.1109/IMS40360.2025.11103923)；de Vreede | F | 膨胀卷积、深度可分离卷积、激活函数和多个种子的比较体现参数效率；该工作使用非因果结构。Table II 的线性化结果是 **SIM / PA 代理模型仿真**，文中硬件验证仍在进行。参数少不等于缓冲少、延迟低或实测功耗小。 |
| R11 / 2025 | [DPD-NeuralEngine: A 22-nm 6.6-TOPS/W/mm² Recurrent Neural Network Accelerator for Wideband Power Amplifier Digital Pre-Distortion](https://arxiv.org/abs/2410.11766)，[ISCAS](https://doi.org/10.1109/ISCAS56072.2025.11043563)；de Vreede | F | 12-bit 定点、特征提取、非线性近似和调度共同决定实现。报告的 250 MSps、7.5 ns、195 mW 等 ASIC 指标来自**布局后仿真**；RF 实测线性化与芯片成本证据是不同环节。2 GHz 时钟也不等于 2 GSps 吞吐。 |

### 3.3 相位、时序、接收链与器件环境

| ID / 年份 | 论文与关联作者 | 阅读 | 与 OpenDPD 最相关的证据及边界 |
|---|---|---|---|
| R03 / 2022 | [A DPLL-Based Phase Modulator Achieving -46dB EVM with A Fast Two-Step DCO Nonlinearity Calibration and Non-Uniform Clock Compensation](https://doi.org/10.1109/VLSITechnologyandCir46769.2022.9830398)，VLSI；Alavi、Babaie | F · [公开稿](https://repository.tudelft.nl/record/uuid%3Afebaf40d-bf50-448b-931c-ce3503c309c4) | 分步骤校准和补偿的消融、收敛过程，比最终一个 EVM 数字更有解释力。64PSK 有实测；256QAM 结果结合实测 PM 与 MATLAB 理想 AM，属于混合证据，不能当成完整发射机实测。 |
| R13 / 2025 | [A Sub-7-GHz Linear Receiver for 5G Local Area Base Station Applications](https://doi.org/10.1109/JSSC.2025.3551673)，JSSC；Babaie | F · [公开稿](https://repository.tudelft.nl/record/uuid%3A92f00cf7-3939-4dce-b934-383c706af381) | 接收机的噪声、线性度、带宽和功耗存在取舍，实测表现随频率变化。对 OpenDPD 的关联是间接的：采集链可能限制观察到的失真；这篇论文不是专门提出 DPD 反馈接收机。 |
| R18 / 2025 | [Characterization and Modeling of MOSFET Gate Capacitance at Cryogenic Temperatures](https://doi.org/10.1109/ESSERC66193.2025.11214132)，ESSERC；Babaie | A | 温度、器件几何、频率和激励幅度影响模型与测量的匹配。辅助支持“模型适用条件必须记录”，不构成开发量子控制或低温器件 CAD 的理由。 |
| R19 / 2026 | [A 74fs-Jitter, −59dBc-Spur Fractional-N DPLL Using a Supply-Resilient Time-Amplifying Dual-Ramp DTC](https://doi.org/10.1109/ISSCC49663.2026.11409328)，ISSCC；Babaie | A | 供电扰动下的抖动、杂散与校准表现，需要说明扰动条件。支持可控非理想因素的诊断和消融；不能推导为随机相位噪声都可以由 DPD 消除。 |
| R20 / 2026 | [A 22-to-25GHz CMOS Non-Magnetic Balanced Circulator Achieving at Least 20dB TX-RX Isolation for an Antenna VSWR of 2](https://doi.org/10.1109/ISSCC49663.2026.11408990)，ISSCC；Spirito、Babaie | A | 天线失配与 TX/RX 隔离、插入损耗相关。辅助支持记录系统状态和负载条件，不应扩张为“Studio 负责优化整个环行器”。 |

### 3.4 测量可信度与空间条件

| ID / 年份 | 论文与关联作者 | 阅读 | 与 OpenDPD 最相关的证据及边界 |
|---|---|---|---|
| R15 / 2023 | [A Rigorous Analysis of the Random Noise in Reflection Coefficients Synthesized via Mixed-Signal Active Tuners](https://doi.org/10.1109/ARFTG57476.2023.10279048)，ARFTG；Spirito | F · [公开稿](https://pure.tudelft.nl/ws/portalfiles/portal/163269060/A_Rigorous_Analysis_of_the_Random_Noise_in_Reflection_Coefficients_Synthesized_via_Mixed_Signal_Active_Tuners.pdf) | 混合信号有源调谐器的噪声通过系统传播到反射系数，不能只用 VNA 本身的噪声估计整个系统的不确定度。支持记录误差来源与传播方法；重复标准差不能替代完整不确定度。 |
| R14 / 2025 | [Advances in Hardware and Measurement Techniques to Enable Better Millimeter-Wave Device Characterization and Modeling](https://doi.org/10.1109/MMM.2024.3524849)，IEEE Microwave Magazine；Spirito | F · [公开稿](https://repository.tudelft.nl/record/uuid%3A7c9bd3b7-d32f-413e-88a8-59fd160b3a1c) | 讨论源谐波、接收机压缩、功率校准、探针落点和误差传播。Fig. 14–17 与 Table I 显示，不同测量配置会改变可信范围。支持参考面、校准记录、仪器限制和条件相关的误差展示。 |
| R16 / 2025 | [Test-Fixture Design Flow for Broadband Validation of CMOS Device Models up to (sub)mm-Waves](https://doi.org/10.1109/TMTT.2025.3586815)，TMTT；Spirito | A | 夹具、探针放置和校准残差影响宽带器件模型验证。支持将 fixture、去嵌入与校准版本关联到数据和模型。 |
| R17 / 2025 | [Performance Characterization of an Active Phased Array Antenna by Simultaneously Measuring the Radiation Pattern and the Error Vector Magnitude](https://doi.org/10.23919/EuMC65286.2025.11235079)，EuMC；Spirito | F · [公开稿](https://repository.tudelft.nl/record/uuid%3A716aa976-34a3-4600-90c2-f0b1532733de) | 将方向图与 EVM 放在同一测量流程中，并展示输入功率变化下从噪声限制到非线性限制的趋势。文中 EVM 由波束正向接收链测得，**不能扩大解读为同步测出完整三维 EVM 地图**。支持后续关联波束条件与线性度。 |

这些论文共同提醒：**一个“更好”的数字，可能来自不同指标定义、不同功率、不同参考波形、不同证据类型，或不同的成本核算边界。** Studio 应把这些差异变成可见、可保存、可复核的信息。

## 4. 四位研究者可能最在意的 feature、展示细节与交互

以下引号内的问题是为设计构造的评审问题，不是本人原话。“信心”表示文献对研究需求推断的支持程度，不代表对个人行为的预测准确率。

### 4.1 Leo de Vreede：让 DPD 的价值回到发射机系统

**假设性评审问题：**“达到要求的线性度以后，这个方法让我多输出了多少功率，节省了多少系统能量，换了条件还成立吗？”

R02、R08、R12 将线性度、调制功率、效率、带宽与工作条件联系起来；R07、R09、R11 则把 DPD 的计算和实现成本纳入问题。因此，对这一需求的信心为**高**。[效率分析](https://doi.org/10.1109/TMTT.2023.3336984)、[DeltaDPD](https://arxiv.org/abs/2505.06250)、[毫米波 Doherty PA](https://doi.org/10.1109/TMTT.2025.3569155)

- **Feature：**输出功率扫描；满足指定 ACPR/EVM 门限的可行区域；线性度—效率—计算成本的 Pareto 图；冻结模型后的跨条件评估。
- **必须 show：**平均调制输出功率、PAPR、功率回退的参考、PA 漏极效率、PAE、整机效率及各自的功耗边界；MAC/sample、存储、吞吐与成本来源。
- **交互假设：**先固定波形、功率条件和比较协议，再叠加若干方法；从总览中的一个点下钻到对应 PSD、AM/PM、配置和原始测量。比频繁切换单个 run 更适合讨论取舍，需通过访谈验证。
- **演示重点：**选择目标线性度，看到哪些工作点达标，再比较平均输出功率及代价。缺少 DC 功耗时明确显示“未提供”，不从归一化 IQ 推算效率。

### 4.2 Morteza Alavi：让界面看得见数字发射机的结构

**假设性评审问题：**“失真来自哪条路径、哪个码值区域或哪个工作模式？一个更简单的 DPD 能否已经解决问题？”

R01、R04 明确涉及数字 I/Q 映射与低复杂度 DPD，R03 涉及相位调制器校准，R12 涉及工作模式与负载条件。对架构相关诊断、轻量基线和实测迭代的需求推断为**高**。[数字 I/Q DPD](https://doi.org/10.1109/IMS37964.2023.10187914)、[相位调制器校准](https://doi.org/10.1109/VLSITechnologyandCir46769.2022.9830398)

- **Feature：**可选的 Cartesian / Polar / Doherty 工作流模板；I/Q 扫描数据导入；码值—幅度/相位曲线；支路或模式切换区域诊断；逐载波频谱与误差；已有 MP/GMP 与神经模型的同协议比较。
- **必须 show：**I/Q 路径、相位参考、幅相路径失配、模型记忆范围、模式与偏置、载波布局、各载波功率/EVM、输入与 DPD 输出 PAPR、循环波形边界是否连续。
- **交互假设：**把“原始数据 → 对齐/校正 → 模型输出 → 实测输出”做成可反复切换的同图对照；支持表格编辑物理参数、保存实验配方、导出给 MATLAB/Python 和仪器脚本。论文使用 MATLAB/仪器工作流只能支持互操作需求，不能证明个人偏好的快捷键或 GUI 风格。
- **演示重点：**在一个真实 I/Q 扫描上定位误差集中区域，再对比简单模型和神经模型。只有具备相应扫描及结构假设时才提供 2×1D LUT；普通随机调制 IQ 不能自动冒充支路独立扫描。

### 4.3 Masoud Babaie：把误差来源和实时实现约束分开看清

**假设性评审问题：**“这个误差是非线性、路径延迟、校准不足，还是观测链噪声？补偿后在目标硬件上能否按时运行？”

R02、R03 对幅相延迟与校准的支持直接；R13、R18、R19 对接收链、环境和供电的关联较间接。前者推断信心为**高**，后者为**中**。这些证据支持诊断能力，不要求 Studio 变成完整电路仿真器。[数字极坐标发射机](https://doi.org/10.1109/JSSC.2021.3133861)、[接收机](https://doi.org/10.1109/JSSC.2025.3551673)、[2026 DPLL](https://doi.org/10.1109/ISSCC49663.2026.11409328)

- **Feature：**分数采样延迟和相位对齐的可视化；校准步骤消融；误差/收敛时间曲线；定点位宽、状态、前视样本和执行成本的联合检查。
- **必须 show：**延迟以 sample 和 ns 同时表达；因果性、前视、warm-up、状态重置、块边界误差；特征提取与非线性函数实现；真实目标硬件的吞吐/延迟来源。
- **交互假设：**先固定一个基线，逐项启用已定义的补偿，叠加曲线并保留配置差异；允许精确数值输入、恢复基线和保存消融集合。
- **演示重点：**展示已知延迟扰动对误差的影响和恢复，再看补偿是否增加前视或计算。人为注入的相位噪声、供电模型必须标为仿真；随机噪声不能承诺由确定性 DPD 全部消除。

### 4.4 Marco Spirito：让每个图和数字都能追到测量过程

**假设性评审问题：**“这个改善比测量波动大多少？参考面在哪里，校准怎么做，处理过程有没有改变结果？”

R14–R17 将校准、误差传播、测量配置与系统表征作为核心问题。对测量可追溯和不确定度呈现的需求推断为**高**；对具体仪器界面布局偏好的推断仍待验证。[测量综述](https://doi.org/10.1109/MMM.2024.3524849)、[有源调谐器噪声](https://doi.org/10.1109/ARFTG57476.2023.10279048)、[阵列测量](https://doi.org/10.23919/EuMC65286.2025.11235079)

- **Feature：**Measurement Session；重复采集管理；校准/参考面/去嵌入关联；输入文件、处理版本和模型的完整追溯；带重复性信息的模型—实测残差比较。
- **必须 show：**采集链路、频率和捕获带宽、增益/衰减、校准时间与版本、原始幅度单位、接收机噪声/压缩限制、独立采集次数、处理前后对照、误差条的含义。
- **交互假设：**从异常图点直接打开对应 capture 和处理步骤；保留仪器式游标、参考轨迹、坐标范围和单位；保存视图后重新打开应得到同一张图。
- **演示重点：**同一条件的多次采集叠加，指出结果是否受噪声底或漂移限制，并从图表导出完整证据。后续再把波束角、方向图或负载 Smith 图接入条件视图。

## 5. 当前 Studio 已经有什么，真正缺口在哪里

代码审阅基于工作区 HEAD **`49c2608c6c49`** 及 2026-09-13 可见实现；本计划不把历史规划文档中的目标自动当成已完成功能。下列链接指向本地实际文件，适合当前工作区阅读。此次仅编写计划，不修改功能代码。

| 能力 | 已有实现 | 下一步应补的缺口 |
|---|---|---|
| 数据与信号查看 | [DatasetDetailPage](https://github.com/lab-emi/OpenDPD/blob/d09ebb6b7ece2dc0b2f2b82ce5bed45bb6c8aadb/frontend/src/pages/DatasetDetailPage.tsx) 已有频谱、时域 IQ、星座/散点、AM/AM 与 AM/PM，以及元数据和版本信息 | 将同类诊断带入结果比较；增加物理条件、误差分布及多视图联动 |
| 图表交互 | [图表交互记录](https://github.com/lab-emi/OpenDPD/blob/d09ebb6b7ece2dc0b2f2b82ce5bed45bb6c8aadb/docs/baseline/studio-plot-interactions-2026-09-12.md) 已有滚轮平移、缩放、框选、快捷键和放大视图状态保持 | 参考轨迹固定、测量游标、关联选择、保存可复现视图；无需重新发明 pan/zoom |
| 实验创建 | [NewExperimentPage](https://github.com/lab-emi/OpenDPD/blob/d09ebb6b7ece2dc0b2f2b82ce5bed45bb6c8aadb/frontend/src/pages/NewExperimentPage.tsx) 已有三步向导、配置导入与专家编辑 | 增加复制后改一个变量、成组实验和物理条件模板；保留现有入门路径 |
| 结果与证据 | [ResultDetailPage](https://github.com/lab-emi/OpenDPD/blob/d09ebb6b7ece2dc0b2f2b82ce5bed45bb6c8aadb/frontend/src/pages/ResultDetailPage.tsx#L108)、[EvidenceBadge](https://github.com/lab-emi/OpenDPD/blob/d09ebb6b7ece2dc0b2f2b82ce5bed45bb6c8aadb/frontend/src/components/EvidenceBadge.tsx#L8) 已有证据标记、x→u→y、无 DPD 基线、缩放及代理模型覆盖信息 | 提升信息位置和可读性；扩展条件维度。当前幅度覆盖诊断不能解释为完整工作域已经验证 |
| 指标及比较 | [比较规则](https://github.com/lab-emi/OpenDPD/blob/d09ebb6b7ece2dc0b2f2b82ce5bed45bb6c8aadb/opendpd/core/metrics/compare.py)、[ComparePage](https://github.com/lab-emi/OpenDPD/blob/d09ebb6b7ece2dc0b2f2b82ce5bed45bb6c8aadb/frontend/src/pages/ComparePage.tsx#L47) 已限制不可比结果排名，提供表格、频谱叠加、CSV 与双配置差异 | 补充结构化 RF 条件和条件差异摘要；明确“同条件排名”与“跨条件趋势”两种任务 |
| 实测闭环 | [measurement schema](https://github.com/lab-emi/OpenDPD/blob/d09ebb6b7ece2dc0b2f2b82ce5bed45bb6c8aadb/opendpd/schemas/measurement.py)、[measurement service](https://github.com/lab-emi/OpenDPD/blob/d09ebb6b7ece2dc0b2f2b82ce5bed45bb6c8aadb/opendpd/services/measurements.py) 已关联播放文件、with/without-DPD 采集、温度、自由文本校准与整数样本对齐 | 结构化参考面/校准、重复采集、观测链限制、实测流程的分数延迟处理。通用[预处理 schema](https://github.com/lab-emi/OpenDPD/blob/d09ebb6b7ece2dc0b2f2b82ce5bed45bb6c8aadb/opendpd/schemas/dataset.py#L89) 已支持浮点延迟，不能宣称全项目缺少该能力 |
| 条件与适应性 | [conditions schema](https://github.com/lab-emi/OpenDPD/blob/d09ebb6b7ece2dc0b2f2b82ce5bed45bb6c8aadb/opendpd/schemas/conditions.py) 已有独立条件数据、source/target、zero-update/few-shot/full-retrain、预算和种子 | [RobustnessPage](https://github.com/lab-emi/OpenDPD/blob/d09ebb6b7ece2dc0b2f2b82ce5bed45bb6c8aadb/frontend/src/pages/RobustnessPage.tsx#L34) 当前侧重只读报告；补建卡片、批量规划和启动界面，复用既有后端 |
| 实现成本 | [流式协议](https://github.com/lab-emi/OpenDPD/blob/d09ebb6b7ece2dc0b2f2b82ce5bed45bb6c8aadb/docs/architecture/streaming.md)、[fixed-point schema](https://github.com/lab-emi/OpenDPD/blob/d09ebb6b7ece2dc0b2f2b82ce5bed45bb6c8aadb/opendpd/schemas/fixed_point.py#L107) 已有前视/状态语义、GRU 定点导出、C 参考验证和资源报告 | 补跨候选方案的成本比较、完整模块账本和外部目标硬件证据；CPU C 参考计时不能标成 FPGA/ASIC 实测 |
| 报告与分享 | [reports service](https://github.com/lab-emi/OpenDPD/blob/d09ebb6b7ece2dc0b2f2b82ce5bed45bb6c8aadb/opendpd/services/reports.py#L162) 已有 Markdown/HTML 报告和频谱图；结果页有 share/full 包导出 | 扩为可保存版式的多面板图、矢量导出、图表数据与生成说明，复用现有结果/包体系 |

特别需要利用已有的[指标协议](https://github.com/lab-emi/OpenDPD/blob/d09ebb6b7ece2dc0b2f2b82ce5bed45bb6c8aadb/docs/protocols/metric-profiles.md)：

- `legacy-opendpd-v1` 的 EVM 是仓库特定的频谱误差定义，**不是解调 EVM**；它应保留用于历史复现，但标题和说明必须清楚。
- `general-spectral-v1` 的 IBE、ACPR 与 legacy 定义不同；单位相同并不代表可直接合并排名。
- `ofdm-lte20-evm-v1` 已实现，但独立交叉验证仍待完成，目前隐藏在 GUI。优先完成验证再公开，不能只画出漂亮星座图就宣称完成标准 EVM 测量。
- 图表抽样、缩放和筛选不应改变正式结果。保存视图与改变评估协议应保持为两个明确操作。

## 6. 功能优先级和具体设计

优先级含义：P0 = 首个两周交付；P1 = 接下来的六周分批完成；P2 = 需要数据或硬件条件后再排期。

| 事项 | 优先级 | 用户收益 | 前提与主要风险 |
|---|---|---|---|
| RF 结果摘要 + 比较工作台 v1 | P0 | 一屏理解条件、证据和改进量 | 复用现有结果；不能绕过比较规则 |
| 可保存的图表视图与导出 v1 | P0 | 讨论时看到的图可以重开和复核 | 先覆盖频谱/已有图；不在两周内承诺全部出版格式 |
| Measurement Session 与重复采集 | P1 | 实测结果可追溯，能判断波动 | 需要一组真实重复采集；schema 与对齐协议需版本化 |
| 条件扫描与批量实验创建 | P1 | 从单个 run 转向系统性实验 | 复用 benchmark/conditions；资源预算和失败单元必须可见 |
| 硬件成本与性能取舍图 | P1 | 能选择具有实现价值的 DPD | 估算模型、主机计时和目标硬件证据严格标源 |
| 独立 EVM 交叉验证与多面板导出 | P1 | 数字和图可用于严肃比较及论文 | 需要可信参考波形、外部对照实现和审阅 |
| DTX 专项诊断与有条件的 2×1D LUT | P2 | 定位结构相关失真，检验轻量方案 | 需要码值扫描、结构说明与适用假设 |
| 单条真实仪器链适配、OTA/负载关联 | P2 | 缩短实验闭环，覆盖空间/负载条件 | 取决于可用硬件、校准与安全操作流程 |

### 6.1 P0：把结果页变成可直接讨论的 RF 实验页

首屏回答三个问题：**测了什么、相对基线改善多少、结论有哪些适用条件**。

1. 顶部固定一条精简事实栏：PA/DUT、载频、信号带宽、采样率、平均输出功率、证据来源、指标 profile。缺少字段用“未提供”，保留补录入口；哈希等细节放到展开层。
2. 保留 x→u→y 与无 DPD 基线，统一曲线名称、颜色、线型和数据来源。颜色表达方法，线型/标签表达仿真与实测，避免只靠颜色区分。
3. 比较时先选择一个参考结果，再添加候选。显示绝对值与相对参考差值；不可比结果可以并排查看，但不自动标注“最佳”。
4. 条件不匹配时给出具体差异，例如“输出功率不同”“指标参考不同”；跨条件趋势在另一种明确模式中查看。
5. 把频带积分区域、左右邻道、指标定义和数据来源直接关联到频谱。每个图例项目都可追到 run/capture/stage。

交互布局建议是：上方条件与指标摘要，下方以现有图表为主的 2×2 工作区，右侧可收起的条件/方法说明。在 1366×768 下应仍能读出单位、主要指标和比较结论；不要求一次铺开所有字段。

### 6.2 P1：Measurement Session——将实测组织成完整实验

现有实测结构最多关联一份 with-DPD 和一份 without-DPD 采集。建议新增可包含多次采集的 session，同时继续读取旧结果。

| 数据对象 | 建议新增/结构化内容 | 保留的语义 |
|---|---|---|
| `MeasurementSession` | DUT、仪器链、参考面、载频、供电/偏置、工作模式、负载、校准记录、时间与人员 | 用户声明与自动采集字段分别标源；仪器导入不等于独立认证 |
| `Capture` | 独立采集 ID、时间、角色、播放文件 hash、校准引用、原始单位、增益/衰减、声明或测量的功率 | 原始文件不可被对齐或归一化覆盖；保留现有播放追溯 |
| `ProcessingRecord` | 重采样、整数/分数延迟、相位/增益拟合、裁剪、loop/single、有效样本区间和版本 | 处理方法属于结果协议，不能作为无记录的显示修饰 |
| `RepeatSummary` | 独立采集次数、统计量、漂移、异常处理理由、可选的不确定度预算引用 | 训练 seed 与硬件 capture 分开统计；失败和排除样本可追溯 |

先交付采集分组、链路信息和重复统计，再做完整的不确定度传播。只有一次采集时不显示“重复性”；多个分段来自同一采集时不能自动算作独立重复。误差条必须注明是标准差、置信区间还是给定不确定度，不能统一写成“± error”。这一分层直接对应 R14、R15 对测量误差来源的讨论。[R14](https://doi.org/10.1109/MMM.2024.3524849)、[R15](https://doi.org/10.1109/ARFTG57476.2023.10279048)

with/without-DPD 比较增加明确的功率匹配说明：可以在事先声明的输出功率容差内评估改进，或在功率扫描中比较达标工作点。容差由实验协议设定。不能让自动幅度归一化掩盖“减小输出功率后指标改善”。

分数延迟优先复用现有预处理能力，但实测流程需要独立验证其相位响应、边界与有效区间，发布新的处理协议版本；不能直接改变旧结果的对齐数值。

### 6.3 P1：Sweep Board——把已有后端变成可操作的成组实验

建议在现有实验与 Robustness 页面上增加两个入口：

- **同条件方法比较：**固定数据、划分、指标、PA 代理模型和工作点，扫描模型配置、参数预算、精度或稀疏阈值，汇总 seeds。
- **跨条件泛化/适应：**以独立捕获的数据建立 source/target 卡，比较冻结模型、限量适应和全量重训，保留全部条件与失败单元。

第一个版本只支持一个明确物理维度加一组方法，避免在没有数据的情况下产生庞大矩阵。可选维度包括平均输出功率、载频、带宽、温度、供电、模式和负载；VSWR 必须同时保留反射系数相位，仅有 VSWR 数值不能描述完整复负载。

提交前显示实验矩阵、训练次数、数据是否齐备和资源预算；允许取消、续跑失败项和从现有配置复制。训练公平性需记录参数数目之外的训练预算、数据量和模型选择规则；目标条件不能用于暗中选超参数。R06 的多种子比较及 R12 的冻结 DPD 负载实验为这两个入口提供直接依据。[R06](https://arxiv.org/abs/2401.08318)、[R12](https://doi.org/10.1109/TMTT.2025.3569155)

### 6.4 P1：Hardware Trade-offs——同时看线性度与实现代价

第一版复用当前流式与定点报告，按同一 RF 协议组织候选方案。推荐视图是 ACPR/EVM 对 MAC/sample 或存储量的散点图；只有存在有依据的功耗数据时，才开放能耗轴。

成本明细至少区分：

- 存储参数总数、每样本实际执行量、动态跳过比例；稀疏算法的“有效参数”不能代替需要存储的权重。
- 权重/激活/累加器位宽、特征提取、LUT/近似函数、状态和输入/前视缓冲；不能把 FP32 特征提取漏出“全定点”标签。
- 前视样本及其时间下界、算法 warm-up、主机执行计时、目标吞吐与端到端延迟；这些数值不能互相替代。
- 证据来源：操作计数估算、CPU 参考实现计时、FPGA 综合/时序、FPGA 板级测量、ASIC 综合、布局后仿真、芯片实测。精确工艺、频率、批量和活动率随记录保存。

R07、R09、R10、R11 分别展示精度、稀疏性、参数效率和硬件映射的不同取舍，最有价值的 feature 是让用户看懂这些代价如何计算，而不是给所有模型自动生成一个“功耗预测”。[MP-DPD](https://arxiv.org/abs/2404.15364)、[DeltaDPD](https://arxiv.org/abs/2505.06250)、[TCN-DPD](https://arxiv.org/abs/2506.12165)、[DPD-NeuralEngine](https://arxiv.org/abs/2410.11766)

### 6.5 P2：有明确数据前提的 DTX 诊断

在已有 AM/AM、AM/PM 和散点图上扩展：幅度/相位条件下的误差分布、I/Q 平面的误差热图及占用密度、残差随记忆滞后的关系，以及具备正交扫描时的 code-to-AM/PM。

热图须显示每个 bin 的样本数，空白区域不能插值成“已验证”；高误差但极少出现的区域与高概率误差应能区分。2×1D LUT 作为特定数字 I/Q 架构的实验基线，需要说明正交性、路径耦合和记忆效应假设，不能预期在任意 PA 上替代神经模型。[R04](https://doi.org/10.1109/IMS37964.2023.10187914)

对于神经训练，后续可以把已有模型的激活、精度、稀疏阈值和预算暴露成受控实验配方；若引入带内/邻道加权目标或硬件约束损失，应作为独立方法验证。不能在产品演示中预先承诺它们必然改善实测性能。

### 6.6 出图与复现：每张图都保存“如何得到”

分两阶段扩展现有报告导出：

- **P0：**保存所选结果、轨迹、坐标范围、参考项和频带；导出频谱图、对应数值 CSV 与视图说明。再次打开应复现同一视图。
- **P1：**支持 PSD、功率扫描、误差分布等多面板版式，SVG/PDF/PNG 输出，并附方法说明、数据/模型/配置 hash、profile、软件版本和生成脚本或可重放命令。

建议增加版本化 `figure_spec`，只引用已经保存的结果与图表数据，不生成另一套指标算法。论文模板可预设单栏/双栏宽度、字体、线宽和黑白可辨线型；图注自动填入事实，但未声明的调制格式、参考面或功耗条件不得补猜。

## 7. 页面上应该 show 的 RF 细节

这些字段不是全部塞进首屏。采用三层：**摘要用于判断、展开用于诊断、原始记录用于复核**。

| 类别 | 摘要与展开内容 | 必须避免的误读 |
|---|---|---|
| 波形与带宽 | 载频；基带 Fs；信号带宽定义；载波中心/间隔/占用；调制；输入和 DPD 输出 PAPR；捕获带宽与指标积分区间 | 射频载频、调制带宽、采样率、FFT 分辨率是不同量；不默认非连续载波可用单个连续频带概括 |
| 功率与效率 | 平均调制 Pout、输入功率、CW 饱和功率/回退参考；PA DC、其它电源轨；效率边界 | 归一化 IQ 不能直接得到 dBm；平均调制功率不能与 CW 峰值混排 |
| PSD / ACPR | 横纵轴单位、相对或绝对基准、窗口、分段长度、重叠率、积分频带、左/右邻道与较差一侧 | `Fs/NFFT` 不是对所有窗都成立的等效噪声带宽；负泄漏比与正抑制度不能同名混用 |
| EVM / NMSE | 参考是输入 IQ、理想符号还是特定频谱定义；同步、频偏、相位、均衡；线性/对数单位及聚合方式 | 原始 IQ 散点不等于解调星座；输入参考 EVM 不等于标准符号 EVM；分段 dB 均值不等于全量 pooled NMSE |
| DPD 与 PA 模型 | 模型/权重、训练域、参数与记忆范围、训练数据量、seed、PA 代理模型、x/u/y 和无 DPD 基线 | 训练幅度范围内不代表跨温度/负载有效；代理模型改善不自动等于硬件改善 |
| 实测与处理 | 原始 capture、链路/参考面、采样及功率标定、对齐/重采样、有效区间、循环边界、饱和或缺失样本 | 自动处理不能无记录地修饰测量结论；波形相似性高不等于功率一致 |
| 统计与误差 | seeds 与独立 captures 分栏；n、均值/中位数、标准差、可选 CI 及方法；异常排除、失败数 | 最佳 seed、种子均值和硬件重复均值不可混排；重复性不覆盖全部系统误差 |
| 实现 | 分模块位宽、存储、MAC/sample、LUT、状态、前视、吞吐、延迟和功耗证据 | 参数少不等于省电；CPU 测速、综合报告和芯片实测分别命名 |

效率字段应明确公式和边界：`DE = P_RF_out / P_DC_PA`；`PAE = (P_RF_out − P_RF_in) / P_DC_PA`；`TX efficiency = P_RF_out / ΣP_DC_included_rails`，功率计算使用线性单位。输入功率或功耗边界未提供时显示不可用。尤其避免把 RF 的 power-added efficiency 与部分加速器论文中 power-area efficiency 的缩写 PAE 混淆。[R02](https://doi.org/10.1109/JSSC.2021.3133861)、[R11](https://arxiv.org/abs/2410.11766)

## 8. 交互习惯：能够合理提出的假设及验证方式

论文能展示研究者如何构造实验、比较方法和呈现证据，不能证明其个人界面偏好。建议按下面的假设设计，再测量是否改善任务完成情况。

| 可验证的交互假设 | 设计方式 | 验证任务 |
|---|---|---|
| 研究者经常围绕固定参考反复比较 | 固定 reference、持久显示配置差异、保留坐标范围；一键复制后修改一个变量 | 找出一个候选结果是否真正优于指定基线，并指出条件差异 |
| 更容易在物理量中理解配置 | 表格可输入数值和单位；专家层保留 JSON/CLI 配方；支持 MATLAB/Python 数据交换 | 配置一次指定带宽/功率实验，导出并在脚本侧重放 |
| 图表需要承担定量读数工作 | 测量游标、参考轨迹、积分边界和同步频率轴；沿用现有平移/缩放行为 | 读取某偏移频率的泄漏，解释左右邻道差别 |
| 诊断需要原始与处理后对照 | 同一视图切换 raw/aligned/evaluated；明确每条曲线所属 stage | 找到由对齐或增益处理引起的变化，恢复原始观察 |
| 一次研究包含多次运行和失败 | 实验组、批量计划、状态矩阵、失败原因和种子分布 | 找出最差条件、定位失败项，并重跑指定单元 |
| 讨论和出图后仍需复现 | 保存视图、下载图表数据、完整来源和可重放说明 | 由另一人从导出包重建同一图与指标 |

不建议依据教授身份默认添加更多控制面板。默认界面保持简洁，必要细节按任务展开；也不预设他们愿意上传未发表测量到公共服务。已有本地/私有工作区应能完成完整审阅；公共演示保留自身资源限制。

## 9. 8 周实施路线与第一轮可直接开工的事项

工作量假设：1 名前端工程师、1 名 RF/后端工程师，以及能定期审阅指标定义和提供测量的研究人员。以下是排期建议，不是已经验证的工期承诺；真实仪器与独立交叉验证的可用性是外部依赖。

| 时间 | 交付 | 完成标准 |
|---|---|---|
| 第 1–2 周 | RF 事实栏、比较工作台 v1、参考轨迹、频带说明、保存视图与频谱导出 | 用现有真实/仿真结果完成一次“识别证据—公平比较—导出复核”；不改变原指标数值 |
| 第 3–4 周 | Measurement Session v1、结构化校准/参考面、重复采集；条件扫描的创建界面 | 一组真实重复采集可追溯；既有 condition card 可从 UI 创建/校验；失败单元保留 |
| 第 5–6 周 | 方法/条件矩阵、硬件成本取舍图、实测分数延迟验证；EVM 外部对照 | 至少一组固定模型跨条件实验；成本项可追溯。EVM 仅在交叉验证通过后向 GUI 开放 |
| 第 7–8 周 | 多面板出版导出、四类研究任务试用、修正信息层次与交互 | 另一位研究者能重建关键图和指标；完成真实用户任务测试，整理后续 P2 决策 |

### 首个两周的六个工作项

| 工作项 | 实现切入点 | 本轮验收 |
|---|---|---|
| W1：RF 事实栏与定义说明 | ResultDetailPage、现有 result/measurement 数据；缺失字段先清楚显示 | 读者不打开 JSON 即可识别数据来源、Fs/BW、声明的功率及指标定义 |
| W2：固定参考的比较 | ComparePage、现有 comparison API | 保留当前不可比保护；可查看绝对值、差值与具体条件差异 |
| W3：频谱读数与频带叠加 | 复用图表控制器与 profile 元数据 | 游标显示单位和轨迹来源；积分边界来自协议；图表交互不改指标 |
| W4：保存审阅视图 | 版本化 figure/view spec，引用 result ID 与轨迹 | 重开后参考项、坐标、可见轨迹和频带一致；不复制训练或评估逻辑 |
| W5：图表与数据导出 v1 | reports/packages 服务 | 导出已有频谱、图表数据、事实栏、profile 与来源说明；缺失条件明确 |
| W6：四个角色任务脚本与演示包 | 现有已注册数据和结果 | 每个任务在约 5 分钟内可演示；材料区分 measured、surrogate、mock 和成本估算 |

建议实施落点是扩展既有页面、schema 和服务；不先增加一套平行的“科研模式”数据结构。新增 RF 条件使用可选字段和显式版本迁移读取历史结果；原始结果、profile 和预处理协议保持可追溯。Measurement Session 与 Experiment Group 引用现有 capture/run/condition 对象，避免复制数据形成不一致。

## 10. 验收、演示和需要准备的数据

### 10.1 科学与工程验收

这里列的是后续实现应执行的检查，不表示本次 Markdown 任务已经运行了这些实验。

1. **指标一致：**同一 result 在结果页、比较页、CSV 和报告中的数值、单位、profile 一致；平移、缩放、抽样只改变显示。
2. **频带正确：**使用已知功率的合成信号验证积分和边界；邻道超出捕获范围时给出不可用原因，不补算不存在的数据。
3. **对齐可解释：**以已知整数/分数延迟、相位和增益构造可控样例，再用真实 capture 核对；有效区间与边界处理随版本保存。
4. **统计不混用：**单采集不产生重复性数字；seed 与 capture 独立计数；区间注明方法；最佳结果与均值使用不同标签。
5. **证据可区分：**不将 TCN 代理仿真、MP/Delta 成本估算或 ASIC 布局后仿真标为芯片实测；mock 数据不进入真实测量排名。
6. **运行可复现：**检查数据、权重、配置、profile、预处理版本与完整 hash；导出包能够重建指标和视图。显示可用短 hash，记录必须保留完整值。
7. **布局与操作：**在 1366×768 和 1920×1080 完成核心任务；现有键盘及图表手势继续可用；关联图只联动相容轴，不强行将频率点当作时域点。
8. **性能：**选取明确规模的数据记录基线，比较加载和操作耗时；优先复用缓存与显示抽样。数据已加载后的参考切换可将约 1 秒内反馈设为设计目标，实测后再定承诺。

### 10.2 四段各约五分钟的演示

| 面向的研究任务 | 演示顺序 | 应得到的判断 |
|---|---|---|
| de Vreede：系统取舍 | 选工作点 → 比较线性度门限 → 查看功率/效率与实现成本 → 打开最有价值的候选 | 看清性能收益与系统代价是否同时成立 |
| Alavi：结构相关失真 | 查看载波和模式 → 检查 AM/PM、I/Q 误差 → 比较简单基线与神经模型 → 关联实测 | 知道改善发生在哪里，是否需要复杂模型 |
| Babaie：时序与校准 | 固定参考 → 逐项查看校准消融 → 定位延迟/相位问题 → 检查流式/定点成本 | 分开解释误差来源和实时实现限制 |
| Spirito：测量可信度 | 查看链路与原始采集 → 审查处理 → 叠加重复测量 → 导出可复核图 | 能判断改善是否受到仪器、漂移或处理方法限制 |

演示顺序体现研究任务，不声称是四位本人的操作习惯。现有数据没有 DC 功耗、真实温度扫描、负载重复测量或 I/Q 独立扫描时，对应面板显示数据缺失；合成示例仅用于演示交互，清楚标注。

### 10.3 最小数据准备与用户验证

应优先准备：一组有明确波形/指标协议的真实 with/without-DPD 数据、一组同条件独立重复采集、一组至少三个独立工作条件的数据，以及一份能与已有定点导出关联的成本报告。后两项分别支撑条件泛化和硬件取舍，缺失时不能用一个 capture 的不同切片替代独立实验。

邀请约 6–8 名覆盖上述四种研究任务的实际使用者，至少包含负责采集和日常训练的博士生或工程师。若四位本人能够参与，可用这些具体任务获得反馈；本报告并未联系任何人。

记录三类结果：能否正确识别证据和指标定义、能否完成公平比较、他人能否复现导出结果。建议目标是核心任务独立完成率至少 80%，操作时间相对当前界面下降约 20%；这只是待验证的目标，测量时应把操作时间与训练/计算等待时间分开。个人手势、界面语言与信息密度偏好在试用中询问，不从论文推断。

## 11. 下一次评审应做的决策

首个两周评审只需要判断：新的结果与比较流程，是否让研究者更快、更准确地解释一项 DPD 结果。随后根据真实数据决定优先投入重复测量、条件泛化还是硬件代价。

成功标准应落在这些具体问题上：**同一个工作条件下是否更好，改善来自哪里，测量能否支持它，实现要付出什么，以及另一个人能否复现。** 这些问题可作为后续功能评审的共同依据。
