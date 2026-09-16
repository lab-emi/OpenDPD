"""Illustrative complex-envelope virtual PAs; no device fitting or circuit claims.

Technology names are application examples. The equations and their normalized
parameters define the simulation; they are not foundry or transistor models.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.signal import welch

from opendpd.schemas.virtual_pa import PALocalizedText, PAParameter, VirtualPAModel, PAAnalysis


def text(en, zh):
    return PALocalizedText(en=en, zh=zh)


def parameter(key, symbol, en, zh, description, description_zh, default, lo, hi, step,
              unit="", group="gain", logarithmic=False, integer=False):
    return PAParameter(key=key, symbol=symbol, label=text(en, zh),
        description=text(description, description_zh), default=default, minimum=lo,
        maximum=hi, step=step, unit=unit, group=group, logarithmic=logarithmic, integer=integer)


GAIN = parameter("gain", "G", "Small-signal gain", "小信号增益",
    "Linear voltage-envelope gain before compression.", "进入压缩前的线性电压包络增益。", 2, .1, 8, .05, "V/V")
SAT = parameter("saturation", "s", "Saturation envelope", "饱和包络",
    "Output-envelope ceiling of the soft limiter; lowering it increases compression.",
    "软限幅的输出包络上限；减小它会增强压缩。", .7, .05, 3, .01, "normalized")
SHAPE = parameter("smoothness", "p", "Compression knee", "压缩拐点锐度",
    "Higher values produce a sharper transition into saturation.",
    "值越大，从线性区进入饱和区的转折越尖锐。", 2, .5, 8, .1)
PHASE = parameter("phase", "φ∞", "AM/PM rotation", "AM/PM 相移",
    "Asymptotic phase rotation at high envelope amplitude.",
    "大信号包络下趋近的相位旋转量。", .25, -1.5, 1.5, .01, "rad")
PHASE_SCALE = parameter("phase_scale", "b", "AM/PM onset", "AM/PM 起始幅度",
    "Input amplitude at half of the asymptotic phase rotation.",
    "相位旋转达到渐近值一半时的输入幅度。", .35, .03, 2, .01, "normalized")
STATIC = [GAIN, SAT, SHAPE]
RAPP = "R(x; G,s,p) = G·x / [1 + (G·|x|/s)^(2p)]^(1/(2p))"
RAPP_BOUND = "R(x) = {{gain}}·x / [1 + ({{gain}}·|x|/{{saturation}})^(2·{{smoothness}})]^(1/(2·{{smoothness}}))"
LIMIT = text("Illustrative complex-baseband behavior, with normalized amplitudes. Defaults are not extracted from a specific device; no calibrated RF power, efficiency or transistor-level prediction.",
             "归一化幅度的复基带示意模型。默认参数未从具体器件提取，不代表校准射频功率、效率或晶体管级预测。")
MEMORY_REF = "https://www.keysight.com/us/en/assets/7018-02538/application-notes/5990-5799.pdf"
GMP_REF = "https://doi.org/10.1109/TSP.2006.879264"
TRAP_REF = "https://ieeexplore.ieee.org/document/9658162/"
ET_REF = "https://helpfiles.keysight.com/csg/n7614/Content/Main/Envelope%20Tracking%20Concept.htm"


def catalog():
    phase_offset = parameter("phase_deg", "θ", "Phase offset", "固定相移",
        "Constant phase rotation; it does not depend on input amplitude.",
        "固定相位旋转，不依赖输入幅度。", 0, -180, 180, 1, "°")
    memory = [
        parameter("cubic", "c₃", "Cubic nonlinearity", "三阶非线性",
            "Negative values compress moderate input levels; positive values expand them.",
            "负值压缩中等输入幅度，正值产生增益扩张。", -1.2, -5, 2, .05),
        parameter("quintic", "c₅", "Quintic curvature", "五阶曲率",
            "Changes high-amplitude curvature. Polynomial models are local approximations.",
            "改变大信号曲率；多项式仅是工作区间内的局部近似。", .4, -2, 2, .05),
        parameter("memory", "μ", "Electrical memory strength", "电气记忆强度",
            "Total magnitude of delayed nonlinear taps relative to the instantaneous path.",
            "延迟非线性抽头相对于瞬时通路的总幅度。", .18, 0, .8, .01, group="memory"),
        parameter("depth", "M", "Memory depth", "记忆深度",
            "Number of causal sample delays. Physical span is M / sample rate.",
            "因果采样延迟的数量；对应物理时长为 M / 采样率。", 5, 1, 24, 1, "samples", "memory", integer=True),
        parameter("decay", "d", "Tap decay", "抽头衰减",
            "Larger values spread the same total memory strength over more delayed samples.",
            "值越大，同样的总记忆强度会分配到更多延迟样本。", .65, .05, 1, .01, group="memory"),
        parameter("memory_phase", "ψ", "Tap phase progression", "抽头相位递进",
            "Phase increment per delayed tap; creates dispersive memory and asymmetric spectra.",
            "每个延迟抽头的相位增量，可产生色散记忆与非对称频谱。", 20, -90, 90, 1, "°", "memory"),
    ]
    mp_equations = [
        "u[n] = {{gain}}·x[n] + {{cubic}}·x[n]|x[n]|² + {{quintic}}·x[n]|x[n]|⁴",
        "wₘ = {{decay}}^(m−1)·exp(j·m·{{memory_phase}}·π/180) / Σₖ₌₁^{{depth}} {{decay}}^(k−1)",
        "z[n] = u[n] + {{memory}}·Σₘ₌₁^{{depth}} wₘ·u[n−m]",
    ]
    models = [
        VirtualPAModel(model_id="linear-reference", category="reference", name=text("Linear reference", "线性参考 PA"),
            technology="Ideal / calibration", description=text(
                "An ideal gain and phase stage for checking scaling, alignment and the training pipeline.",
                "理想增益与相位通路，适合检查缩放、对齐和训练流程。"), limitations=LIMIT,
            parameters=[GAIN, phase_offset], equations=["y[n] = {{gain}}·x[n]·exp(j·{{phase_deg}}·π/180)"], references=[]),
        VirtualPAModel(model_id="rapp-solid-state", category="static", name=text("Solid-state soft limiter", "固态软饱和 PA"),
            technology="CMOS / SiGe / GaAs", description=text(
                "Rapp AM/AM saturation: a compact radio or array-element PA with adjustable compression knee and no phase distortion.",
                "Rapp AM/AM 饱和模型：面向紧凑无线电或阵列单元 PA，可调整压缩拐点，不含相位失真。"),
            limitations=LIMIT, parameters=STATIC, equations=[RAPP_BOUND, "y[n] = R(x[n])"], references=[MEMORY_REF]),
        VirtualPAModel(model_id="rapp-am-pm", category="static", name=text("Solid-state AM/AM + AM/PM", "固态幅相非线性 PA"),
            technology="GaAs HBT / CMOS", description=text(
                "Soft saturation plus saturating amplitude-dependent phase rotation, useful for handset and WLAN PA experiments.",
                "软饱和叠加随幅度变化并趋于饱和的相移，适合手机与 WLAN PA 仿真实验。"),
            limitations=LIMIT, parameters=[*STATIC, PHASE, PHASE_SCALE],
            equations=[RAPP_BOUND, "Φ(r) = {{phase}}·r² / (r² + {{phase_scale}}²)", "y[n] = R(x[n])·exp(j·Φ(|x[n]|))"],
            references=[MEMORY_REF]),
        VirtualPAModel(model_id="saleh-twta", category="static", name=text("Satellite traveling-wave tube", "卫星行波管 PA"),
            technology="TWTA / vacuum electronics", description=text(
                "Saleh amplitude and phase laws for a satellite transponder-style nonlinear channel. Overdrive can produce gain roll-off.",
                "用于卫星转发器类非线性通道的 Saleh 幅相模型；过驱动时可出现输出回落。"), limitations=LIMIT,
            parameters=[GAIN, parameter("compression", "βₐ", "Amplitude compression", "幅度压缩",
                "Sets the turnover amplitude at 1 / sqrt(βₐ).", "输出峰值对应的输入幅度为 1 / √βₐ。", 1.5, .05, 8, .05),
                parameter("phase", "αφ", "Phase strength", "相移强度",
                    "Scales the amplitude-dependent phase response.", "缩放随输入幅度变化的相位响应。", .8, -3, 3, .05, "rad"),
                parameter("phase_scale", "βφ", "Phase saturation", "相移饱和系数",
                    "Limits high-level phase rotation to αφ / βφ.", "将大信号相移限制在 αφ / βφ。", 2, .1, 10, .1)],
            equations=["y[n] = A(r)·exp(j·(arg(x[n]) + Φ(r))),  r = |x[n]|",
                "A(r) = {{gain}}·r / (1 + {{compression}}·r²)",
                "Φ(r) = {{phase}}·r² / (1 + {{phase_scale}}·r²)"],
            references=["https://doi.org/10.1109/TCOM.1981.1094911"]),
        VirtualPAModel(model_id="memory-polynomial", category="memory", name=text("Wideband electrical memory", "宽带电气记忆 PA"),
            technology="LDMOS / GaN class AB", description=text(
                "A separable fifth-order memory polynomial for matching-network and bias-network memory in a wideband base-station PA.",
                "可分离的五阶记忆多项式，示意宽带基站 PA 中匹配网络和偏置网络的电气记忆。"),
            limitations=LIMIT, parameters=[GAIN, *memory], equations=[*mp_equations, "y[n] = z[n]"], references=[GMP_REF, MEMORY_REF]),
        VirtualPAModel(model_id="generalized-memory", category="memory", name=text("Lagging-envelope cross memory", "滞后包络交叉记忆 PA"),
            technology="GaN / LDMOS wideband", description=text(
                "Adds a causal lagging-envelope cross term to the memory polynomial, for interaction between today's carrier and past envelope power.",
                "在记忆多项式中加入因果滞后包络交叉项，模拟当前载波与历史包络功率的相互作用。"), limitations=LIMIT,
            parameters=[GAIN, *memory, parameter("cross_memory", "c×", "Envelope cross-memory", "包络交叉记忆",
                "Couples the current input to delayed envelope power; this is the lagging subset of a GMP.",
                "将当前输入与延迟包络功率耦合；这里实现 GMP 的滞后包络子集。", -.6, -3, 3, .05, group="memory")],
            equations=[*mp_equations, "q[n] = Σₘ₌₁^{{depth}} ({{decay}}^(m−1)/Σₖ₌₁^{{depth}} {{decay}}^(k−1))·|x[n−m]|²",
                "y[n] = z[n] + {{cross_memory}}·x[n]·q[n]"], references=[GMP_REF]),
    ]
    dynamics = [
        parameter("trap_strength", "κt", "Trapping / current collapse", "俘获 / 电流崩塌强度",
            "Occupied trap state reduces envelope gain by 1 − κt·qₜ.",
            "已占据的俘获状态通过 1 − κt·qₜ 降低包络增益。", .28, 0, .8, .01, group="dynamics"),
        parameter("capture_us", "τc", "Trap capture time", "俘获时间",
            "Charging time when envelope excitation exceeds the current trap state.",
            "包络激励超过当前俘获状态时的充电时间。", .3, .01, 100, .01, "µs", "dynamics", True),
        parameter("release_us", "τe,25", "Trap release time at 25 °C", "25 °C 下的释放时间",
            "Recovery time at 25 °C. Heating changes it through the activation-energy term.",
            "25 °C 下的恢复时间；温升通过激活能项改变该时间。", 20, .01, 10000, .01, "µs", "dynamics", True),
        parameter("activation_ev", "Ea", "Effective trap activation energy", "有效俘获激活能",
            "Controls how strongly temperature accelerates trap emission; illustrative effective energy.",
            "控制温度对释放速度的影响，属于示意性的有效激活能。", .18, 0, .6, .01, "eV", "dynamics"),
        parameter("trap_phase", "φt", "Trap-induced phase shift", "俘获引起的相移",
            "Phase rotation per unit of occupied trap state.", "每单位俘获占据状态产生的相移。", .2, -1, 1, .01, "rad", "dynamics"),
        parameter("thermal_us", "τT", "Thermal time constant", "热时间常数",
            "Response time of the effective single-pole temperature state.",
            "等效单极点温度状态的响应时间。", 50, .1, 100000, .1, "µs", "dynamics", True),
        parameter("ambient_c", "Ta", "Ambient temperature", "环境温度",
            "Reference ambient temperature; affects gain and trap emission in this virtual PA.",
            "参考环境温度，影响虚拟 PA 的增益与俘获释放。", 25, -20, 100, 1, "°C", "dynamics"),
        parameter("heating_c", "ΔT", "Effective self-heating rise", "等效自热温升",
            "Temperature rise at unit normalized envelope excitation, not a calibrated junction-temperature prediction.",
            "单位归一化包络激励下的温升，不是经过校准的结温预测。", 35, 0, 100, 1, "K", "dynamics"),
        parameter("thermal_gain", "κT", "Thermal gain sensitivity", "温度增益敏感度",
            "Exponential gain reduction per kelvin relative to 25 °C.",
            "相对于 25 °C，每开尔文温升导致的指数增益下降。", .004, 0, .025, .0005, "1/K", "dynamics"),
        parameter("ir_drop", "ρ", "Normalized IR drop", "归一化 IR 压降",
            "Envelope-dependent supply sag; ρ = 0 removes this bias-memory contribution.",
            "随包络变化的供电压降；ρ = 0 会关闭这部分偏置记忆。", .12, 0, .6, .01, group="dynamics"),
        parameter("bias_us", "τb", "Bias-network recovery", "偏置网络恢复时间",
            "Response time of the supply-droop envelope state.", "供电压降包络状态的响应时间。", 1, .01, 1000, .01, "µs", "dynamics", True),
    ]
    models.append(VirtualPAModel(model_id="gan-trap-thermal", category="dynamics",
        name=text("GaN trapping, heat & supply memory", "GaN 俘获、热与供电记忆"),
        technology="GaN HEMT / GaN-on-SiC", description=text(
            "A physics-inspired envelope model for pulsed or TDD base-station/radar studies: trapping, temperature-dependent recovery, self-heating and supply sag.",
            "面向脉冲、TDD 基站或雷达研究的物理启发式包络模型：俘获、随温度变化的恢复、自热及供电压降。"),
        limitations=LIMIT, parameters=[*STATIC, *dynamics], equations=[
            RAPP_BOUND, "P[n] = |x[n]|² / (|x[n]|² + ({{saturation}}/{{gain}})²)",
            "T[n] = {{ambient_c}} + {{heating_c}}·LPF(P, {{thermal_us}})",
            "τe[n] = {{release_us}}·exp({{activation_ev}}/kB · (1/(T[n]+273.15) − 1/298.15))",
            "τ[n] = {{capture_us}} if P[n] ≥ qₜ[n−1], otherwise τe[n]",
            "qₜ[n] = a[n]·qₜ[n−1] + (1−a[n])·P[n],  a[n] = exp(−1/(Fs·τ[n]·10⁻⁶))",
            "v[n] = 1 − {{ir_drop}}·LPF(P, {{bias_us}})",
            "y[n] = R(x[n])·(1−{{trap_strength}}·qₜ[n])·v[n]·exp(−{{thermal_gain}}·(T[n]−25) + j·{{trap_phase}}·qₜ[n])",
            "LPF(P,τ)[n] = a·LPF[n−1] + (1−a)·P[n]; a = exp(−1/(Fs·τ·10⁻⁶)); τ in µs; kB = 8.617333262×10⁻⁵ eV/K",
        ], references=[TRAP_REF, "https://ieeexplore.ieee.org/document/9143297/", MEMORY_REF]))
    models.append(VirtualPAModel(model_id="doherty-two-path", category="architecture",
        name=text("Doherty-inspired two-path PA", "Doherty 启发式双通路 PA"),
        technology="GaN / LDMOS Doherty", description=text(
            "A carrier path plus a delayed-turn-on peaking path for studying the AM/AM knee and path mismatch in infrastructure PAs.",
            "主放大通路叠加延迟开启的峰值通路，用于研究基站 PA 的 AM/AM 拐点和通路失配。"),
        limitations=text(LIMIT.en + " This envelope construction does not solve an impedance inverter or predict load modulation/efficiency.",
                        LIMIT.zh + " 该包络构造不求解阻抗变换网络，也不预测真实负载调制或效率。"),
        parameters=[*STATIC,
            parameter("knee", "rₖ", "Peaking turn-on level", "峰值通路开启幅度",
                "Only the input envelope above this threshold excites the peaking path.",
                "仅超过此阈值的输入包络会激励峰值通路。", .2, 0, 1, .01, "normalized", "architecture"),
            parameter("peaker", "βp", "Peaking-path strength", "峰值通路强度",
                "Relative peaking contribution; zero leaves the carrier path alone.",
                "峰值通路的相对贡献；设为零仅保留主通路。", .75, 0, 2, .05, group="architecture"),
            phase_offset.model_copy(update={"group": "architecture",
                "description": text("Phase mismatch between carrier and peaking paths.", "主通路与峰值通路之间的相位失配。")})],
        equations=[RAPP_BOUND, "xₚ[n] = max(|x[n]|−{{knee}}, 0)·exp(j·arg(x[n]))",
            "y[n] = R(x[n]) + {{peaker}}·R(xₚ[n])·exp(j·{{phase_deg}}·π/180)"], references=[MEMORY_REF]))
    models.append(VirtualPAModel(model_id="envelope-tracking", category="architecture",
        name=text("Envelope-tracking supply PA", "包络跟踪供电 PA"),
        technology="CMOS / GaAs handset ET", description=text(
            "A time-varying saturation ceiling driven by a finite-speed supply tracker, with bias recovery and normalized IR drop.",
            "有限速度的供电跟踪器控制时变饱和上限，同时包含偏置恢复和归一化 IR 压降。"),
        limitations=LIMIT, parameters=[*STATIC,
            parameter("supply_floor", "vmin", "Minimum supply fraction", "最低供电比例",
                "Normalized floor on available saturation headroom.",
                "可用饱和余量的归一化下限。", .4, .1, .9, .01, group="architecture"),
            parameter("supply_span", "hv", "Tracking voltage swing", "跟踪电压摆幅",
                "Added supply fraction at full envelope demand.", "最大包络需求下增加的供电比例。", .8, .1, 1.5, .01, group="architecture"),
            parameter("tracking_us", "τv", "Supply-tracker response", "供电跟踪响应时间",
                "Slower tracking creates dynamic clipping on fast envelope peaks.",
                "跟踪越慢，快速包络峰值越容易出现动态削顶。", .05, .001, 100, .001, "µs", "architecture", True),
            dynamics[-2].model_copy(update={"group": "architecture"}),
            dynamics[-1].model_copy(update={"group": "architecture"}),
            PHASE.model_copy(update={"group": "architecture", "description": text(
                "Phase sensitivity to supply sag, Φ = φ∞·(v−1).", "供电压降引起的相位敏感度，Φ = φ∞·(v−1)。")})],
        equations=[RAPP, "D[n] = min({{gain}}·|x[n]|/{{saturation}}, 1)",
            "P[n] = |x[n]|² / (|x[n]|² + ({{saturation}}/{{gain}})²)",
            "v[n] = clip({{supply_floor}} + {{supply_span}}·LPF(D,{{tracking_us}}) − {{ir_drop}}·LPF(P,{{bias_us}}), {{supply_floor}}, 1.5)",
            "y[n] = R(x[n]; {{gain}}, {{saturation}}·v[n], {{smoothness}})·exp(j·{{phase}}·(v[n]−1))",
            "LPF(P,τ)[n] = a·LPF[n−1] + (1−a)·P[n]; a = exp(−1/(Fs·τ·10⁻⁶)); τ in µs"],
        references=[ET_REF, MEMORY_REF]))
    from .pa_equations import attach_latex
    return attach_latex(models)


def resolve(model_id, supplied):
    model = next((item for item in catalog() if item.model_id == model_id), None)
    if model is None:
        raise ValueError("Unknown virtual PA. Choose a model from PA Library.")
    expected = {p.key for p in model.parameters}
    if set(supplied) - expected:
        raise ValueError("Unknown PA parameters: " + ", ".join(sorted(set(supplied) - expected)))
    values = {}
    for p in model.parameters:
        value = supplied.get(p.key, p.default)
        if isinstance(value, bool) or not math.isfinite(value) or not p.minimum <= value <= p.maximum:
            raise ValueError(f"{p.label.en} must be between {p.minimum} and {p.maximum} {p.unit}.")
        if p.integer and value != int(value):
            raise ValueError(f"{p.label.en} must be an integer.")
        values[p.key] = float(value)
    return model, values


def simulate(x, fs, model_id, parameters):
    from opendpd.core.virtual_pa_kernel import simulate_resolved
    _, values = resolve(model_id, parameters)
    return simulate_resolved(x, fs, model_id, values)


def analyze(x, y, fs, states, parameters):
    x, y = np.asarray(x), np.asarray(y)
    indices = np.linspace(0, len(x)-1, min(len(x), 1536), dtype=int)
    mask = np.abs(x[indices]) > max(np.max(np.abs(x))*1e-6, 1e-12)
    am_indices = indices[mask]
    nfft = min(2048, len(x))
    f, px = welch(x, fs=fs, nperseg=nfft, noverlap=nfft//2, return_onesided=False, detrend=False)
    _, py = welch(y, fs=fs, nperseg=nfft, noverlap=nfft//2, return_onesided=False, detrend=False)
    power_x, power_y = float(np.mean(np.abs(x)**2)), float(np.mean(np.abs(y)**2))
    def papr(z, power):
        return float(10*np.log10(np.max(np.abs(z)**2)/power)) if power > 0 else None
    notes = ["Synthetic PA output, computed from the displayed equations; no measured PA output is present.",
             "Input and output share the exact sample clock and sample count. Amplitudes are normalized; RF carrier is metadata.",
             "Memory states start at zero (cold start). Effective temperature starts at ambient. Plots use at most 1,536 time samples; metrics use every sample.",
             "The continuous capture carries state across future train/validation/test boundaries; guard samples are not a claim of independent acquisitions."]
    duration_us = len(x)/fs*1e6
    for name in ("thermal_us", "release_us", "bias_us", "tracking_us"):
        if parameters.get(name, 0) > duration_us:
            notes.append(f"Capture duration {duration_us:.3g} µs is shorter than {name}={parameters[name]:.3g} µs; increase signal duration to observe settling.")
    return PAAnalysis(n_samples=len(x), duration_ms=len(x)/fs*1000,
        input_rms=math.sqrt(power_x), output_rms=math.sqrt(power_y),
        rms_gain_db=float(10*np.log10(power_y/power_x)) if min(power_x, power_y)>0 else None,
        input_papr_db=papr(x, power_x), output_papr_db=papr(y, power_y),
        time_us=(indices/fs*1e6).tolist(), input_envelope=np.abs(x[indices]).tolist(),
        output_envelope=np.abs(y[indices]).tolist(), am_input=np.abs(x[am_indices]).tolist(),
        am_output=np.abs(y[am_indices]).tolist(), am_pm_deg=np.rad2deg(np.angle(y[am_indices]*np.conj(x[am_indices]))).tolist(),
        frequency_mhz=(np.fft.fftshift(f)/1e6).tolist(),
        input_psd=(10*np.log10(np.maximum(np.fft.fftshift(px), 1e-30))).tolist(),
        output_psd=(10*np.log10(np.maximum(np.fft.fftshift(py), 1e-30))).tolist(),
        states={name: value[indices].tolist() for name, value in states.items()}, notes=notes)
