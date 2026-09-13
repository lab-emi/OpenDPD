"""Display-only LaTeX for the fixed virtual-PA catalog (never evaluated)."""
SYMBOLS = {
    'gain': 'G', 'saturation': 's', 'smoothness': 'p', 'phase': r'\phi_\infty',
    'phase_scale': 'b', 'phase_deg': r'\theta', 'compression': r'\beta_a',
    'cubic': 'c_3', 'quintic': 'c_5', 'memory': r'\mu', 'depth': 'M',
    'decay': 'd', 'memory_phase': r'\psi', 'cross_memory': r'c_\times',
    'trap_strength': r'\kappa_t', 'capture_us': r'\tau_c', 'release_us': r'\tau_{e,25}',
    'activation_ev': 'E_a', 'trap_phase': r'\phi_t', 'thermal_us': r'\tau_T',
    'ambient_c': 'T_a', 'heating_c': r'\Delta T', 'thermal_gain': r'\kappa_T',
    'ir_drop': r'\rho', 'bias_us': r'\tau_b', 'knee': 'r_k', 'peaker': r'\beta_p',
    'supply_floor': r'v_{\min}', 'supply_span': 'h_v', 'tracking_us': r'\tau_v',
}
RAPP = r'R(x;G,s,p)=\frac{Gx}{[1+(G|x|/s)^{2p}]^{1/(2p)}}'
BOUND = r'R(x)=\frac{{{gain}}x}{[1+({{gain}}|x|/{{saturation}})^{2{{smoothness}}}]^{1/(2{{smoothness}})}}'
MP = [
    r'u[n]={{gain}}x[n]+{{cubic}}x[n]|x[n]|^2+{{quintic}}x[n]|x[n]|^4',
    r'w_m=\frac{{{decay}}^{m-1}e^{jm{{memory_phase}}\pi/180}}{\sum_{k=1}^{{{depth}}}{{decay}}^{k-1}}',
    r'z[n]=u[n]+{{memory}}\sum_{m=1}^{{{depth}}} w_m u[n-m]',
]
LPF = r'\operatorname{LPF}(P,\tau)[n]=a\operatorname{LPF}[n-1]+(1-a)P[n],\quad a=e^{-1/(F_s\tau\,10^{-6})},\quad\tau\text{ in }\mu\mathrm{s}'
EQUATIONS = {
    'linear-reference': [r'y[n]={{gain}}x[n]e^{j{{phase_deg}}\pi/180}'],
    'rapp-solid-state': [BOUND, r'y[n]=R(x[n])'],
    'rapp-am-pm': [BOUND, r'\Phi(r)=\frac{{{phase}}r^2}{r^2+{{phase_scale}}^2}', r'y[n]=R(x[n])e^{j\Phi(|x[n]|)}'],
    'saleh-twta': [r'y[n]=A(r)e^{j(\arg x[n]+\Phi(r))},\quad r=|x[n]|',
        r'A(r)=\frac{{{gain}}r}{1+{{compression}}r^2}', r'\Phi(r)=\frac{{{phase}}r^2}{1+{{phase_scale}}r^2}'],
    'memory-polynomial': [*MP, 'y[n]=z[n]'],
    'generalized-memory': [*MP, r'q[n]=\sum_{m=1}^{{{depth}}}\frac{{{decay}}^{m-1}}{\sum_{k=1}^{{{depth}}}{{decay}}^{k-1}}|x[n-m]|^2',
        r'y[n]=z[n]+{{cross_memory}}x[n]q[n]'],
    'gan-trap-thermal': [BOUND,
        r'P[n]=\frac{|x[n]|^2}{|x[n]|^2+({{saturation}}/{{gain}})^2}',
        r'T[n]={{ambient_c}}+{{heating_c}}\operatorname{LPF}(P,{{thermal_us}})',
        r'\tau_e[n]={{release_us}}\exp\!\left[\frac{{{activation_ev}}}{k_B}\left(\frac{1}{T[n]+273.15}-\frac{1}{298.15}\right)\right]',
        r'\tau[n]=\begin{cases}{{capture_us}}&P[n]\ge q_t[n-1]\\\tau_e[n]&\text{otherwise}\end{cases}',
        r'q_t[n]=a[n]q_t[n-1]+(1-a[n])P[n],\quad a[n]=e^{-1/(F_s\tau[n]10^{-6})}',
        r'v[n]=1-{{ir_drop}}\operatorname{LPF}(P,{{bias_us}})',
        r'y[n]=R(x[n])(1-{{trap_strength}}q_t[n])v[n]e^{-{{thermal_gain}}(T[n]-25)+j{{trap_phase}}q_t[n]}',
        LPF, r'k_B=8.617333262\times 10^{-5}\,\mathrm{eV/K}'],
    'doherty-two-path': [BOUND, r'x_p[n]=\max(|x[n]|-{{knee}},0)e^{j\arg x[n]}',
        r'y[n]=R(x[n])+{{peaker}}R(x_p[n])e^{j{{phase_deg}}\pi/180}'],
    'envelope-tracking': [RAPP, r'D[n]=\min({{gain}}|x[n]|/{{saturation}},1)',
        r'P[n]=\frac{|x[n]|^2}{|x[n]|^2+({{saturation}}/{{gain}})^2}',
        r'v[n]=\operatorname{clip}({{supply_floor}}+{{supply_span}}\operatorname{LPF}(D,{{tracking_us}})-{{ir_drop}}\operatorname{LPF}(P,{{bias_us}}),{{supply_floor}},1.5)',
        r'y[n]=R(x[n];{{gain}},{{saturation}}v[n],{{smoothness}})e^{j{{phase}}(v[n]-1)}', LPF],
}


def attach_latex(models):
    for model in models:
        model.equations_latex = EQUATIONS[model.model_id]
        for p in model.parameters:
            p.symbol_latex = SYMBOLS[p.key]
            if model.model_id == 'saleh-twta':
                p.symbol_latex = {'phase': r'\alpha_\phi', 'phase_scale': r'\beta_\phi'}.get(p.key, p.symbol_latex)
    return models
