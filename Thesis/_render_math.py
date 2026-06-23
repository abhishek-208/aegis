# -*- coding: utf-8 -*-
"""Render LaTeX (amsmath/amssymb) equation blocks -> transparent PNGs via MiKTeX,
and compile the paper's TikZ Aegis architecture diagram -> PNG.
Outputs to ./_math/.
"""
import os, re, subprocess, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
MATH = os.path.join(HERE, "_math")
TMP = os.path.join(MATH, "_tmp")
os.makedirs(MATH, exist_ok=True)
os.makedirs(TMP, exist_ok=True)

NAVY = (14, 42, 71)
CRIMSON = (158, 42, 36)
GREEN = (30, 95, 45)

DOC = r"""\documentclass[12pt]{article}
\usepackage[paperwidth=30in,paperheight=30in,margin=0in]{geometry}
\usepackage{amsmath,amssymb,bm,xcolor}
\pagestyle{empty}
\definecolor{fg}{RGB}{%d,%d,%d}
\begin{document}
\color{fg}
\setlength{\abovedisplayskip}{0pt}\setlength{\belowdisplayskip}{0pt}
%s
\end{document}
"""

def run(cmd, cwd):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, shell=False)
    return p.returncode, (p.stdout or "") + (p.stderr or "")

def render_eq(name, body, color=NAVY, dpi=500):
    tex = os.path.join(TMP, name + ".tex")
    with open(tex, "w", encoding="utf-8") as f:
        f.write(DOC % (color[0], color[1], color[2], body))
    rc, log = run(["latex", "-interaction=nonstopmode", "-halt-on-error",
                   f"-output-directory={TMP}", tex], HERE)
    dvi = os.path.join(TMP, name + ".dvi")
    if not os.path.exists(dvi):
        print(f"[FAIL latex] {name}")
        print(log[-1500:])
        return False
    out = os.path.join(MATH, name + ".png")
    rc, log = run(["dvipng", "-D", str(dpi), "-T", "tight", "-bg", "Transparent",
                   "-q", "-o", out, dvi], HERE)
    ok = os.path.exists(out)
    print(("[ok] " if ok else "[FAIL dvipng] ") + name)
    if not ok:
        print(log[-1200:])
    return ok

# ---------------- equation blocks ----------------
EQ = {}
EQ["fl_objective"] = r"\[ \min_{\mathbf{w}}\; F(\mathbf{w})=\sum_{k=1}^{K}\frac{n_k}{K}\,F_k(\mathbf{w}) \qquad\text{honest update } \mathbf{g}_k=\mathbf{w}_k-\mathbf{w}_t \]"

EQ["attacks"] = r"""\setlength{\arraycolsep}{3pt}$\begin{array}{ll@{\qquad}ll}
\textbf{Sign-flip:} & \mathbf{g}_k=-\,s\,\bar{\mathbf{g}}_H &
\textbf{IPM:} & \mathbf{g}_k=-\,\varepsilon\,\bar{\mathbf{g}}_H,\ \ \varepsilon=0.5\\[6pt]
\textbf{Label-flip:} & y\mapsto(C-1)-y &
\textbf{ALIE:} & \mathbf{g}_k=\bar{\mathbf{g}}_H-z\,\sigma_H,\ \ z=1\\[6pt]
\textbf{Volume spam:} & n_k\uparrow\uparrow &
\textbf{Sybil:} & m\ \text{clones},\ f_{\text{eff}}\!\uparrow
\end{array}$"""

EQ["step1"] = r"\[ \mathbf{g}_k \;=\; \mathbf{w}_k \;-\; \mathbf{w}_t \]"

EQ["step2"] = r"""\begin{align*}
\textbf{Pass 1:}\quad \mathbf{m}_1 &= \operatorname{med}_{k}\!\left(\mathbf{g}_k\right),
\qquad c_{k}=\frac{\langle \mathbf{g}_k,\,\mathbf{m}_1\rangle}{\lVert\mathbf{g}_k\rVert_2\,\lVert\mathbf{m}_1\rVert_2}\\[5pt]
&S_1=\bigl\{\,k:c_{k}\,\ge\,\tau\,\bigr\},\qquad \tau=-0.3\\[7pt]
\textbf{Pass 2:}\quad \mathbf{m}_2 &= \operatorname{med}_{k\in S_1}\!\left(\mathbf{g}_k\right)\quad\text{(debiased clean center)}
\end{align*}"""

EQ["step3"] = r"""\begin{align*}
E_k &= \lVert \mathbf{g}_k-\mathbf{m}_2\rVert_2 &&\text{(magnitude anomaly)}\\[6pt]
P_k &= 1-\frac{\langle\mathbf{g}_k,\,\mathbf{m}_2\rangle}{\lVert\mathbf{g}_k\rVert_2\,\lVert\mathbf{m}_2\rVert_2} &&\text{(direction anomaly)}
\end{align*}"""

EQ["step4"] = r"""\begin{align*}
E_{\text{med}} &= \operatorname{med}_k(E_k),\qquad \mathrm{MAD}=\operatorname{med}_k\!\bigl(\lvert E_k-E_{\text{med}}\rvert\bigr)\\[5pt]
k_A^{(t)} &= k_{\max}-(k_{\max}-k_{\min})\,\min\!\Bigl(\tfrac{t}{T_{\text{warm}}},\,1\Bigr)\\[5pt]
k^{(t)} &= \max\!\Bigl(k_{\text{floor}},\;k_A^{(t)}\,(1+\nu\,\mathrm{CV})\Bigr),\qquad \mathrm{CV}=\tfrac{\mathrm{MAD}}{E_{\text{med}}}\\[5pt]
T^{(t)} &= E_{\text{med}}+k^{(t)}\cdot \mathrm{MAD},\qquad A_t=\bigl\{\,k:E_k\le T^{(t)}\,\bigr\}
\end{align*}"""

EQ["step5"] = r"""\begin{align*}
\widetilde{V}_k &= \min\!\bigl(v_k,\;2M\bigr),\quad M=\operatorname{med}_{j\in A_t}(v_j) &&\text{(volume clip)}\\[5pt]
R_k &\leftarrow \gamma\,R_k+(1-\gamma)\,P_k &&\text{(EMA reputation)}\\[5pt]
S_k &= \frac{\widetilde{V}_k}{E_k+\alpha\,P_k+\lambda\,R_k+\varepsilon},\qquad
w_k=\frac{S_k}{\sum_{j\in A_t}S_j} &&\text{(soft credit)}
\end{align*}"""

EQ["step6"] = r"\[ \hat{\mathbf{g}}_t \;=\; \sum_{k\in A_t} w_k\,\mathbf{g}_k \qquad\Longrightarrow\qquad \mathbf{w}_{t+1} \;=\; \mathbf{w}_t+\hat{\mathbf{g}}_t \;=\; \sum_{k\in A_t} w_k\,\mathbf{w}_k \]"

EQ["complexity"] = r"\[ \underbrace{\mathcal{O}(kd)}_{\textsc{Aegis}}\;\;\ll\;\;\underbrace{\mathcal{O}(kd \log k)}_{\text{Trimmed Mean}}\;\;\ll\;\;\underbrace{\mathcal{O}(k^{2}d)}_{\text{Krum, Bulyan, FoolsGold}} \]"

EQ["evo_initial"] = r"""\begin{align*}
&\text{Hard dual gate:}\;\; \cos(\mathbf{g}_k,\mathbf{m}_2)<0 \;\Rightarrow\; \text{discard}\\[4pt]
&\text{Credit:}\;\; S_k=\dfrac{\widetilde{V}_k}{E_k+10\,P_k}\quad(\text{no reputation; server momentum on})
\end{align*}"""

EQ["evo_enhanced"] = r"""\begin{align*}
&\text{Euclidean-only gate:}\;\; A_t=\{\,k:E_k\le T\,\}\quad(\text{soft }\alpha P_k)\\[4pt]
&\text{Credit:}\;\; S_k=\dfrac{\widetilde{V}_k}{E_k+\alpha P_k+\lambda R_k+\varepsilon},\;\; R_k=\gamma R_k+(1-\gamma)P_k
\end{align*}"""

EQ["hypergeom"] = r"\[ X \sim \mathrm{Hypergeometric}(K=30,\; f=9,\; m),\qquad m\in[15,25] \]"

EQ["byz_bound"] = r"\[ f=|\mathcal{B}| \;<\; \tfrac{K}{2} \;\;\Longrightarrow\;\; |\mathcal{H}| \;>\; \tfrac{K}{2} \qquad(\text{honest majority, A4}) \]"

EQ["center_drag"] = r"\[ \langle \mathbf{m}_{\text{single}},\,\bar{\mathbf{g}}_H\rangle \;<\; \langle \mathbf{m}_{\text{clean}},\,\bar{\mathbf{g}}_H\rangle \qquad\text{(IPM/sign-flip tilt the median's \emph{direction})} \]"

# ---- theoretical guarantees (thesis Appendix B) ----
EQ["lemma1"] = r"\[ c_{\mathrm{ipm}}=\frac{\langle -\varepsilon\bar{\mathbf{g}}_H,\,\mathbf{m}_1\rangle}{\lVert\varepsilon\bar{\mathbf{g}}_H\rVert\,\lVert\mathbf{m}_1\rVert}=-\,\frac{\langle\bar{\mathbf{g}}_H,\,\mathbf{m}_1\rangle}{\lVert\bar{\mathbf{g}}_H\rVert\,\lVert\mathbf{m}_1\rVert}\;\le\;0\;<\;\tau \]"

EQ["lemma2"] = r"\[ \bigl|\hat{m}-m^{*}\bigr|\;\le\;\max\!\bigl(|x_{(f+1)}-m^{*}|,\;|x_{(K-f)}-m^{*}|\bigr),\qquad f<\tfrac{K}{2} \]"

EQ["thm_conv"] = r"\[ \frac{1}{T}\sum_{t=0}^{T-1}\mathbb{E}\Bigl\lVert\nabla F(\mathbf{w}^{(t)})\Bigr\rVert^{2}\;\le\;\underbrace{\frac{2\,\bigl(F(\mathbf{w}^{(0)})-F^{*}\bigr)}{\eta\sqrt{T}}}_{O(1/\sqrt{T})\text{ SGD rate}}\;+\;\underbrace{\eta L G^{2}}_{\text{noise floor}}\;+\;\underbrace{2\,\zeta\,G}_{\text{Byzantine residual}} \]"

EQ["zeta_def"] = r"\[ \zeta=\bigl\lVert \hat{S}^{(t)}-\bar{g}^{(t)}\bigr\rVert\quad\Bigg\{\;\begin{array}{ll}\text{label-flip: } \zeta\ \text{small} & \Rightarrow\ \text{converges within 2 pp}\\[2pt]\text{ALIE: } \zeta\to \dfrac{fG}{1-f} & \Rightarrow\ \text{no convergence (10\%)}\end{array} \]"

EQ["total_cost"] = r"\[ \text{Total cost}=T\times\mathcal{O}(kd)=\mathcal{O}\!\Bigl(\tfrac{kd}{\varepsilon^{2}}\Bigr)\qquad(\zeta=0\Rightarrow\text{exact SGD rate}) \]"

# ---- NEW convergence proof (handwritten notes -> thesis Appendix B) ----
# Aegis Robustness Certificate: per-round aggregation error, three interpretable sources.
EQ["cert_error"] = r"\[ \bigl\lVert\,\hat{g}_t-\bar{g}_H\,\bigr\rVert \;\le\; \underbrace{(1+w_B)\sqrt{\kappa_H\,V_H}}_{\text{honest reweighting distortion}} \;+\; \underbrace{\frac{2M\,\lvert B_A\rvert}{S}}_{\text{clipped volume leakage}} \;+\; \underbrace{w_B\,\Delta_m}_{\text{median contamination}} \]"

# The three certificate terms, rendered standalone for the per-term panels on the certificate slide.
EQ["cert_t1"] = r"\[ (1+w_B)\sqrt{\kappa_H\,V_H} \]"
EQ["cert_t2"] = r"\[ \dfrac{2M\,\lvert B_A\rvert}{S} \]"
EQ["cert_t3"] = r"\[ w_B\,\Delta_m \]"

# Byzantine weight is score-suppressed: collapses with the score-advantage ratio r.
EQ["wb_bound"] = r"\[ w_B \;\le\; \frac{\lvert B_A\rvert\,r}{\lvert H_A\rvert+\lvert B_A\rvert\,r},\qquad r=\frac{S_B^{\max}}{S_H^{\min}}\;:\qquad \underbrace{r\to 0}_{\text{weak / filtered}}\!\Rightarrow w_B\to 0,\qquad \underbrace{r\to 1}_{\text{ALIE}}\!\Rightarrow\ \text{attackers indistinguishable} \]"

# Final Aegis convergence theorem (Appendix B.11): four components.
EQ["final_conv"] = r"\[ \sum_{t=0}^{T-1}\eta_t\!\Bigl(\tfrac{1}{2}-L\eta_t\Bigr)\mathbb{E}\bigl\lVert\nabla f(x_t)\bigr\rVert^{2} \;\le\; \underbrace{f(x_0)-f^{*}}_{\text{optimisation gap}\,\to 0} + \underbrace{L\!\sum_t\eta_t^{2}\sigma_H^{2}}_{\text{SGD noise}\,\to 0} + \underbrace{\sum_t\!\bigl(\tfrac{\eta_t}{2}+L\eta_t^{2}\bigr)\kappa_A V_H}_{\text{honest heterogeneity}} + \underbrace{\sum_t\!\bigl(\tfrac{\eta_t}{2}+L\eta_t^{2}\bigr)\zeta_A}_{\text{Byzantine residual bias}} \]"

# Residual-bias decomposition + distortion coefficient.
EQ["zeta_decomp"] = r"\[ \zeta_A \;\le\; \underbrace{\frac{12\,M^{2}\lvert B_A\rvert^{2}}{S^{2}}}_{\zeta_{\mathrm{vol}}\,:\ \text{volume clip caps }2M} \;+\; \underbrace{3\Bigl(\tfrac{f_A\,r}{(1-f_A)+f_A\,r}\Bigr)^{\!2}\Delta_m^{2}}_{\zeta_{\mathrm{med}}\,:\ \text{median contamination}},\qquad \kappa_A=3(1+w_B)^{2}\kappa_H \]"

# Perfect-filtering corollary (Appendix B): no surviving attacker -> standard SGD + heterogeneity only.
EQ["perfect_filter"] = r"\[ \lvert B_A\rvert=0 \;\Rightarrow\; w_B=0,\;\; \zeta_A=0,\;\; \kappa_A=3\kappa_H \quad\Longrightarrow\quad \text{exact non-convex SGD rate}\;+\;\text{honest-heterogeneity term only} \]"

# ---- hyperparameter justification ----
EQ["hp_credit"] = r"\[ \frac{S_{\mathrm{hon}}}{S_{\mathrm{byz}}}=\frac{E_{\mathrm{byz}}+\alpha P_{\mathrm{byz}}}{E_{\mathrm{hon}}+\alpha P_{\mathrm{hon}}}\approx\frac{25+30(2.0)}{8+30(0.25)}=\frac{85}{15.5}\approx 6.1\;\;\Rightarrow\;\;w_{B}\le 13.5\% \]"

EQ["hp_zscore"] = r"\[ z_{\mathrm{eff}}=\frac{K}{1.4826}\,;\qquad K_{\mathrm{floor}}=4\Rightarrow z\approx2.70\Rightarrow \mathrm{FP}\approx0.35\%\;;\qquad K=2\Rightarrow z\approx1.35\Rightarrow \mathrm{FP}\approx8.9\% \]"

if __name__ == "__main__":
    okc = 0
    for name, body in EQ.items():
        color = CRIMSON if name == "evo_initial" else (GREEN if name == "evo_enhanced" else NAVY)
        if render_eq(name, body, color):
            okc += 1
    print(f"\nEquations rendered: {okc}/{len(EQ)}")

    # -------------- TikZ architecture from paper --------------
    if os.path.exists(os.path.join(MATH, "arch-1.png")):
        print("[skip] architecture diagram already rendered (arch-1.png)")
        raise SystemExit(0)
    paper = r"D:\IITD\MTP 2\My Research paper\main.tex"
    with open(paper, "r", encoding="utf-8") as f:
        src = f.read()
    m = re.search(r"(\\begin\{tikzpicture\}.*?\\end\{tikzpicture\})", src, re.S)
    if m:
        tikz = m.group(1)
        arch = r"""\documentclass[border=8pt]{standalone}
\usepackage{amsmath,amssymb,xcolor}
\usepackage{tikz}
\usetikzlibrary{positioning, arrows.meta, shapes.geometric, calc, fit, backgrounds}
\begin{document}
""" + tikz + "\n\\end{document}\n"
        atex = os.path.join(TMP, "arch.tex")
        with open(atex, "w", encoding="utf-8") as f:
            f.write(arch)
        rc, log = run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                       f"-output-directory={TMP}", atex], HERE)
        apdf = os.path.join(TMP, "arch.pdf")
        if os.path.exists(apdf):
            run(["pdftoppm", "-png", "-r", "300", apdf, os.path.join(MATH, "arch")], HERE)
            print("[ok] architecture diagram ->", "arch-1.png" if os.path.exists(os.path.join(MATH,"arch-1.png")) else "arch.png")
        else:
            print("[FAIL arch pdflatex]")
            print(log[-2000:])
    else:
        print("tikzpicture not found in paper")
