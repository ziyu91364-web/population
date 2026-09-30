#!/usr/bin/env python3
"""Reproduce population-generator interference and restart illustrations.

Requirements: Python >= 3.10, NumPy, Matplotlib.  No external data are used.
Run: python3 reproduce.py --output-dir .
The curves illustrate selected times; they do not optimize over generators.
"""
import argparse
import csv
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
LOCAL_DEPS = ROOT / "plotting_dependencies"
if LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "matplotlib_config"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def norm(a):
    return float(np.linalg.norm(a, 2))


def expm_pade13(a):
    """Scaling and squaring with the order-13 diagonal Pade approximant."""
    a = np.asarray(a, dtype=complex)
    b = np.array([64764752532480000., 32382376266240000.,
                  7771770303897600., 1187353796428800.,
                  129060195264000., 10559470521600., 670442572800.,
                  33522128640., 1323241920., 40840800., 960960.,
                  16380., 182., 1.])
    scale = max(0, int(np.ceil(np.log2(max(1., norm1(a) / 5.371920351148152)))))
    a = a / (2. ** scale)
    eye = np.eye(len(a), dtype=complex)
    a2, a4 = a @ a, None
    a4 = a2 @ a2
    a6 = a4 @ a2
    u = a @ (a6 @ (b[13]*a6 + b[11]*a4 + b[9]*a2)
             + b[7]*a6 + b[5]*a4 + b[3]*a2 + b[1]*eye)
    v = (a6 @ (b[12]*a6 + b[10]*a4 + b[8]*a2)
         + b[6]*a6 + b[4]*a4 + b[2]*a2 + b[0]*eye)
    result = np.linalg.solve(v-u, v+u)
    for _ in range(scale):
        result = result @ result
    return result


def norm1(a):
    return float(np.linalg.norm(a, 1))


def expm_eigen(a):
    vals, vecs = np.linalg.eig(a)
    return (vecs * np.exp(vals)[None, :]) @ np.linalg.inv(vecs)


def square(phi):
    return np.array([[0, 1, 0, np.exp(-1j*phi)], [1, 0, 1, 0],
                     [0, 1, 0, 1], [np.exp(1j*phi), 0, 1, 0]], complex)


def path():
    return np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], complex)


def path_dephasers(c):
    """Diagonal Lindblad vectors giving r12=r23=1 and r13=c."""
    assert 0 < c < 4
    f1, f2 = np.sqrt(2-c/2), np.sqrt(c/2)
    return [np.array([f1, 0., f1]), np.array([f2, 0., -f2])]


def model(h, dephasers=None):
    n = len(h)
    # Matrix-unit construction is independent of the Kronecker identity.
    a = np.zeros((n*n, n*n), complex)
    for j in range(n*n):
        e = np.zeros((n, n), complex)
        e[j % n, j // n] = 1.
        a[:, j] = (-1j*(h@e-e@h)).reshape(n*n, order="F")
    assert norm(a+1j*(np.kron(np.eye(n), h)-np.kron(h.T, np.eye(n)))) < 1e-12
    pop = np.arange(n)*(n+1)
    coh = np.array([i for i in range(n*n) if i not in pop])
    if dephasers is None:
        rates = np.ones((n, n))-np.eye(n)
    else:
        rates = sum(.5*(f[:, None]-f[None, :])**2 for f in dephasers)
    gamma = np.diag(rates.reshape(n*n, order="F"))
    b, c, d = a[np.ix_(pop, coh)], a[np.ix_(coh, pop)], a[np.ix_(coh, coh)]
    r = np.diag(1/np.diag(gamma)[coh])
    k0 = b@r@c
    k1 = b@r@d@r@c
    k2 = b@r@d@r@d@r@c-b@r@r@c@k0
    for k in [k0, k1, k2]:
        assert norm(k.imag) < 1e-11
        assert norm(k@np.ones(n)) < 1e-11
        assert norm(k.T@np.ones(n)) < 1e-11
    return dict(h=h, a=a, gamma=gamma, pop=pop, b=b, c=c, d=d,
                r=r, rates=rates, k0=k0.real, k1=k1.real, k2=k2.real)


def population(m, eps, time):
    l = -m["gamma"]/eps**2+m["a"]/eps
    e = expm_pade13(time*l)
    p = m["pop"]
    exact = e[np.ix_(p, p)]
    derivative = (l@e)[np.ix_(p, p)]
    generator = np.linalg.solve(exact.T, derivative.T).T
    return exact, generator, l


def slow_generator(m, eps):
    # Independent invariant-graph fixed point. This is only a crosscheck,
    # not used for any displayed exact-population curve.
    b, c, d, r = (m[key] for key in ["b", "c", "d", "r"])
    y = eps*r@c
    for _ in range(20000):
        new = eps*r@c+eps*r@d@y-eps*r@y@b@y
        if norm(new-y) < 3e-15:
            return b@new/eps
        y = new
    raise RuntimeError("Invariant graph iteration failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    out = parser.parse_args().output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    eps_values = [.08, .04, .02]
    phi_values = np.linspace(0, np.pi, 121)
    c_values = np.linspace(.2, 1.2, 121)
    time = .2
    rows = []
    checks = dict(eigen_propagator=0., finite_difference_generator=0.,
                  stationary_residual=0., imaginary_residual=0.,
                  coefficient_formula=0., slow_graph_late_generator=0.)
    curves = {"square": {}, "weighted_path": {}}
    for name, params in [("square", phi_values), ("weighted_path", c_values)]:
        for eps in eps_values:
            curve = []
            for j, param in enumerate(params):
                m = model(square(param)) if name == "square" else model(path(), path_dephasers(param))
                limit = 12*np.cos(param)-4 if name == "square" else 2*(1/param-2)
                checks["coefficient_formula"] = max(checks["coefficient_formula"], abs(m["k2"][0, 2]-limit))
                if name == "weighted_path":
                    assert np.max(np.abs(m["rates"]-np.array([[0, 1, param], [1, 0, 1], [param, 1, 0]]))) < 2e-14
                exact, generator, l = population(m, eps, time)
                value = float(generator[0, 2].real/eps**2)
                curve.append(value)
                rows.append(dict(panel=name, epsilon=eps, parameter=param,
                                 time=time, restart_time="", exact_scaled_rate=value,
                                 limiting_scaled_rate=limit, restart_error="", approximation=""))
                n = len(m["h"])
                checks["stationary_residual"] = max(checks["stationary_residual"],
                    norm(exact@np.ones(n)-np.ones(n)), norm(exact.T@np.ones(n)-np.ones(n)),
                    norm(generator@np.ones(n)), norm(generator.T@np.ones(n)))
                checks["imaginary_residual"] = max(checks["imaginary_residual"], norm(exact.imag), norm(generator.imag))
                assert np.min(exact.real) > -1e-11
                if j in [0, len(params)//2, len(params)-1]:
                    p = m["pop"]
                    independent = expm_eigen(time*l)[np.ix_(p, p)]
                    checks["eigen_propagator"] = max(checks["eigen_propagator"], norm(exact-independent))
                    step = 2e-5
                    plus = expm_pade13((time+step)*l)[np.ix_(p, p)]
                    minus = expm_pade13((time-step)*l)[np.ix_(p, p)]
                    fd = np.linalg.solve(exact.T, ((plus-minus)/(2*step)).T).T
                    checks["finite_difference_generator"] = max(checks["finite_difference_generator"], norm(generator-fd))
                    if eps == .02:
                        slow = slow_generator(m, eps)
                        checks["slow_graph_late_generator"] = max(checks["slow_graph_late_generator"], norm(generator-slow))
            curves[name][eps] = np.array(curve)
    assert checks["coefficient_formula"] < 1e-11
    assert checks["eigen_propagator"] < 1e-9
    assert checks["finite_difference_generator"] < 2e-6
    assert checks["stationary_residual"] < 1e-9
    assert checks["imaginary_residual"] < 1e-9
    assert checks["slow_graph_late_generator"] < 1e-9

    restart_eps = np.array([.1, .05, .025, .0125, .00625])
    delta, final = .15, .35
    restart = {}
    for name, m, use_k2 in [("Equal-rate path: K0", model(path()), False),
                             ("Zero-flux square: S2", model(square(0)), True),
                             ("Path, c=1/4: S2", model(path(), path_dephasers(.25)), True)]:
        errors = []
        for eps in restart_eps:
            generator = m["k0"]+eps*m["k1"]+(eps**2*m["k2"] if use_k2 else 0)
            offdiag = generator[~np.eye(len(generator), dtype=bool)]
            assert np.min(offdiag) >= -1e-12
            at_delta = population(m, eps, delta)[0]
            at_final = population(m, eps, final)[0]
            approximation = expm_pade13((final-delta)*generator)@at_delta
            error = norm(at_final-approximation)
            errors.append(error)
            rows.append(dict(panel="restart", epsilon=eps, parameter="", time=final,
                             restart_time=delta, exact_scaled_rate="", limiting_scaled_rate="",
                             restart_error=error, approximation=name))
        slope = float(np.polyfit(np.log(restart_eps[-3:]), np.log(errors[-3:]), 1)[0])
        restart[name] = dict(errors=np.array(errors), slope=slope)
        assert (1.95 < slope < 2.05) if not use_k2 else (3.9 < slope < 4.1)

    with (out/"interference-data.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.4,
                         "axes.labelsize": 9, "axes.titlesize": 9.4,
                         "legend.fontsize": 7.6, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "axes.linewidth": .65,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    styles = [("#205e9c", "-"), ("#ba511a", "--"), ("#30836d", "-.")]
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.85), layout="constrained")
    for ax, name, params, title in [(axes[0], "square", phi_values, "(a) Four-cycle: changing the phase"),
                                    (axes[1], "weighted_path", c_values, "(b) Three-site path: changing dephasing")]:
        for eps, (color, ls) in zip(eps_values, styles):
            ax.plot(params, curves[name][eps], color=color, ls=ls, lw=1.35, label=rf"$\varepsilon={eps:g}$")
        limit = 12*np.cos(params)-4 if name == "square" else 2*(1/params-2)
        ax.plot(params, limit, color=".15", ls=":", lw=1.8,
                label=r"$12\cos\phi-4$" if name == "square" else r"$2(c_0^{-1}-2)$")
        threshold = np.arccos(1/3) if name == "square" else .5
        ax.axhline(0, color=".55", lw=.75)
        ax.axvline(threshold, color=".72", lw=.75, ls=":")
        ax.set(xlim=(params[0], params[-1]), title=title,
               ylabel=r"$[\dot M_\varepsilon(t)M_\varepsilon(t)^{-1}]_{13}/\varepsilon^2$",
               xlabel=r"Cycle phase $\phi$" if name == "square" else r"Coherence decay rate $c_0=r_{13}$")
        ax.legend(loc="lower left" if name == "square" else "upper right", frameon=False, handlelength=2.6)
        ax.grid(color=".92", lw=.55)
        if name == "square":
            ax.set_xticks([0, np.pi/2, np.pi], ["0", r"$\pi/2$", r"$\pi$"])
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out/"interference-figure.pdf", metadata={"Title": "Interference and dephasing control the effective population rate", "Author": ""})
    fig.savefig(out/"interference-figure.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.75, 2.95), layout="constrained")
    labels = [r"Equal-rate path: $K_0$", r"Zero-flux square: $S_\varepsilon$", r"Path, $c_0=1/4$: $S_\varepsilon$"]
    for (name, result), label, (color, ls), marker in zip(restart.items(), labels, styles, ["o", "s", "^"]):
        ax.loglog(restart_eps, result["errors"], marker+ls, color=color, lw=1.35, ms=3.7, label=label)
    guide = restart_eps/restart_eps[-1]
    ax.loglog(restart_eps, next(iter(restart.values()))["errors"][-1]*guide**2, ":", color=".35", lw=1)
    ax.loglog(restart_eps, list(restart.values())[1]["errors"][-1]*guide**4, ":", color=".35", lw=1)
    ax.set(xlabel=r"$\varepsilon$", ylabel="Restart error (operator norm)",
           title=r"Restart at $a=0.15$; comparison at $t=0.35$")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(color=".92", lw=.55)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out/"restart-errors.pdf", metadata={"Title": "Errors after a population restart", "Author": ""})
    fig.savefig(out/"restart-errors.png", dpi=240)
    plt.close(fig)

    report = ["# Interference and post-transient numerical illustrations", "",
              "The figures evaluate explicitly specified times. They do not compute a time supremum or optimize over Markov generators and initial corrections.", "",
              "## Main figure", "",
              "Both panels display entry (1,3) of the exact time-local population generator M'(t) M(t)^(-1), divided by epsilon^2, at t=0.2. The derivative is evaluated directly as P L exp(t L) P. Epsilon values are 0.08, 0.04 and 0.02.", "",
              "Panel (a): H=[[0,1,0,exp(-i phi)],[1,0,1,0],[0,1,0,1],[exp(i phi),0,1,0]], with equal dephasing rates. There are 121 uniformly spaced phi values from 0 to pi. The limiting entry is 12 cos(phi)-4; the fourth-order Markov threshold is phi=acos(1/3).", "",
              "Panel (b): H=[[0,1,0],[1,0,1],[0,1,0]] and r12=r23=1, r13=c. There are 121 uniformly spaced c values from 0.2 to 1.2. These rates are realized by two real diagonal Lindblad operators: F1=diag(sqrt(2-c/2),0,sqrt(2-c/2)), F2=diag(sqrt(c/2),0,-sqrt(c/2)). The construction is valid for 0<c<4. The limiting entry is 2(1/c-2), with sign change at c=1/2. No arbitrary positive-rate array is assumed to be Lindblad-realizable.", "",
              "The full quantum evolution is a Lindblad semigroup; a negative plotted entry concerns only stochastic divisibility of the observed population maps.", "",
              "## Restart figure", "",
              "For delta=0.15 and t=0.35 this figure displays ||M(t)-exp((t-delta)R) M(delta)||_2. Every displayed R is a Markov generator. The equal-rate path uses R=K0; the other two models use R=K0+epsilon K1+epsilon^2 K2. The two improved examples are bipartite, and consequently exhibit fourth-order accuracy at this fixed time, stronger than a generic third-order bound. Dotted lines are quadratic and quartic guides anchored at the smallest-epsilon values, not fitted coefficients.", "",
              "| Model | epsilon | Restart error |", "|---|---:|---:|"]
    for name, result in restart.items():
        for eps, error in zip(restart_eps, result["errors"]):
            report.append(f"| {name} | {eps:g} | {error:.12g} |")
    report.append("")
    for name, result in restart.items():
        report.append(f"Log-log slope over the three smallest epsilon values ({name}): {result['slope']:.8f}.\n")
    report += ["## Independent checks", "",
               "All exponentials in displayed curves use a NumPy-only scaling-and-squaring Pade(13) routine. Selected points (both endpoints and the midpoint of each parameter grid, at each epsilon) are independently checked by full Liouvillian eigendecomposition and by a central-difference population derivative (step 2e-5). At epsilon=0.02 those selected time-local generators are also compared with the independently solved slow invariant graph. The leading fourth-order coefficient is checked algebraically at every grid point. Row/column stationarity, reality and nonnegative population propagators are monitored.", ""]
    report += [f"- Maximum {key.replace('_', ' ')} discrepancy: {value:.10g}." for key, value in checks.items()]
    report += ["", "Maximum absolute deviation of displayed normalized rate from its limiting curve:"]
    for name, params in [("square", phi_values), ("weighted_path", c_values)]:
        limit = 12*np.cos(params)-4 if name == "square" else 2*(1/params-2)
        for eps in eps_values:
            report.append(f"- {name}, epsilon={eps:g}: {np.max(np.abs(curves[name][eps]-limit)):.10g}.")
    report += ["", "## Reproduction", "", f"Python {sys.version.split()[0]}, NumPy {np.__version__}, Matplotlib {matplotlib.__version__}. No SciPy or external data.", "",
               "```sh", "python3 reproduce.py --output-dir .", "```", "",
               "This regenerates interference-figure.pdf/png, restart-errors.pdf/png, interference-data.csv and this report."]
    (out/"numerical-results.md").write_text("\n".join(report)+"\n")
    print("\n".join(report[report.index("## Independent checks"):]))


if __name__ == "__main__":
    main()
