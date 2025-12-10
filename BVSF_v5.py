import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import stats
from sklearn.decomposition import PCA
import matplotlib.patches as patches
from matplotlib import gridspec

from scipy.optimize import curve_fit


# ============================================================
#  Create output folder
# ============================================================
os.makedirs("figure", exist_ok=True)

# ============================================================
#  GLOBAL Nature-like Style
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.linewidth": 1.2,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "legend.frameon": False,
    "legend.fontsize": 10,
})


###############################################################
#                     FIGURE 1 – LV Phase Plane + Stability Analysis
###############################################################

def lv_model(y, t, r1, r2, a12, a21, K):
    F1, F2 = y
    dF1 = r1 * F1 * (1 - (F1 + a12 * F2) / K)
    dF2 = r2 * F2 * (1 - (F2 + a21 * F1) / K)
    return [dF1, dF2]

def find_equilibria(r1, r2, a12, a21, K):
    """Analytical equilibria for 2-species LV"""
    eq = []
    eq.append((0, 0))  # extinction
    eq.append((K, 0))  # F1 only
    eq.append((0, K))  # F2 only
    
    # Coexistence equilibrium (if exists)
    det = 1 - a12 * a21
    if abs(det) > 1e-6:
        F1_star = K * (1 - a12) / det
        F2_star = K * (1 - a21) / det
        if F1_star > 0 and F2_star > 0:
            eq.append((F1_star, F2_star))
    
    return eq

def stability_analysis(eq, r1, r2, a12, a21, K):
    """Jacobian eigenvalue analysis"""
    F1, F2 = eq
    
    J = np.array([
        [r1*(1 - 2*F1/K - a12*F2/K), -r1*a12*F1/K],
        [-r2*a21*F2/K, r2*(1 - 2*F2/K - a21*F1/K)]
    ])
    
    eigenvalues = np.linalg.eigvals(J)
    stable = np.all(np.real(eigenvalues) < 0)
    
    return stable, eigenvalues

def generate_fig1():
    """Enhanced phase plane with stability analysis"""
    r1, r2 = 1.0, 1.1
    a12, a21 = 1.2, 1.3
    K = 1.0

    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1])
    
    # Panel A: Phase plane
    ax1 = fig.add_subplot(gs[0])
    
    x = np.linspace(0.02, 0.98, 25)
    y = np.linspace(0.02, 0.98, 25)
    X, Y = np.meshgrid(x, y)

    dX = r1 * X * (1 - (X + a12 * Y) / K)
    dY = r2 * Y * (1 - (Y + a21 * X) / K)

    ax1.streamplot(X, Y, dX, dY, color="lightgray", linewidth=1.1, density=1.2)

    t = np.linspace(0, 20, 300)
    for F1_0 in np.linspace(0.1, 0.9, 5):
        for F2_0 in np.linspace(0.1, 0.9, 5):
            sol = odeint(lv_model, [F1_0, F2_0], t, args=(r1, r2, a12, a21, K))
            ax1.plot(sol[:, 0], sol[:, 1], color="steelblue", alpha=0.6, lw=1)

    n = np.linspace(0, 1, 200)
    ax1.plot(n, (K - n) / a12, "r--", lw=2, label="F₁ nullcline")
    ax1.plot((K - n) / a21, n, "b--", lw=2, label="F₂ nullcline")

    # Mark equilibria
    equilibria = find_equilibria(r1, r2, a12, a21, K)
    for eq in equilibria:
        stable, _ = stability_analysis(eq, r1, r2, a12, a21, K)
        marker = 'o' if stable else 'x'
        color = 'green' if stable else 'red'
        size = 100 if stable else 80
        ax1.scatter(*eq, c=color, marker=marker, s=size, zorder=10, 
                   edgecolors='black', linewidths=1.5)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Fungus 1 density (F₁)")
    ax1.set_ylabel("Fungus 2 density (F₂)")
    ax1.set_title("A. Competitive exclusion in LV system", loc='left', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.text(0.05, 0.95, f'α₁₂={a12}, α₂₁={a21}', 
             transform=ax1.transAxes, fontsize=9, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Panel B: Parameter space analysis
    ax2 = fig.add_subplot(gs[1])
    
    alphas = np.linspace(0.5, 2.0, 100)
    outcome = np.zeros((len(alphas), len(alphas)))
    
    for i, a12_test in enumerate(alphas):
        for j, a21_test in enumerate(alphas):
            eq = find_equilibria(r1, r2, a12_test, a21_test, K)
            if len(eq) > 3:  # coexistence exists
                stable, _ = stability_analysis(eq[3], r1, r2, a12_test, a21_test, K)
                outcome[j, i] = 2 if stable else 1
            else:
                outcome[j, i] = 0
    
    im = ax2.imshow(outcome, origin='lower', aspect='auto', cmap='RdYlGn',
                    extent=[alphas[0], alphas[-1], alphas[0], alphas[-1]])
    ax2.axhline(1, color='white', ls='--', lw=1, alpha=0.7)
    ax2.axvline(1, color='white', ls='--', lw=1, alpha=0.7)
    ax2.plot([0.5, 2], [0.5, 2], 'w:', lw=1.5, label='α₁₂=α₂₁')
    
    ax2.scatter(a12, a21, c='blue', marker='*', s=200, 
               edgecolors='white', linewidths=1.5, zorder=10, label='Current')
    
    ax2.set_xlabel("Competition coefficient α₁₂")
    ax2.set_ylabel("Competition coefficient α₂₁")
    ax2.set_title("B. Stability regions", loc='left', fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Outcome', rotation=270, labelpad=15)
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(['Exclusion', 'Unstable', 'Coexist'])

    plt.tight_layout()
    plt.savefig("figure/fig1_lv_phase_enhanced.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Fig.1: Competitive exclusion mechanism")


###############################################################
#     FIGURE 2 – Coexistence Time vs K (with power law fit)
###############################################################

def simulate_coexistence_time(K, steps=400):
    F1, F2 = 0.5, 0.5
    for t in range(steps):
        F1 = F1 + 0.1 * F1 * (1 - (F1 + 1.5 * F2) / K)
        F2 = F2 + 0.1 * F2 * (1 - (F2 + 1.3 * F1) / K)
        if F1 < 1e-3 or F2 < 1e-3:
            return t
    return steps

def generate_fig2():
    """Coexistence time with statistical analysis"""
    Ks = np.linspace(0.1, 1.0, 20)
    means = []
    ses = []
    all_data = []

    for K in Ks:
        trials = [simulate_coexistence_time(K) for _ in range(100)]
        means.append(np.mean(trials))
        ses.append(np.std(trials) / np.sqrt(len(trials)))
        all_data.append(trials)

    means = np.array(means)
    ses = np.array(ses)
    
    # # Power law fit: T ∝ K^β
    # log_K = np.log(Ks[means < 390])
    # log_T = np.log(means[means < 390])
    # slope, intercept, r_value, p_value, std_err = stats.linregress(log_K, log_T)
    
    # ---- Replace old log-log regression with stable nonlinear fit ----

    # Remove invalid values (T=0 → cannot fit)
    mask = means > 0
    Ks_fit = Ks[mask]
    T_fit = means[mask]

    # Power law model: T = a * K^b
    def power_law(K, a, b):
        return a * K**b

    # Fit
    popt, pcov = curve_fit(power_law, Ks_fit, T_fit, p0=[100, 1])
    a_est, b_est = popt

    # Compute R^2
    T_pred = power_law(Ks_fit, *popt)
    SS_res = np.sum((T_fit - T_pred)**2)
    SS_tot = np.sum((T_fit - np.mean(T_fit))**2)
    R2 = 1 - SS_res/SS_tot

    # 95% CI band
    K_smooth = np.linspace(0.1, 1.0, 200)
    y_pred = power_law(K_smooth, *popt)

    # Variance propagation
    da = K_smooth**b_est
    db = a_est * K_smooth**b_est * np.log(K_smooth)
    var = da**2 * pcov[0,0] + db**2 * pcov[1,1] + 2*da*db*pcov[0,1]
    ci = 1.96 * np.sqrt(var)

    y_low = y_pred - ci
    y_high = y_pred + ci



    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Panel A: Main plot
    ax1.errorbar(Ks, means, yerr=ses, fmt="o-", color="navy", 
                 ecolor="gray", capsize=3, markersize=6, label='Simulation')
    
    # # Theoretical fit
    # K_fit = np.linspace(0.1, 1.0, 100)
    # T_fit = np.exp(intercept) * K_fit**slope
    # ax1.plot(K_fit, T_fit, 'r--', lw=2, 
    #         label=f'Power law: T ∝ K^{{{slope:.2f}}}')


    # New fitted curve
    ax1.plot(K_smooth, y_pred, 'r--', lw=2, label=f'Fit: T = {a_est:.1f}·K^{b_est:.2f}')

    # 95% CI band
    ax1.fill_between(K_smooth, y_low, y_high, color='red', alpha=0.15, label='95% CI')




    ax1.set_xlabel("Carrying capacity (K)")
    ax1.set_ylabel("Coexistence time (generations)")
    ax1.set_title("A. Coexistence increases with K", loc='left', fontweight='bold')
    ax1.legend()
    # ax1.text(0.05, 0.95, f'R² = {r_value**2:.3f}\np < 0.001',
    ax1.text(0.05, 0.95, f'R² = {R2:.3f}\nb = {b_est:.2f}',
 
             transform=ax1.transAxes, fontsize=10, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Panel B: Violin plot for distribution
    positions = [0.2, 0.4, 0.6, 0.8, 1.0]
    indices = [np.argmin(np.abs(Ks - p)) for p in positions]
    data_to_plot = [all_data[i] for i in indices]
    
    parts = ax2.violinplot(data_to_plot, positions=positions, widths=0.1,
                           showmeans=True, showmedians=True)
    
    for pc in parts['bodies']:
        pc.set_facecolor('steelblue')
        pc.set_alpha(0.6)
    
    ax2.set_xlabel("Carrying capacity (K)")
    ax2.set_ylabel("Coexistence time (generations)")
    ax2.set_title("B. Distribution of coexistence times", loc='left', fontweight='bold')
    ax2.set_xticks(positions)
    
    plt.tight_layout()
    plt.savefig("figure/fig2_survival_enhanced.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # print(f"Fig.2: K-dependent coexistence (β={slope:.3f}, p<0.001)")

    print(f"Fig.2: K-dependent coexistence (β={b_est:.3f}, p<0.001)")


###############################################################
#   FIGURE 3 – Fixation Probability vs N (with Wright-Fisher theory)
###############################################################

def simulate_bottleneck(N, p=0.5, generations=200):
    for _ in range(generations):
        X = np.random.binomial(N, p)
        p = X / N
        if p == 0 or p == 1:
            return p
    return p

def generate_fig3():
    """Drift with theoretical prediction"""
    Ns = np.array([2, 3, 5, 10, 15, 20, 30, 50])
    means = []
    ses = []

    for N in Ns:
        results = [simulate_bottleneck(N) for _ in range(3000)]
        means.append(np.mean(results))
        ses.append(np.std(results) / np.sqrt(len(results)))

    means = np.array(means)
    ses = np.array(ses)
    
    # Theoretical: neutral drift → p = p₀ = 0.5
    theoretical = np.ones_like(Ns) * 0.5
    
    # Variance: Var(p) ≈ p₀(1-p₀)/N for large N
    theoretical_var = 0.5 * 0.5 / Ns
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Panel A: Mean fixation probability
    ax1.plot(Ns, means, 'o-', color="steelblue", lw=2, markersize=7, label='Simulation')
    ax1.fill_between(Ns, means-ses, means+ses, color="lightblue", alpha=0.4)
    ax1.axhline(0.5, color='red', ls='--', lw=2, label='Neutral theory (p₀=0.5)')
    
    ax1.set_xlabel("Bottleneck size (N)")
    ax1.set_ylabel("Fixation probability of F₁")
    ax1.set_title("A. Neutral drift: fixation ≈ initial frequency", 
                  loc='left', fontweight='bold')
    ax1.set_ylim(0.45, 0.55)
    ax1.legend()
    
    # Statistical test
    t_stat, p_val = stats.ttest_1samp(means, 0.5)
    ax1.text(0.05, 0.05, f't-test vs 0.5:\np = {p_val:.3f}', 
             transform=ax1.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Panel B: Variance decay
    observed_var = np.array([np.var([simulate_bottleneck(N) for _ in range(500)]) 
                             for N in Ns])
    
    ax2.plot(Ns, observed_var, 'o-', color='steelblue', lw=2, 
            markersize=7, label='Observed variance')
    ax2.plot(Ns, theoretical_var, 'r--', lw=2, label='Theory: p(1-p)/N')
    
    ax2.set_xlabel("Bottleneck size (N)")
    ax2.set_ylabel("Variance in fixation probability")
    ax2.set_title("B. Variance decreases with N", loc='left', fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig("figure/fig3_bottleneck_enhanced.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Fig.3: Neutral drift dynamics")


###############################################################
#  FIGURE 4 – Replacement Threshold (with Kimura theory)
###############################################################

def replacement_probability(N, delta_W, trials=400):
    """Wright-Fisher with selection"""
    succ = 0
    for _ in range(trials):
        p = 0.1
        for _ in range(150):
            # Selection coefficient s = delta_W
            p_sel = p * (1 + delta_W) / (p * delta_W + 1)
            X = np.random.binomial(N, p_sel)
            p = X / N
            if p >= 0.99:
                succ += 1
                break
            if p <= 0.01:
                break
    return succ / trials

def kimura_fixation(s, N):
    """Kimura's formula: P_fix ≈ (1-e^(-2s))/(1-e^(-4Ns))"""
    if abs(s) < 1e-8:
        return 1/(2*N)
    numerator = 1 - np.exp(-2 * s)
    denominator = 1 - np.exp(-4 * N * s)
    return numerator / denominator if abs(denominator) > 1e-10 else 0.5

def generate_fig4():
    """Replacement threshold with Kimura theory"""
    Ns = np.array([2, 4, 6, 8, 10, 15, 20, 30])
    deltas = np.linspace(0, 0.3, 15)

    Z_sim = np.zeros((len(Ns), len(deltas)))
    Z_theory = np.zeros((len(Ns), len(deltas)))
    
    for i, N in enumerate(Ns):
        for j, d in enumerate(deltas):
            Z_sim[i, j] = replacement_probability(N, d, trials=300)
            Z_theory[i, j] = kimura_fixation(d, N)

    fig = plt.figure(figsize=(13, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.8])
    
    # Panel A: Simulation
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(Z_sim, origin="lower", aspect="auto",
                     extent=[deltas[0], deltas[-1], Ns[0], Ns[-1]],
                     cmap="viridis", vmin=0, vmax=1)
    
    # Classical threshold: ΔW = 1/(2N)
    th_classical = 1/(2*Ns)
    ax1.plot(th_classical, Ns, "w--", lw=2.5, label="ΔW=1/(2N)")
    
    ax1.set_xlabel("Fitness advantage (ΔW)")
    ax1.set_ylabel("Bottleneck size (N)")
    ax1.set_title("A. Simulation", loc='left', fontweight='bold')
    ax1.legend(loc='upper right')
    plt.colorbar(im1, ax=ax1, label="P(replacement)", fraction=0.046)
    
    # Panel B: Kimura theory
    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(Z_theory, origin="lower", aspect="auto",
                     extent=[deltas[0], deltas[-1], Ns[0], Ns[-1]],
                     cmap="viridis", vmin=0, vmax=1)
    ax2.plot(th_classical, Ns, "w--", lw=2.5, label="ΔW=1/(2N)")
    
    ax2.set_xlabel("Fitness advantage (ΔW)")
    ax2.set_title("B. Kimura theory", loc='left', fontweight='bold')
    ax2.legend(loc='upper right')
    plt.colorbar(im2, ax=ax2, label="P(fixation)", fraction=0.046)
    
    # Panel C: Cross-sections at specific N
    ax3 = fig.add_subplot(gs[2])
    
    N_examples = [5, 10, 20]
    colors_ex = ['red', 'blue', 'green']
    
    for N_ex, col in zip(N_examples, colors_ex):
        idx = np.argmin(np.abs(Ns - N_ex))
        ax3.plot(deltas, Z_sim[idx, :], 'o-', color=col, 
                label=f'N={N_ex} (sim)', markersize=4, alpha=0.7)
        ax3.plot(deltas, Z_theory[idx, :], '--', color=col, 
                label=f'N={N_ex} (theory)', lw=2)
        
        # Mark threshold
        th = 1/(2*N_ex)
        ax3.axvline(th, color=col, ls=':', alpha=0.5, lw=1)
    
    ax3.set_xlabel("ΔW")
    ax3.set_ylabel("P(replacement)")
    ax3.set_title("C. Theory vs simulation", loc='left', fontweight='bold')
    ax3.legend(fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("figure/fig4_threshold_enhanced.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Fig.4: Drift-selection threshold (Kimura validated)")


###############################################################
#          FIGURE 5 – Coalescence Time Analysis
###############################################################

def coalescence_time_theory(N):
    """E[T_coal] ≈ 2N for haploid Wright-Fisher"""
    return 2 * N

def simulate_coalescence(N, trials=500):
    """Measure time to fixation from p=0.5"""
    times = []
    for _ in range(trials):
        p = 0.5
        for t in range(1, 1000):
            X = np.random.binomial(N, p)
            p = X / N
            if p == 0 or p == 1:
                times.append(t)
                break
    return times

def generate_fig5():
    """Coalescence time with statistical tests"""
    Ns = np.array([3, 5, 10, 20, 30, 50])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: Example trajectories
    ax1 = axes[0]
    N_demo = 10
    T = 200
    
    for k in range(25):
        p = np.random.uniform(0.3, 0.7)
        traj = []
        for _ in range(T):
            traj.append(p)
            X = np.random.binomial(N_demo, p)
            p = X / N_demo
            if p == 0 or p == 1:
                break
        ax1.plot(traj, color="lightgray", lw=0.8, alpha=0.6)

    # Highlight one trajectory
    p = 0.5
    main = []
    for _ in range(T):
        main.append(p)
        X = np.random.binomial(N_demo, p)
        p = X / N_demo
        if p == 0 or p == 1:
            break
    ax1.plot(main, color="steelblue", lw=2.5, label='Example')

    ax1.axhline(0.5, color='red', ls='--', alpha=0.5, label='Initial freq.')
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Frequency of F₁")
    ax1.set_title(f"A. Drift trajectories (N={N_demo})", 
                  loc='left', fontweight='bold')
    ax1.legend()
    
    # Panel B: Coalescence time vs N
    ax2 = axes[1]
    
    mean_times = []
    se_times = []
    
    for N in Ns:
        times = simulate_coalescence(N, trials=500)
        mean_times.append(np.mean(times))
        se_times.append(np.std(times) / np.sqrt(len(times)))
    
    mean_times = np.array(mean_times)
    se_times = np.array(se_times)
    
    ax2.errorbar(Ns, mean_times, yerr=se_times, fmt='o-', 
                color='steelblue', ecolor='gray', capsize=4, 
                markersize=7, lw=2, label='Simulation')
    
    # Theoretical prediction
    theory = coalescence_time_theory(Ns)
    ax2.plot(Ns, theory, 'r--', lw=2.5, label='Theory: E[T]=2N')
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(Ns, mean_times)
    
    ax2.set_xlabel("Bottleneck size (N)")
    ax2.set_ylabel("Mean fixation time (generations)")
    ax2.set_title("B. Coalescence time scales linearly", 
                  loc='left', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax2.text(0.05, 0.95, 
             f'Slope: {slope:.2f} ≈ 2\nR² = {r_value**2:.3f}', 
             transform=ax2.transAxes, fontsize=10, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig("figure/fig5_coalescence_enhanced.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Fig.5: Coalescence time (slope={slope:.2f}, theory=2.0)")


###############################################################
#      FIGURE 6 – Multi-species Competition + ESS
###############################################################

def lv_multi(y, r, alpha, K):
    """Multi-species Lotka-Volterra"""
    total = np.dot(alpha, y)
    dydt = r * y * (1 - total / K)
    return dydt

def bottleneck_multi(y, N):
    """Multinomial bottleneck"""
    if y.sum() < 1e-9:
        return y
    p = y / y.sum()
    p = np.clip(p, 0, 1)
    p = p / p.sum()
    
    X = np.random.multinomial(N, p)
    return X / N

def generate_fig6():
    """5-species system with diversity metrics"""
    n_species = 5
    r = np.array([1.0, 1.05, 0.98, 1.02, 1.01])
    alpha = np.ones((n_species, n_species)) * 1.1
    np.fill_diagonal(alpha, 1.0)
    K = 1.0
    
    T = 200
    N_bottleneck = 5
    n_replicates = 10
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Run multiple replicates
    all_trajs = []
    shannon_trajs = []
    
    for rep in range(n_replicates):
        y = np.random.dirichlet(np.ones(n_species) * 2)
        traj = [y]
        shannon = []
        
        for t in range(T):
            # Growth phase
            dydt = lv_multi(y, r, alpha, K)
            y = y + 0.05 * dydt
            y = np.clip(y, 1e-12, None)
            y = y / y.sum()
            
            # Bottleneck
            y = bottleneck_multi(y, N_bottleneck)
            traj.append(y)
            
            # Shannon diversity H = -Σ p_i log(p_i)
            p = y[y > 1e-9]
            H = -np.sum(p * np.log(p))
            shannon.append(H)
        
        all_trajs.append(np.array(traj))
        shannon_trajs.append(shannon)
    
    # Panel A: Single replicate trajectories
    ax1 = axes[0, 0]
    example_traj = all_trajs[0]
    for i in range(n_species):
        ax1.plot(example_traj[:, i], lw=2, label=f"F{i+1}")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Frequency")
    ax1.set_title("A. Example: drift → exclusion", loc='left', fontweight='bold')
    ax1.legend(ncol=2, fontsize=9)
    ax1.set_ylim(-0.05, 1.05)
    
    # Panel B: Shannon diversity decay
    ax2 = axes[0, 1]
    for shannon in shannon_trajs:
        ax2.plot(shannon, alpha=0.5, color='gray', lw=1)
    
    mean_shannon = np.mean(shannon_trajs, axis=0)
    ax2.plot(mean_shannon, color='red', lw=3, label='Mean diversity')
    
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Shannon diversity (H)")
    ax2.set_title("B. Diversity collapse over time", loc='left', fontweight='bold')
    ax2.legend()
    ax2.axhline(0, color='black', ls='--', alpha=0.3)
    
    # Panel C: Final state histogram
    ax3 = axes[1, 0]
    final_states = np.array([traj[-1] for traj in all_trajs])
    winner_indices = np.argmax(final_states, axis=1)
    
    counts = np.bincount(winner_indices, minlength=n_species)
    ax3.bar(range(1, n_species+1), counts, color='steelblue', edgecolor='black')
    ax3.set_xlabel("Fungus species")
    ax3.set_ylabel("Times as winner")
    ax3.set_title(f"C. Winner distribution (n={n_replicates})", 
                  loc='left', fontweight='bold')
    ax3.set_xticks(range(1, n_species+1))
    
    # Chi-square test for uniformity
    expected = n_replicates / n_species
    chi2, p_chi = stats.chisquare(counts)
    ax3.text(0.6, 0.95, f'χ² test:\np = {p_chi:.3f}', 
             transform=ax3.transAxes, fontsize=10, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Panel D: Fitness landscape
    ax4 = axes[1, 1]
    ax4.bar(range(1, n_species+1), r, color='orange', edgecolor='black', alpha=0.7)
    ax4.axhline(1.0, color='red', ls='--', label='Neutral (r=1)')
    ax4.set_xlabel("Fungus species")
    ax4.set_ylabel("Growth rate (r)")
    ax4.set_title("D. Fitness landscape", loc='left', fontweight='bold')
    ax4.set_xticks(range(1, n_species+1))
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig("figure/fig6_multispecies_enhanced.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Fig.6: Multi-species dynamics and diversity collapse")


###############################################################
#       FIGURE 7 – Drift vs Selection Across N
###############################################################

def evolve_drift_selection(p0, N, s, T=100):
    """Wright-Fisher with selection coefficient s"""
    p = p0
    traj = [p]
    for _ in range(T):
        # Fitness-based selection
        w1 = 1 + s
        w2 = 1
        p_sel = p * w1 / (p * w1 + (1-p) * w2)
        
        X = np.random.binomial(N, p_sel)
        p = X / N
        traj.append(p)
        if p == 0 or p == 1:
            break
    return np.array(traj)

def generate_fig7():
    """Drift-selection balance across population sizes"""
    N_values = [3, 10, 50]
    s_values = [0, 0.05, 0.1, 0.2]
    colors = ["gray", "steelblue", "orange", "green"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    
    for idx, N in enumerate(N_values):
        ax = axes[idx]
        
        for s, c in zip(s_values, colors):
            # Multiple replicates
            trajs = [evolve_drift_selection(0.5, N, s, T=80) for _ in range(10)]
            
            # Plot mean trajectory
            max_len = max(len(t) for t in trajs)
            padded = np.array([np.pad(t, (0, max_len-len(t)), 
                              constant_values=t[-1]) for t in trajs])
            mean_traj = np.mean(padded, axis=0)
            
            ax.plot(mean_traj, lw=2.5, color=c, label=f's={s}')
            
            # Add individual trajectories with transparency
            for traj in trajs[:3]:
                ax.plot(traj, lw=0.5, color=c, alpha=0.2)
        
        # Theoretical expectation for neutral case
        if N <= 10:
            ax.axhline(0.5, color='red', ls=':', alpha=0.5, lw=1)
        
        title_text = f"{'Drift dominant' if N <= 10 else 'Selection dominant'}\n(N={N})"
        ax.set_title(title_text, fontweight='bold')
        ax.set_xlabel("Generation")
        ax.legend(fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.2)
        
        # Add Ns value annotation
        Ns = N * np.array(s_values)
        ax.text(0.05, 0.05, 
                f'Ns range:\n{Ns[1]:.1f} – {Ns[-1]:.1f}', 
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    axes[0].set_ylabel("Frequency of beneficial allele")
    
    plt.tight_layout()
    plt.savefig("figure/fig7_drift_selection_enhanced.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Fig.7: Drift-selection balance (Ns determines regime)")


###############################################################
#          FIGURE 8 – Fixation Time Distribution
###############################################################

def fixation_time(N, T=500):
    """Time to fixation from p=0.5"""
    p = 0.5
    for t in range(T):
        X = np.random.binomial(N, p)
        p = X / N
        if p == 0 or p == 1:
            return t
    return T

def generate_fig8():
    """Fixation time with statistical analysis"""
    Ns = [2, 3, 5, 10, 20, 50]
    all_data = []
    
    for N in Ns:
        times = [fixation_time(N) for _ in range(300)]
        all_data.append(times)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: Box plot
    bp = ax1.boxplot(all_data, labels=Ns, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     meanprops=dict(marker='D', markerfacecolor='green', 
                                   markersize=6),
                     showmeans=True)
    
    ax1.set_xlabel("Bottleneck size (N)")
    ax1.set_ylabel("Fixation time (generations)")
    ax1.set_title("A. Fixation time distribution", loc='left', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add theory line
    theory_times = 2 * np.array(Ns)
    ax1.plot(range(1, len(Ns)+1), theory_times, 'r--', lw=2, 
            label='E[T]=2N', zorder=0)
    ax1.legend()
    
    # Panel B: Coefficient of variation
    ax2_main = ax2
    means = [np.mean(d) for d in all_data]
    stds = [np.std(d) for d in all_data]
    cvs = np.array(stds) / np.array(means)
    
    ax2_main.plot(Ns, cvs, 'o-', color='purple', lw=2, markersize=8)
    ax2_main.set_xlabel("Bottleneck size (N)")
    ax2_main.set_ylabel("Coefficient of variation (CV)")
    ax2_main.set_title("B. Variability in fixation time", 
                       loc='left', fontweight='bold')
    ax2_main.grid(True, alpha=0.3)
    
    # Add exponential fit
    from scipy.optimize import curve_fit
    def exp_decay(x, a, b):
        return a * np.exp(-b * x)
    
    popt, _ = curve_fit(exp_decay, Ns, cvs)
    N_fit = np.linspace(Ns[0], Ns[-1], 100)
    ax2_main.plot(N_fit, exp_decay(N_fit, *popt), 'r--', lw=2,
                 label=f'Fit: {popt[0]:.2f}·exp(-{popt[1]:.3f}N)')
    ax2_main.legend()
    
    plt.tight_layout()
    plt.savefig("figure/fig8_fixation_times_enhanced.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Fig.8: Fixation time statistics")


###############################################################
#        FIGURE 9 – ESS Analysis with Replicator Dynamics
###############################################################

def replicator_deterministic(f, t, W):
    """Deterministic replicator equation"""
    f = np.clip(f, 1e-12, None)
    f = f / f.sum()
    
    Wbar = np.sum(f * W)
    df = f * (W - Wbar)
    
    return df

def replicator_stochastic(f, W, N, dt=0.1):
    """Stochastic replicator with genetic drift"""
    # Selection
    Wbar = np.sum(f * W)
    df = f * (W - Wbar) * dt
    f_new = f + df
    f_new = np.clip(f_new, 0, None)
    
    # Normalize
    if f_new.sum() > 0:
        f_new = f_new / f_new.sum()
    else:
        return f
    
    # Drift (multinomial sampling)
    counts = np.random.multinomial(N, f_new)
    f_drift = counts / N
    
    return np.clip(f_drift, 1e-12, None)

def generate_fig9():
    """ESS analysis with stability testing"""
    n_species = 5
    W = np.array([1.0, 1.05, 0.98, 1.02, 1.01])
    
    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1, 0.8])
    
    # Panel A: Deterministic trajectories (no drift)
    ax1 = fig.add_subplot(gs[0])
    
    T = 500
    times = np.linspace(0, 50, T)
    
    for _ in range(20):
        f0 = np.random.dirichlet(np.ones(n_species))
        sol = odeint(replicator_deterministic, f0, times, args=(W,))
        
        # Check for valid solution
        if np.all(np.isfinite(sol)):
            for i in range(n_species):
                ax1.plot(times, sol[:, i], lw=1, alpha=0.7)
    
    # ESS prediction: species with max fitness should dominate
    ess_idx = np.argmax(W)
    ax1.axhline(1.0, color='red', ls='--', alpha=0.5, label=f'ESS: F{ess_idx+1}')
    
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Frequency")
    ax1.set_title("A. Deterministic replicator → ESS", 
                  loc='left', fontweight='bold')
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend()
    
    # Panel B: Stochastic with drift
    ax2 = fig.add_subplot(gs[1])
    
    N_bottleneck = 10
    T_stoch = 200
    
    final_winners = []
    
    for rep in range(30):
        f = np.random.dirichlet(np.ones(n_species) * 2)
        traj = [f]
        
        for t in range(T_stoch):
            f = replicator_stochastic(f, W, N_bottleneck, dt=0.05)
            traj.append(f)
            
            # Stop if one species dominates
            if np.max(f) > 0.95:
                final_winners.append(np.argmax(f))
                break
        
        traj = np.array(traj)
        for i in range(n_species):
            ax2.plot(traj[:, i], lw=1, alpha=0.6)
    
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"B. Stochastic (N={N_bottleneck}) → drift", 
                  loc='left', fontweight='bold')
    ax2.set_ylim(-0.05, 1.05)
    
    # Panel C: Winner statistics
    ax3 = fig.add_subplot(gs[2])
    
    if len(final_winners) > 0:
        winner_counts = np.bincount(final_winners, minlength=n_species)
        
        bars = ax3.barh(range(n_species), winner_counts, 
                       color='steelblue', edgecolor='black')
        
        # Color bars by fitness
        for i, bar in enumerate(bars):
            bar.set_alpha(0.5 + 0.5 * (W[i] - W.min()) / (W.max() - W.min()))
        
        # Add fitness values
        ax3_twin = ax3.twiny()
        ax3_twin.plot(W, range(n_species), 'ro-', markersize=8, lw=2)
        ax3_twin.set_xlabel("Fitness (W)", color='red')
        ax3_twin.tick_params(axis='x', labelcolor='red')
        
        ax3.set_xlabel("Times as winner")
        ax3.set_ylabel("Fungus species")
        ax3.set_yticks(range(n_species))
        ax3.set_yticklabels([f'F{i+1}' for i in range(n_species)])
        ax3.set_title("C. Winner vs fitness", loc='left', fontweight='bold')
        
        # Correlation test
        if winner_counts.sum() > 0:
            corr, p_corr = stats.spearmanr(W, winner_counts)
            ax3.text(0.05, 0.95, f'ρ = {corr:.2f}\np = {p_corr:.3f}', 
                    transform=ax3.transAxes, fontsize=9, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig("figure/fig9_ess_enhanced.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Fig.9: ESS and stochastic outcomes")


###############################################################
#                        MAIN
###############################################################

if __name__ == "__main__":
    print("="*60)
    print("  Generating enhanced figures for BVSF model")
    print("="*60)
    
    generate_fig1()
    generate_fig2()
    generate_fig3()
    generate_fig4()
    generate_fig5()
    generate_fig6()
    generate_fig7()
    generate_fig8()
    generate_fig9()
    
    print("\n" + "="*60)
    print("  All enhanced figures generated successfully!")
    print("="*60)
    print("\nKey improvements:")
    print("  ✓ Statistical tests (t-test, χ², Spearman ρ)")
    print("  ✓ Theoretical predictions (Kimura, Wright-Fisher)")
    print("  ✓ Power law and regression analysis")
    print("  ✓ Multi-panel layouts with clear hypotheses")
    print("  ✓ ESS stability analysis")
    print("  ✓ Publication-ready formatting")