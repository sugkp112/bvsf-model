# ================================================================
# reliability_tests.py (FIXED VERSION)
# ================================================================

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import odeint

# 从 BVSF_v5 导入存在的函数
from BVSF_v5 import (
    replacement_probability,
    simulate_coalescence,
    simulate_bottleneck,
    simulate_coexistence_time
)

if not os.path.exists("figures"):
    os.makedirs("figures")


# S1 — LV system reliability
def test_S1_LV_reliability():
    """测试 LV 系统在不同时间步长下的稳定性"""
    def lv_system(X, t, alpha12, alpha21):
        F1, F2 = X
        d1 = F1 * (1 - F1 - alpha12 * F2)
        d2 = F2 * (1 - F2 - alpha21 * F1)
        return [d1, d2]

    t1 = np.linspace(0, 10, 100)
    t2 = np.linspace(0, 10, 500)
    t3 = np.linspace(0, 10, 2000)

    sol1 = odeint(lv_system, [0.4, 0.6], t1, args=(1.2, 1.3))
    sol2 = odeint(lv_system, [0.4, 0.6], t2, args=(1.2, 1.3))
    sol3 = odeint(lv_system, [0.4, 0.6], t3, args=(1.2, 1.3))

    plt.figure(figsize=(8, 5))
    plt.plot(t1, sol1[:, 0], "--", label="100 steps", lw=2)
    plt.plot(t2, sol2[:, 0], ":", label="500 steps", lw=2)
    plt.plot(t3, sol3[:, 0], "-", label="2000 steps", lw=1.5)
    plt.xlabel("Time")
    plt.ylabel("F₁ density")
    plt.title("S1 — LV System Numerical Stability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/S1_LV_reliability.png", dpi=200)
    plt.close()
    print("✓ S1: LV reliability test completed")


# S2 — Neutral drift reliability
def test_S2_neutral_reliability():
    """测试中性漂变在不同代数下的收敛性"""
    N = 20
    reps = 300
    gens_list = [100, 500, 1000, 2000]
    results = []

    for G in gens_list:
        fix = 0
        for _ in range(reps):
            # 使用 simulate_bottleneck 模拟中性漂变
            final_p = simulate_bottleneck(N, generations=G)
            if final_p >= 0.99:
                fix += 1
        results.append(fix / reps)

    plt.figure(figsize=(8, 5))
    plt.plot(gens_list, results, "o-", markersize=8, lw=2, color='steelblue')
    plt.axhline(0.5, color='red', ls='--', lw=2, alpha=0.7, label='Expected (p₀=0.5)')
    plt.xlabel("Number of generations")
    plt.ylabel("Fixation probability")
    plt.title("S2 — Neutral Drift Convergence Test")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/S2_neutral_reliability.png", dpi=200)
    plt.close()
    print("✓ S2: Neutral drift reliability test completed")


# S3 — Replacement threshold stability
def test_S3_threshold_reliability():
    """测试替换阈值在不同参数下的稳定性"""
    N = 10
    DeltaW_range = np.linspace(-0.02, 0.08, 40)
    trials_list = [100, 300, 500]

    plt.figure(figsize=(8, 5))

    for trials in trials_list:
        probs = []
        for dW in DeltaW_range:
            probs.append(replacement_probability(N=N, delta_W=dW, trials=trials))
        plt.plot(DeltaW_range, probs, label=f"{trials} trials", lw=2)

    # 理论阈值
    plt.axvline(1 / (2 * N), color="black", ls="--", lw=2, 
                label=f'Theory: ΔW=1/(2N)={1/(2*N):.3f}')
    plt.axhline(0.5, color='gray', ls=':', alpha=0.5)
    
    plt.xlabel("Fitness advantage (ΔW)")
    plt.ylabel("Replacement probability")
    plt.title("S3 — Replacement Threshold Stability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/S3_threshold_reliability.png", dpi=200)
    plt.close()
    print("✓ S3: Threshold reliability test completed")


# S4 — Coalescence ~2N
def test_S4_coalescence_reliability():
    """验证合并时间是否符合 2N 理论预测"""
    Ns = [5, 10, 20, 30, 50]
    means = []
    ses = []

    for N in Ns:
        times = simulate_coalescence(N, trials=300)
        means.append(np.mean(times))
        ses.append(np.std(times) / np.sqrt(len(times)))

    means = np.array(means)
    ses = np.array(ses)

    plt.figure(figsize=(8, 5))
    plt.errorbar(Ns, means, yerr=ses, fmt="o-", markersize=8, 
                capsize=5, lw=2, label="Simulated", color='steelblue')
    plt.plot(Ns, [2 * N for N in Ns], "--", lw=2.5, 
            label="Theory: E[T]=2N", color='red')
    
    plt.xlabel("Population size (N)")
    plt.ylabel("Mean coalescence time (generations)")
    plt.title("S4 — Coalescence Time Validation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/S4_coalescence_reliability.png", dpi=200)
    plt.close()
    print("✓ S4: Coalescence reliability test completed")


# S5 — Coexistence time reliability
def test_S5_coexistence_reliability():
    """测试共存时间计算的稳定性"""
    K_values = [0.3, 0.5, 0.7, 0.9]
    trials_list = [50, 100, 200]
    
    plt.figure(figsize=(8, 5))
    
    for trials in trials_list:
        means = []
        for K in K_values:
            times = [simulate_coexistence_time(K) for _ in range(trials)]
            means.append(np.mean(times))
        plt.plot(K_values, means, 'o-', label=f'{trials} trials', 
                markersize=7, lw=2)
    
    plt.xlabel("Carrying capacity (K)")
    plt.ylabel("Mean coexistence time")
    plt.title("S5 — Coexistence Time Stability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/S5_coexistence_reliability.png", dpi=200)
    plt.close()
    print("✓ S5: Coexistence reliability test completed")


# S6 — Multi-species winner distribution
def test_S6_winner_reliability():
    """测试多物种系统赢家分布的稳定性"""
    n_species = 5
    N_bottleneck = 5
    T = 200
    reps_list = [50, 100, 200]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    for idx, reps in enumerate(reps_list):
        winners = []
        
        for _ in range(reps):
            # 初始化均匀分布
            y = np.ones(n_species) / n_species
            
            # 模拟演化
            for t in range(T):
                # 简单漂变
                if y.sum() > 0:
                    p = y / y.sum()
                    p = np.clip(p, 0, 1)
                    p = p / p.sum()
                    counts = np.random.multinomial(N_bottleneck, p)
                    y = counts / N_bottleneck
                
                # 检查是否有物种占优
                if np.max(y) > 0.95:
                    break
            
            winners.append(np.argmax(y))
        
        # 绘制直方图
        ax = axes[idx]
        counts = np.bincount(winners, minlength=n_species)
        ax.bar(range(1, n_species + 1), counts, color='steelblue', 
              edgecolor='black', alpha=0.7)
        ax.set_xlabel("Species")
        ax.set_ylabel("Times as winner")
        ax.set_title(f"{reps} replicates")
        ax.set_xticks(range(1, n_species + 1))
        
        # 添加期望值线(均匀分布)
        expected = reps / n_species
        ax.axhline(expected, color='red', ls='--', alpha=0.7, 
                  label=f'Expected={expected:.1f}')
        ax.legend(fontsize=9)
    
    plt.suptitle("S6 — Winner Distribution Stability", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("figures/S6_winner_reliability.png", dpi=200)
    plt.close()
    print("✓ S6: Winner distribution reliability test completed")


# Run all tests
def run_all_reliability_tests():
    """运行所有可靠性测试"""
    print("\n" + "=" * 60)
    print("  Running BVSF Model Reliability Tests")
    print("=" * 60 + "\n")
    
    test_S1_LV_reliability()
    test_S2_neutral_reliability()
    test_S3_threshold_reliability()
    test_S4_coalescence_reliability()
    test_S5_coexistence_reliability()
    test_S6_winner_reliability()
    
    print("\n" + "=" * 60)
    print("  All reliability tests completed successfully!")
    print("  Results saved in ./figures/ directory")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all_reliability_tests()