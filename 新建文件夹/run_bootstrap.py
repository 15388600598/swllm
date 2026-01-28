"""
运行Bootstrap推断 (B=30)

这个脚本会:
1. 加载数据和MNL参数
2. 运行30次聚类Bootstrap
3. 计算标准误和置信区间
4. 保存结果到CSV
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime

print("=" * 60)
print("Bootstrap推断")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# ============================================================
# 1. 加载数据
# ============================================================
print("\n1. 加载数据...")

from swissmetro_llm.data import load_swissmetro, build_matrices, split_train_test_by_id
from swissmetro_llm.data.preprocessing import compute_scales_from_train, apply_scales

df = load_swissmetro("swissmetro.dat")
df_train, df_test, train_ids, test_ids = split_train_test_by_id(df, test_size=0.2, seed=42)
print(f"训练集: {len(df_train)} 行, {len(train_ids)} 个受访者")

train_data = build_matrices(df_train)
tt_scale, co_scale, he_scale = compute_scales_from_train(train_data)
train_data = apply_scales(train_data, tt_scale, co_scale, he_scale)

# ============================================================
# 2. 运行MNL Step2获取初始参数
# ============================================================
print("\n2. 估计MNL Step2参数...")

from swissmetro_llm.models import fit_mnl_step1, fit_mnl_step2, build_X_ind

res1 = fit_mnl_step1(train_data)
theta1 = res1.x
print(f"Step1 完成: LL = {-res1.fun:.2f}")

X_train = build_X_ind(train_data)
res2 = fit_mnl_step2(train_data, X_train, theta1=theta1, maxiter=20000, maxfun=20000)
theta2 = res2.x
print(f"Step2 完成: LL = {-res2.fun:.2f}, 参数数量 = {len(theta2)}")

# ============================================================
# 3. 运行Bootstrap
# ============================================================
print("\n3. 运行Bootstrap (B=30)...")
print("=" * 60)

from swissmetro_llm.models import cluster_bootstrap_thetas, compute_bootstrap_se, compute_bootstrap_ci

B = 30
thetas_boot = cluster_bootstrap_thetas(
    df_train,
    train_ids,
    tt_scale, co_scale, he_scale,
    theta_init=theta2,
    B=B,
    seed=2025,
    verbose_every=3,
    maxiter=20000,
    maxfun=20000
)

print(f"\n成功完成 {len(thetas_boot)} / {B} 次")

# ============================================================
# 4. 计算统计量
# ============================================================
if len(thetas_boot) > 0:
    print("\n4. 计算Bootstrap统计量...")

    se_boot = compute_bootstrap_se(thetas_boot)
    ci_lower, ci_upper = compute_bootstrap_ci(thetas_boot, alpha=0.05)

    # 计算z值和p值
    z_values = np.zeros(len(theta2))
    p_values = np.ones(len(theta2))

    for i in range(len(theta2)):
        if se_boot[i] > 1e-10:
            z_values[i] = theta2[i] / se_boot[i]
            # 双侧p值 (近似正态)
            from scipy import stats
            p_values[i] = 2 * (1 - stats.norm.cdf(abs(z_values[i])))

    # ============================================================
    # 5. 保存结果
    # ============================================================
    print("\n5. 保存结果...")

    os.makedirs("results", exist_ok=True)

    from swissmetro_llm.models.mnl import get_x_names
    # X_ind有19列（含FIRST），参数数量为 6 + 2*19 = 44
    # 直接生成完整的参数名
    x_names = get_x_names()  # 19个特征名: FIRST, MALE, AGE_2-6, INCOME_1-4, PURPOSE_2-9
    param_names = (
        ["B_TT", "B_CO", "B_HE", "B_SEATS", "ASC_SM", "ASC_CAR"]
        + [f"G_SM:{n}" for n in x_names]
        + [f"G_CAR:{n}" for n in x_names]
    )
    print(f"参数名数量: {len(param_names)}, theta2长度: {len(theta2)}")

    boot_df = pd.DataFrame({
        "param": param_names[:len(theta2)],
        "coef": theta2,
        "boot_se": se_boot,
        "z": z_values,
        "p_value": p_values,
        "ci_2.5": ci_lower,
        "ci_97.5": ci_upper,
    })

    boot_df.to_csv("results/bootstrap_B30_results.csv", index=False)
    print(f"已保存: results/bootstrap_B30_results.csv")

    # 保存原始theta矩阵
    thetas_arr = np.array(thetas_boot)
    np.save("results/bootstrap_thetas.npy", thetas_arr)
    print(f"已保存: results/bootstrap_thetas.npy")

    # 打印结果摘要
    print("\n" + "=" * 70)
    print("Bootstrap结果摘要 (前15个参数)")
    print("=" * 70)
    print(f"{'参数':<20} {'估计值':>10} {'SE':>10} {'z':>8} {'p值':>10}")
    print("-" * 70)

    for i in range(min(15, len(theta2))):
        sig = ""
        if p_values[i] < 0.001:
            sig = "***"
        elif p_values[i] < 0.01:
            sig = "**"
        elif p_values[i] < 0.05:
            sig = "*"
        print(f"{param_names[i]:<20} {theta2[i]:>10.4f} {se_boot[i]:>10.4f} {z_values[i]:>8.2f} {p_values[i]:>10.4f} {sig}")

    print("-" * 70)
    print("显著性: *** p<0.001, ** p<0.01, * p<0.05")

else:
    print("\n警告: Bootstrap全部失败!")

# ============================================================
# 完成
# ============================================================
print("\n" + "=" * 60)
print("Bootstrap推断完成!")
print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)
