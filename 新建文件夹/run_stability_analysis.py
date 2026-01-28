"""
参数稳定性分析

此脚本运行参数稳定性分析 (real vs real+synthetic)
如果没有LLM生成的数据，将使用模拟数据

使用方法:
    python run_stability_analysis.py [synthetic_data.csv]
"""

import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '.')

import numpy as np
import pandas as pd

print("=" * 60)
print("参数稳定性分析")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# ============================================================
# 配置
# ============================================================
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

# 检查是否有命令行指定的合成数据文件
syn_data_path = None
if len(sys.argv) > 1:
    syn_data_path = sys.argv[1]
    if not os.path.exists(syn_data_path):
        print(f"警告: 文件不存在 {syn_data_path}")
        syn_data_path = None

# ============================================================
# 加载数据
# ============================================================
print("\n1. 加载数据...")

from swissmetro_llm.data import load_swissmetro, build_matrices, split_train_test_by_id
from swissmetro_llm.data.preprocessing import compute_scales_from_train, apply_scales

df = load_swissmetro("swissmetro.dat")
df_train, df_test, train_ids, test_ids = split_train_test_by_id(df, test_size=0.2, seed=42)
print(f"训练集: {len(df_train)} 行, {len(train_ids)} 个受访者")
print(f"测试集: {len(df_test)} 行, {len(test_ids)} 个受访者")

train_data = build_matrices(df_train)
tt_scale, co_scale, he_scale = compute_scales_from_train(train_data)
train_data = apply_scales(train_data, tt_scale, co_scale, he_scale)

# ============================================================
# 获取基准MNL参数
# ============================================================
print("\n2. 估计MNL参数...")

from swissmetro_llm.models import fit_mnl_step1, fit_mnl_step2, build_X_ind

res1 = fit_mnl_step1(train_data)
theta1 = res1.x
X_train = build_X_ind(train_data)
res2 = fit_mnl_step2(train_data, X_train, theta1=theta1, maxiter=20000, maxfun=20000)
theta2 = res2.x
print(f"MNL Step2: LL = {-res2.fun:.2f}, 参数数量 = {len(theta2)}")

# ============================================================
# 加载Bootstrap结果 (如果有)
# ============================================================
boot_se = None
ci_lower = None
ci_upper = None
p_boot = None

boot_results_path = output_dir / "bootstrap_B30_results.csv"
if boot_results_path.exists():
    print("\n3. 加载Bootstrap结果...")
    boot_df = pd.read_csv(boot_results_path)
    boot_se = boot_df["boot_se"].to_numpy()
    ci_lower = boot_df["ci_2.5"].to_numpy()
    ci_upper = boot_df["ci_97.5"].to_numpy()
    p_boot = boot_df["p_value"].to_numpy()
    print(f"加载了 {len(boot_se)} 个参数的Bootstrap结果")
else:
    print("\n3. 未找到Bootstrap结果 (跳过)")

# ============================================================
# 加载或创建合成数据
# ============================================================
print("\n4. 准备合成数据...")

if syn_data_path and os.path.exists(syn_data_path):
    # 使用指定的合成数据
    syn_df = pd.read_csv(syn_data_path)
    print(f"加载合成数据: {syn_data_path} ({len(syn_df)} 行)")
else:
    # 检查results目录下是否有LLM生成的数据
    syn_files = list(output_dir.glob("syn_route_*.csv"))
    if syn_files:
        # 使用第一个找到的合成数据
        syn_data_path = syn_files[0]
        syn_df = pd.read_csv(syn_data_path)
        print(f"使用已有合成数据: {syn_data_path.name} ({len(syn_df)} 行)")
    else:
        # 创建模拟合成数据
        print("创建模拟合成数据用于演示...")
        rng = np.random.default_rng(42)
        syn_df = df_train.sample(n=1000, replace=True, random_state=999).copy()
        syn_df["ID"] = np.arange(2_000_000, 2_000_000 + len(syn_df))
        # 随机调整10%的选择
        flip_mask = rng.random(len(syn_df)) < 0.1
        syn_df.loc[flip_mask, "CHOICE"] = rng.integers(1, 4, size=flip_mask.sum())
        print(f"模拟合成数据: {len(syn_df)} 行 (10%选择被随机修改)")

# ============================================================
# 运行稳定性分析
# ============================================================
print("\n5. 运行稳定性分析...")

from swissmetro_llm.stability import stability_analysis, format_stability_for_report

# 分析多个ratio
ratios = [0.25, 0.5, 1.0]
all_results = []

for ratio in ratios:
    print(f"\n--- ratio = {ratio} ---")

    stab_df, metadata = stability_analysis(
        real_train=df_train,
        syn_df=syn_df,
        tt_scale=tt_scale,
        co_scale=co_scale,
        he_scale=he_scale,
        theta_init=theta2,
        boot_se=boot_se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_boot=p_boot,
        ratio=ratio,
        seed=123,
        verbose=True
    )

    # 保存每个ratio的结果
    stab_df["ratio"] = ratio
    all_results.append(stab_df)

    # 格式化输出
    summary = format_stability_for_report(
        stab_df,
        metadata,
        output_path=str(output_dir / f"param_stability_ratio{ratio}.csv"),
        top_n=10
    )

# 合并所有结果
all_df = pd.concat(all_results, ignore_index=True)
all_df.to_csv(output_dir / "param_stability_all_ratios.csv", index=False)

# ============================================================
# 汇总报告
# ============================================================
print("\n" + "=" * 60)
print("稳定性分析汇总")
print("=" * 60)

# 按ratio计算平均绝对变化
summary_stats = []
for ratio in ratios:
    df_r = all_df[all_df["ratio"] == ratio].reset_index(drop=True)
    if len(df_r) > 0:
        max_idx = df_r["abs_diff"].idxmax()
        max_param = df_r.loc[max_idx, "param"]
    else:
        max_param = "N/A"
    stats = {
        "ratio": ratio,
        "mean_abs_diff": df_r["abs_diff"].mean() if len(df_r) > 0 else 0,
        "median_abs_diff": df_r["abs_diff"].median() if len(df_r) > 0 else 0,
        "max_abs_diff": df_r["abs_diff"].max() if len(df_r) > 0 else 0,
        "max_param": max_param,
    }
    summary_stats.append(stats)

summary_df = pd.DataFrame(summary_stats)
print("\n各ratio下的参数变化统计:")
print(summary_df.to_string(index=False))

summary_df.to_csv(output_dir / "stability_summary.csv", index=False)

# ============================================================
# 完成
# ============================================================
print("\n" + "=" * 60)
print("稳定性分析完成!")
print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

print("\n生成的文件:")
for f in sorted(output_dir.glob("*stability*.csv")):
    print(f"  - {f.name}")
