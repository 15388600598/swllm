"""
Swissmetro LLM 包使用示例

运行前请确保:
1. 已安装依赖: pip install -r requirements.txt
2. swissmetro.dat 文件在当前目录
3. 如需LLM生成，设置环境变量: set OPENAI_API_KEY=你的密钥
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd

# ============================================================
# 1. 数据加载
# ============================================================
print("=" * 60)
print("1. 加载数据")
print("=" * 60)

from swissmetro_llm.data import load_swissmetro, build_matrices, split_train_test_by_id
from swissmetro_llm.data.preprocessing import compute_scales_from_train, apply_scales

# 加载数据
DATA_PATH = "swissmetro.dat"
df = load_swissmetro(DATA_PATH)
print(f"加载完成: {len(df)} 行, {df['ID'].nunique()} 个受访者")

# 划分训练/测试集
df_train, df_test, train_ids, test_ids = split_train_test_by_id(df, test_size=0.2, seed=42)
print(f"训练集: {len(df_train)} 行, {len(train_ids)} 个受访者")
print(f"测试集: {len(df_test)} 行, {len(test_ids)} 个受访者")

# ============================================================
# 2. 构建矩阵并计算缩放参数
# ============================================================
print("\n" + "=" * 60)
print("2. 数据预处理")
print("=" * 60)

train_data = build_matrices(df_train)
test_data = build_matrices(df_test)

# 从训练数据计算缩放参数
tt_scale, co_scale, he_scale = compute_scales_from_train(train_data)
print(f"TT缩放: {tt_scale}")
print(f"CO缩放: {co_scale}")
print(f"HE缩放: {he_scale}")

# 应用缩放
train_data = apply_scales(train_data, tt_scale, co_scale, he_scale)
test_data = apply_scales(test_data, tt_scale, co_scale, he_scale)

# ============================================================
# 3. MNL模型估计
# ============================================================
print("\n" + "=" * 60)
print("3. MNL模型估计")
print("=" * 60)

from swissmetro_llm.models import fit_mnl_step1, predict_step1, fit_mnl_step2, predict_step2
from swissmetro_llm.models import build_X_ind
from swissmetro_llm.models.utils import accuracy, neg_loglike_from_P

# Step 1: 基础MNL (6个参数)
print("\n--- Step 1: 基础MNL ---")
res1 = fit_mnl_step1(train_data)
print(f"收敛: {res1.success}")
print(f"负对数似然: {res1.fun:.2f}")

theta1 = res1.x
param_names1 = ["B_TT", "B_CO", "B_HE", "B_SEATS", "ASC_SM", "ASC_CAR"]
print("\n参数估计:")
for name, val in zip(param_names1, theta1):
    print(f"  {name:12s}: {val:8.4f}")

# Step 2: 扩展MNL (含个体特征交互项)
print("\n--- Step 2: 扩展MNL ---")
X_train = build_X_ind(train_data)
X_test = build_X_ind(test_data)

res2 = fit_mnl_step2(train_data, X_train, theta1=theta1, maxiter=20000, maxfun=20000)
print(f"收敛: {res2.success}")
print(f"负对数似然: {res2.fun:.2f}")
print(f"参数数量: {len(res2.x)}")

theta2 = res2.x

# 评估
P_train = predict_step2(theta2, train_data, X_train)
P_test = predict_step2(theta2, test_data, X_test)

print(f"\n训练集准确率: {accuracy(P_train, train_data['y']):.3f}")
print(f"测试集准确率: {accuracy(P_test, test_data['y']):.3f}")

# ============================================================
# 4. Bootstrap推断 (可选，耗时较长)
# ============================================================
print("\n" + "=" * 60)
print("4. Bootstrap推断 (使用少量样本演示)")
print("=" * 60)

from swissmetro_llm.models import cluster_bootstrap_thetas, compute_bootstrap_se

# 为了演示，只运行5次bootstrap
B = 5
print(f"运行 {B} 次 Bootstrap (实际研究建议 B=30+)...")

thetas_boot = cluster_bootstrap_thetas(
    df_train,
    train_ids,
    tt_scale, co_scale, he_scale,
    theta_init=theta2,
    B=B,
    seed=2025,
    verbose_every=2
)

if len(thetas_boot) > 0:
    se_boot = compute_bootstrap_se(thetas_boot)
    print(f"\n成功完成 {len(thetas_boot)} / {B} 次")
    print("\n前6个参数的Bootstrap标准误:")
    for name, val, se in zip(param_names1, theta2[:6], se_boot[:6]):
        z = val / se if se > 0 else 0
        print(f"  {name:12s}: {val:8.4f} (SE: {se:.4f}, z: {z:.2f})")

# ============================================================
# 5. 评估框架演示
# ============================================================
print("\n" + "=" * 60)
print("5. 评估框架演示")
print("=" * 60)

from swissmetro_llm.evaluation import feasibility_min, diversity_report
from swissmetro_llm.evaluation import score_with_baseline_mnl, downstream_metrics
from swissmetro_llm.models.mnl import get_step2_main_param_names

# 准备beta字典
param_names = get_step2_main_param_names(K=18)
beta = dict(zip(param_names, theta2))
scales = {
    "tt_scale": tt_scale,
    "co_scale": co_scale,
    "he_scale": he_scale,
    "train_ids": train_ids,
}

# 评估训练数据
print("\n--- 训练数据评估 ---")
train_scored = score_with_baseline_mnl(df_train, beta, scales)
train_metrics = downstream_metrics(train_scored)
print(f"  平均选中概率: {train_metrics['avg_P_chosen']:.4f}")
print(f"  准确率: {train_metrics['accuracy']:.4f}")

# 评估测试数据
print("\n--- 测试数据评估 ---")
test_scored = score_with_baseline_mnl(df_test, beta, scales)
test_metrics = downstream_metrics(test_scored)
print(f"  平均选中概率: {test_metrics['avg_P_chosen']:.4f}")
print(f"  准确率: {test_metrics['accuracy']:.4f}")

# ============================================================
# 6. 合成数据生成 (需要OpenAI API)
# ============================================================
print("\n" + "=" * 60)
print("6. 合成数据生成")
print("=" * 60)

import os
if os.environ.get("OPENAI_API_KEY"):
    print("检测到OPENAI_API_KEY，可以运行LLM生成")
    print("(此示例跳过实际API调用)")

    # 创建模板演示
    from swissmetro_llm.generation import create_templates

    templates = create_templates(
        real_train=df_train,
        real_test=df_test,
        N=100,  # 生成100个模板
        p_unseen=0.2,
        seed=123
    )
    print(f"创建了 {len(templates)} 个模板")
    print(f"模板列: {list(templates.columns)[:8]}...")

    # 要实际运行生成，取消下面的注释:
    # from swissmetro_llm.generation import generate_from_utilities_batch
    # syn_u = generate_from_utilities_batch(
    #     templates,
    #     model="gpt-4o-mini",
    #     tau=1.0,
    #     seed=123
    # )
else:
    print("未设置OPENAI_API_KEY")
    print("如需运行LLM生成，请设置环境变量:")
    print("  Windows CMD:  set OPENAI_API_KEY=你的密钥")
    print("  Windows PowerShell:  $env:OPENAI_API_KEY='你的密钥'")
    print("  Linux/Mac:  export OPENAI_API_KEY=你的密钥")

# ============================================================
# 7. 参数稳定性分析演示 (real vs real+synthetic)
# ============================================================
print("\n" + "=" * 60)
print("7. 参数稳定性分析")
print("=" * 60)

from swissmetro_llm.stability import (
    make_augmented,
    stability_analysis,
    format_stability_for_report
)

# 创建模拟的合成数据（实际使用时应该是LLM生成的）
# 这里用真实数据的随机打乱版本来演示
print("\n创建模拟合成数据用于演示...")
syn_demo = df_train.sample(n=500, replace=True, random_state=999).copy()
syn_demo["ID"] = np.arange(2_000_000, 2_000_000 + len(syn_demo))
# 随机调整一些CHOICE来模拟LLM输出
rng = np.random.default_rng(42)
flip_mask = rng.random(len(syn_demo)) < 0.1  # 10%的选择被随机改变
syn_demo.loc[flip_mask, "CHOICE"] = rng.integers(1, 4, size=flip_mask.sum())
print(f"模拟合成数据: {len(syn_demo)} 行")

# 运行稳定性分析
print("\n运行参数稳定性分析...")
stab_df, metadata = stability_analysis(
    real_train=df_train,
    syn_df=syn_demo,
    tt_scale=tt_scale,
    co_scale=co_scale,
    he_scale=he_scale,
    theta_init=theta2,  # 用之前估计的参数作为初始值
    ratio=0.5,  # 使用50%的合成数据
    seed=123,
    verbose=True
)

# 输出结果摘要
summary = format_stability_for_report(
    stab_df,
    metadata,
    output_path="param_stability_demo.csv",
    top_n=10
)
print("\n" + summary)

# 显示关键统计
print("\n--- 稳定性关键统计 ---")
print(f"参数总数: {len(stab_df)}")
print(f"最大绝对变化: {stab_df['abs_diff'].max():.4f} ({stab_df.iloc[0]['param']})")
print(f"平均绝对变化: {stab_df['abs_diff'].mean():.4f}")
print(f"中位数绝对变化: {stab_df['abs_diff'].median():.4f}")

# ============================================================
# 完成
# ============================================================
print("\n" + "=" * 60)
print("示例运行完成!")
print("=" * 60)
