"""
运行三条LLM生成路线

运行前请设置环境变量:
    Windows CMD:    set OPENAI_API_KEY=你的密钥
    Windows PS:     $env:OPENAI_API_KEY='你的密钥'
    Linux/Mac:      export OPENAI_API_KEY=你的密钥

此脚本将运行:
- 路线A: Two-stage CHOICE (直接生成CHOICE，低概率样本修复)
- 路线B: Utility + Softmax (生成效用值，tau温度参数扫描)
- 路线C: MNL + Residual (生成残差，与基线MNL结合)
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import pickle

# ============================================================
# 检查API Key
# ============================================================
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("=" * 60)
    print("错误: 未设置 OPENAI_API_KEY")
    print("=" * 60)
    print("请设置环境变量后重新运行:")
    print("  Windows CMD:    set OPENAI_API_KEY=你的密钥")
    print("  Windows PS:     $env:OPENAI_API_KEY='你的密钥'")
    print("  Linux/Mac:      export OPENAI_API_KEY=你的密钥")
    print("=" * 60)
    sys.exit(1)

print("=" * 60)
print("LLM合成数据生成")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"API Key: {api_key[:8]}...{api_key[-4:]}")
print("=" * 60)

# ============================================================
# 配置
# ============================================================
CONFIG = {
    "synth_N": 3000,          # 合成样本数量
    "p_unseen": 0.2,          # unseen组合比例
    "seed": 123,
    "model_stage1": "gpt-4o-mini",
    "model_stage2": "gpt-4o",
    "tau_list": [0.5, 1.0],   # Utility路线的tau值 (减少以节省费用)
    "lambda_list": [0.2, 0.5], # Residual路线的lambda值
}

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

# ============================================================
# 加载数据
# ============================================================
print("\n1. 加载数据...")

from swissmetro_llm.data import load_swissmetro, build_matrices, split_train_test_by_id
from swissmetro_llm.data.preprocessing import compute_scales_from_train, apply_scales
from swissmetro_llm.models import fit_mnl_step1, fit_mnl_step2, build_X_ind
from swissmetro_llm.models.mnl import get_x_names

df = load_swissmetro("swissmetro.dat")
df_train, df_test, train_ids, test_ids = split_train_test_by_id(df, test_size=0.2, seed=42)
print(f"训练集: {len(df_train)} 行, 测试集: {len(df_test)} 行")

train_data = build_matrices(df_train)
tt_scale, co_scale, he_scale = compute_scales_from_train(train_data)
train_data = apply_scales(train_data, tt_scale, co_scale, he_scale)

# ============================================================
# 估计MNL参数
# ============================================================
print("\n2. 估计MNL参数...")

res1 = fit_mnl_step1(train_data)
theta1 = res1.x
X_train = build_X_ind(train_data)
res2 = fit_mnl_step2(train_data, X_train, theta1=theta1, maxiter=20000, maxfun=20000)
theta2 = res2.x
print(f"MNL Step2: LL = {-res2.fun:.2f}, 参数数量 = {len(theta2)}")

# ============================================================
# 创建评估函数
# ============================================================
print("\n3. 创建评估函数...")

from swissmetro_llm.evaluation import score_with_baseline_mnl

x_names = get_x_names()
param_names = (
    ["B_TT", "B_CO", "B_HE", "B_SEATS", "ASC_SM", "ASC_CAR"]
    + [f"G_SM:{n}" for n in x_names]
    + [f"G_CAR:{n}" for n in x_names]
)
beta = dict(zip(param_names, theta2))
scales = {
    "tt_scale": tt_scale,
    "co_scale": co_scale,
    "he_scale": he_scale,
    "train_ids": train_ids,
}

def score_fn(df):
    return score_with_baseline_mnl(df, beta, scales)

# ============================================================
# 创建模板
# ============================================================
print("\n4. 创建模板...")

from swissmetro_llm.generation import create_templates

templates = create_templates(
    real_train=df_train,
    real_test=df_test,
    N=CONFIG["synth_N"],
    p_unseen=CONFIG["p_unseen"],
    seed=CONFIG["seed"]
)
print(f"创建了 {len(templates)} 个模板")

# 保存模板
templates.to_csv(output_dir / "templates.csv", index=False)

# ============================================================
# 路线A: Two-stage CHOICE
# ============================================================
print("\n" + "=" * 60)
print("5A. 路线A: Two-stage CHOICE")
print("=" * 60)

from swissmetro_llm.generation import generate_two_stage

jsonl_dir_a = output_dir / "route_a"
jsonl_dir_a.mkdir(exist_ok=True)

start_time = time.time()
try:
    syn_a = generate_two_stage(
        templates=templates,
        score_fn=score_fn,
        model_stage1=CONFIG["model_stage1"],
        model_stage2=CONFIG["model_stage2"],
        low_prob_threshold=0.01,
        jsonl_dir=str(jsonl_dir_a),
        seed=CONFIG["seed"],
        cot_stage1=False,
        cot_stage2=True
    )
    elapsed = time.time() - start_time
    print(f"\n路线A完成! 耗时: {elapsed/60:.1f}分钟, 样本数: {len(syn_a)}")
    syn_a.to_csv(output_dir / "syn_route_a.csv", index=False)
except Exception as e:
    print(f"\n路线A失败: {e}")
    syn_a = None

# ============================================================
# 路线B: Utility + Softmax
# ============================================================
print("\n" + "=" * 60)
print("5B. 路线B: Utility + Softmax")
print("=" * 60)

from swissmetro_llm.generation import generate_from_utilities_batch

syn_b_results = {}

for tau in CONFIG["tau_list"]:
    print(f"\n--- tau = {tau} ---")
    jsonl_path = str(output_dir / f"route_b_tau{tau}.jsonl")
    out_path = str(output_dir / f"route_b_tau{tau}_out.jsonl")

    try:
        start_time = time.time()
        syn_b = generate_from_utilities_batch(
            templates=templates,
            model=CONFIG["model_stage1"],
            tau=tau,
            jsonl_path=jsonl_path,
            out_path=out_path,
            seed=CONFIG["seed"],
            cot=False
        )
        elapsed = time.time() - start_time
        print(f"tau={tau}: 完成! 耗时: {elapsed/60:.1f}分钟, 样本数: {len(syn_b)}")
        syn_b.to_csv(output_dir / f"syn_route_b_tau{tau}.csv", index=False)
        syn_b_results[tau] = syn_b
    except Exception as e:
        print(f"tau={tau}: 失败 - {e}")

# ============================================================
# 路线C: MNL + Residual
# ============================================================
print("\n" + "=" * 60)
print("5C. 路线C: MNL + Residual")
print("=" * 60)

from swissmetro_llm.generation import generate_from_residual_batch

syn_c_results = {}

for lam in CONFIG["lambda_list"]:
    print(f"\n--- lambda = {lam} ---")
    jsonl_path = str(output_dir / f"route_c_lam{lam}.jsonl")
    out_path = str(output_dir / f"route_c_lam{lam}_out.jsonl")

    try:
        start_time = time.time()
        syn_c = generate_from_residual_batch(
            templates=templates,
            score_fn=score_fn,
            model=CONFIG["model_stage1"],
            lam=lam,
            jsonl_path=jsonl_path,
            out_path=out_path,
            seed=CONFIG["seed"],
            cot=False
        )
        elapsed = time.time() - start_time
        print(f"lambda={lam}: 完成! 耗时: {elapsed/60:.1f}分钟, 样本数: {len(syn_c)}")
        syn_c.to_csv(output_dir / f"syn_route_c_lam{lam}.csv", index=False)
        syn_c_results[lam] = syn_c
    except Exception as e:
        print(f"lambda={lam}: 失败 - {e}")

# ============================================================
# 评估
# ============================================================
print("\n" + "=" * 60)
print("6. 评估合成数据")
print("=" * 60)

from swissmetro_llm.evaluation import evaluate_one

eval_results = []

# 评估所有生成的数据集
all_syn = {}
if syn_a is not None:
    all_syn["route_a"] = syn_a
for tau, syn in syn_b_results.items():
    all_syn[f"route_b_tau{tau}"] = syn
for lam, syn in syn_c_results.items():
    all_syn[f"route_c_lam{lam}"] = syn

for label, syn_df in all_syn.items():
    print(f"\n评估: {label}")
    try:
        metrics = evaluate_one(
            synth_df=syn_df,
            beta=beta,
            scales=scales,
            real_train=df_train,
            real_test=df_test,
            label=label
        )
        metrics["label"] = label
        eval_results.append(metrics)
        print(f"  准确率: {metrics.get('accuracy', 'N/A'):.4f}" if metrics.get('accuracy') else "  准确率: N/A")
    except Exception as e:
        print(f"  评估失败: {e}")

if eval_results:
    eval_df = pd.DataFrame(eval_results)
    eval_df.to_csv(output_dir / "evaluation_results.csv", index=False)
    print(f"\n评估结果已保存: {output_dir / 'evaluation_results.csv'}")

# ============================================================
# 完成
# ============================================================
print("\n" + "=" * 60)
print("LLM生成完成!")
print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

print("\n生成的文件:")
for f in sorted(output_dir.glob("*.csv")):
    print(f"  - {f.name}")
