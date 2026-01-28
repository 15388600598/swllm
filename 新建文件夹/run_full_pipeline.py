"""
Swissmetro LLM 完整管线运行脚本

运行前请确保:
1. 已安装依赖: pip install -r requirements.txt
2. swissmetro.dat 文件在当前目录
3. 设置环境变量: set OPENAI_API_KEY=你的密钥

此脚本将运行:
1. MNL模型估计 (Step1 + Step2)
2. Bootstrap推断 (B=30)
3. 三条LLM生成路线 (A:Two-stage, B:Utility, C:Residual)
4. 参数稳定性分析
5. 导出所有结果CSV
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '.')

import numpy as np
import pandas as pd

# ============================================================
# 配置参数
# ============================================================
CONFIG = {
    # 数据路径
    "data_path": "swissmetro.dat",
    "output_dir": "./results",

    # Bootstrap配置
    "bootstrap_B": 30,
    "bootstrap_seed": 2025,

    # LLM生成配置
    "synth_N": 3000,  # 合成样本数量
    "p_unseen": 0.2,  # unseen组合比例
    "llm_seed": 123,

    # 模型
    "model_stage1": "gpt-4o-mini",  # 快速模型
    "model_stage2": "gpt-4o",       # 修复模型

    # Utility路线的tau参数
    "tau_list": [0.2, 0.5, 0.8, 1.0, 1.2],

    # Residual路线的lambda参数
    "lambda_list": [0.1, 0.2, 0.3, 0.5],

    # 稳定性分析ratio
    "stability_ratio": 0.5,
}

# ============================================================
# 创建输出目录
# ============================================================
output_dir = Path(CONFIG["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# 检查API Key
# ============================================================
def check_api_key():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("=" * 60)
        print("警告: 未设置 OPENAI_API_KEY")
        print("=" * 60)
        print("如需运行LLM生成，请设置环境变量:")
        print("  Windows CMD:    set OPENAI_API_KEY=你的密钥")
        print("  Windows PS:     $env:OPENAI_API_KEY='你的密钥'")
        print("  Linux/Mac:      export OPENAI_API_KEY=你的密钥")
        print("=" * 60)
        return False
    return True

# ============================================================
# 1. 数据加载
# ============================================================
def load_data():
    print("\n" + "=" * 60)
    print("1. 加载数据")
    print("=" * 60)

    from swissmetro_llm.data import load_swissmetro, build_matrices, split_train_test_by_id
    from swissmetro_llm.data.preprocessing import compute_scales_from_train, apply_scales

    df = load_swissmetro(CONFIG["data_path"])
    print(f"加载完成: {len(df)} 行, {df['ID'].nunique()} 个受访者")

    # 划分训练/测试集
    df_train, df_test, train_ids, test_ids = split_train_test_by_id(df, test_size=0.2, seed=42)
    print(f"训练集: {len(df_train)} 行, {len(train_ids)} 个受访者")
    print(f"测试集: {len(df_test)} 行, {len(test_ids)} 个受访者")

    # 构建矩阵
    train_data = build_matrices(df_train)
    test_data = build_matrices(df_test)

    # 计算缩放参数
    tt_scale, co_scale, he_scale = compute_scales_from_train(train_data)
    print(f"TT缩放: {tt_scale}")
    print(f"CO缩放: {co_scale}")
    print(f"HE缩放: {he_scale}")

    # 应用缩放
    train_data = apply_scales(train_data, tt_scale, co_scale, he_scale)
    test_data = apply_scales(test_data, tt_scale, co_scale, he_scale)

    return {
        "df": df,
        "df_train": df_train,
        "df_test": df_test,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "train_data": train_data,
        "test_data": test_data,
        "tt_scale": tt_scale,
        "co_scale": co_scale,
        "he_scale": he_scale,
    }

# ============================================================
# 2. MNL模型估计
# ============================================================
def fit_mnl_models(data):
    print("\n" + "=" * 60)
    print("2. MNL模型估计")
    print("=" * 60)

    from swissmetro_llm.models import fit_mnl_step1, predict_step1, fit_mnl_step2, predict_step2
    from swissmetro_llm.models import build_X_ind
    from swissmetro_llm.models.utils import accuracy

    train_data = data["train_data"]
    test_data = data["test_data"]

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

    # Step 2: 扩展MNL
    print("\n--- Step 2: 扩展MNL (含个体特征交互项) ---")
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

    acc_train = accuracy(P_train, train_data['y'])
    acc_test = accuracy(P_test, test_data['y'])

    print(f"\n训练集准确率: {acc_train:.4f}")
    print(f"测试集准确率: {acc_test:.4f}")

    # 保存Step1和Step2参数
    step1_df = pd.DataFrame({
        "param": param_names1,
        "coef": theta1
    })
    step1_df.to_csv(output_dir / "mnl_step1_params.csv", index=False)

    return {
        "theta1": theta1,
        "theta2": theta2,
        "res1": res1,
        "res2": res2,
        "X_train": X_train,
        "X_test": X_test,
        "acc_train": acc_train,
        "acc_test": acc_test,
    }

# ============================================================
# 3. Bootstrap推断
# ============================================================
def run_bootstrap(data, mnl_results):
    print("\n" + "=" * 60)
    print("3. Bootstrap推断")
    print("=" * 60)

    from swissmetro_llm.models import cluster_bootstrap_thetas, compute_bootstrap_se, compute_bootstrap_ci
    from swissmetro_llm.models.mnl import get_step2_main_param_names

    B = CONFIG["bootstrap_B"]
    print(f"运行 {B} 次 Bootstrap...")

    thetas_boot = cluster_bootstrap_thetas(
        data["df_train"],
        data["train_ids"],
        data["tt_scale"],
        data["co_scale"],
        data["he_scale"],
        theta_init=mnl_results["theta2"],
        B=B,
        seed=CONFIG["bootstrap_seed"],
        verbose_every=5
    )

    print(f"\n成功完成 {len(thetas_boot)} / {B} 次")

    if len(thetas_boot) == 0:
        print("警告: Bootstrap全部失败!")
        return None

    # 计算标准误和置信区间
    se_boot = compute_bootstrap_se(thetas_boot)
    ci_lower, ci_upper = compute_bootstrap_ci(thetas_boot, alpha=0.05)

    # 计算p-value (两侧)
    theta_base = mnl_results["theta2"]
    p_boot = np.zeros(len(theta_base))
    for i in range(len(theta_base)):
        if se_boot[i] > 0:
            z = theta_base[i] / se_boot[i]
            p_boot[i] = 2 * (1 - np.minimum(np.abs(z), 10) / 10)  # 简化p-value
        else:
            p_boot[i] = 1.0

    # 保存Bootstrap结果
    param_names = get_step2_main_param_names(K=18)
    boot_df = pd.DataFrame({
        "param": param_names[:len(theta_base)],
        "coef": theta_base,
        "boot_se": se_boot,
        "ci_2.5": ci_lower,
        "ci_97.5": ci_upper,
        "p_boot": p_boot,
    })
    boot_df.to_csv(output_dir / "bootstrap_results.csv", index=False)
    print(f"\nBootstrap结果已保存: {output_dir / 'bootstrap_results.csv'}")

    # 打印前10个参数
    print("\n前10个参数的Bootstrap结果:")
    print("-" * 70)
    print(f"{'参数':<15} {'估计值':>10} {'SE':>10} {'CI 2.5%':>10} {'CI 97.5%':>10}")
    print("-" * 70)
    for i in range(min(10, len(theta_base))):
        print(f"{param_names[i]:<15} {theta_base[i]:>10.4f} {se_boot[i]:>10.4f} {ci_lower[i]:>10.4f} {ci_upper[i]:>10.4f}")

    return {
        "thetas_boot": thetas_boot,
        "se_boot": se_boot,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_boot": p_boot,
    }

# ============================================================
# 4. 创建评估函数
# ============================================================
def create_score_function(data, mnl_results):
    """创建基准MNL打分函数"""
    from swissmetro_llm.evaluation import score_with_baseline_mnl
    from swissmetro_llm.models.mnl import get_step2_main_param_names

    param_names = get_step2_main_param_names(K=18)
    beta = dict(zip(param_names, mnl_results["theta2"]))
    scales = {
        "tt_scale": data["tt_scale"],
        "co_scale": data["co_scale"],
        "he_scale": data["he_scale"],
        "train_ids": data["train_ids"],
    }

    def score_fn(df):
        return score_with_baseline_mnl(df, beta, scales)

    return score_fn, beta, scales

# ============================================================
# 5. LLM生成 - 路线A: Two-stage CHOICE
# ============================================================
def run_route_a(data, score_fn):
    print("\n" + "=" * 60)
    print("5A. LLM生成 - 路线A: Two-stage CHOICE")
    print("=" * 60)

    from swissmetro_llm.generation import create_templates, generate_two_stage

    # 创建模板
    templates = create_templates(
        real_train=data["df_train"],
        real_test=data["df_test"],
        N=CONFIG["synth_N"],
        p_unseen=CONFIG["p_unseen"],
        seed=CONFIG["llm_seed"]
    )
    print(f"创建了 {len(templates)} 个模板")

    # 创建临时目录
    jsonl_dir = output_dir / "route_a"
    jsonl_dir.mkdir(exist_ok=True)

    # 运行Two-stage生成
    print("\n开始Two-stage生成...")
    start_time = time.time()

    syn_final = generate_two_stage(
        templates=templates,
        score_fn=score_fn,
        model_stage1=CONFIG["model_stage1"],
        model_stage2=CONFIG["model_stage2"],
        low_prob_threshold=0.01,
        jsonl_dir=str(jsonl_dir),
        seed=CONFIG["llm_seed"],
        cot_stage1=False,
        cot_stage2=True
    )

    elapsed = time.time() - start_time
    print(f"\n路线A完成! 耗时: {elapsed/60:.1f}分钟")
    print(f"生成样本数: {len(syn_final)}")

    # 保存
    syn_final.to_csv(output_dir / "syn_route_a_twostage.csv", index=False)

    return syn_final, templates

# ============================================================
# 6. LLM生成 - 路线B: Utility + Softmax
# ============================================================
def run_route_b(templates):
    print("\n" + "=" * 60)
    print("5B. LLM生成 - 路线B: Utility + Softmax")
    print("=" * 60)

    from swissmetro_llm.generation import generate_from_utilities_batch

    results = {}

    for tau in CONFIG["tau_list"]:
        print(f"\n--- tau = {tau} ---")

        jsonl_path = str(output_dir / f"route_b_tau{tau}.jsonl")
        out_path = str(output_dir / f"route_b_tau{tau}_out.jsonl")

        syn_u = generate_from_utilities_batch(
            templates=templates,
            model=CONFIG["model_stage1"],
            tau=tau,
            jsonl_path=jsonl_path,
            out_path=out_path,
            seed=CONFIG["llm_seed"],
            cot=False
        )

        print(f"tau={tau}: 生成 {len(syn_u)} 样本")

        # 保存
        syn_u.to_csv(output_dir / f"syn_route_b_tau{tau}.csv", index=False)
        results[tau] = syn_u

    return results

# ============================================================
# 7. LLM生成 - 路线C: MNL + Residual
# ============================================================
def run_route_c(templates, score_fn):
    print("\n" + "=" * 60)
    print("5C. LLM生成 - 路线C: MNL + Residual")
    print("=" * 60)

    from swissmetro_llm.generation import generate_from_residual_batch

    results = {}

    for lam in CONFIG["lambda_list"]:
        print(f"\n--- lambda = {lam} ---")

        jsonl_path = str(output_dir / f"route_c_lam{lam}.jsonl")
        out_path = str(output_dir / f"route_c_lam{lam}_out.jsonl")

        syn_r = generate_from_residual_batch(
            templates=templates,
            score_fn=score_fn,
            model=CONFIG["model_stage1"],
            lam=lam,
            jsonl_path=jsonl_path,
            out_path=out_path,
            seed=CONFIG["llm_seed"],
            cot=False
        )

        print(f"lambda={lam}: 生成 {len(syn_r)} 样本")

        # 保存
        syn_r.to_csv(output_dir / f"syn_route_c_lam{lam}.csv", index=False)
        results[lam] = syn_r

    return results

# ============================================================
# 8. 评估合成数据
# ============================================================
def evaluate_synthetic(data, beta, scales, syn_datasets):
    print("\n" + "=" * 60)
    print("6. 评估合成数据")
    print("=" * 60)

    from swissmetro_llm.evaluation import evaluate_one, downstream_metrics

    eval_results = []

    for label, syn_df in syn_datasets.items():
        print(f"\n--- 评估: {label} ---")

        metrics = evaluate_one(
            synth_df=syn_df,
            beta=beta,
            scales=scales,
            real_train=data["df_train"],
            real_test=data["df_test"],
            label=label
        )

        print(f"  可行样本: {metrics.get('feasible_n', 'N/A')}")
        print(f"  准确率: {metrics.get('accuracy', 'N/A'):.4f}" if metrics.get('accuracy') else "  准确率: N/A")
        print(f"  平均选中概率: {metrics.get('avg_P_chosen', 'N/A'):.4f}" if metrics.get('avg_P_chosen') else "")

        metrics["label"] = label
        eval_results.append(metrics)

    # 保存评估结果
    eval_df = pd.DataFrame(eval_results)
    eval_df.to_csv(output_dir / "evaluation_results.csv", index=False)
    print(f"\n评估结果已保存: {output_dir / 'evaluation_results.csv'}")

    return eval_df

# ============================================================
# 9. 参数稳定性分析
# ============================================================
def run_stability_analysis(data, boot_results, syn_datasets):
    print("\n" + "=" * 60)
    print("7. 参数稳定性分析")
    print("=" * 60)

    from swissmetro_llm.stability import stability_analysis, format_stability_for_report

    # 选择一个合成数据集进行稳定性分析
    # 默认使用路线A的结果,如果没有则用第一个可用的
    if "route_a" in syn_datasets:
        syn_for_stability = syn_datasets["route_a"]
        label = "route_a"
    else:
        label = list(syn_datasets.keys())[0]
        syn_for_stability = syn_datasets[label]

    print(f"\n使用 {label} 进行稳定性分析")
    print(f"合成数据样本数: {len(syn_for_stability)}")

    # 运行稳定性分析
    stab_df, metadata = stability_analysis(
        real_train=data["df_train"],
        syn_df=syn_for_stability,
        tt_scale=data["tt_scale"],
        co_scale=data["co_scale"],
        he_scale=data["he_scale"],
        theta_init=None,  # 从头估计
        boot_se=boot_results["se_boot"] if boot_results else None,
        ci_lower=boot_results["ci_lower"] if boot_results else None,
        ci_upper=boot_results["ci_upper"] if boot_results else None,
        p_boot=boot_results["p_boot"] if boot_results else None,
        ratio=CONFIG["stability_ratio"],
        seed=CONFIG["llm_seed"],
        verbose=True
    )

    # 格式化并保存
    summary = format_stability_for_report(
        stab_df,
        metadata,
        output_path=str(output_dir / "param_stability.csv"),
        top_n=15
    )

    print("\n" + summary)

    return stab_df, metadata

# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("Swissmetro LLM 完整管线")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 检查API Key
    has_api_key = check_api_key()

    # 1. 加载数据
    data = load_data()

    # 2. MNL估计
    mnl_results = fit_mnl_models(data)

    # 3. Bootstrap
    boot_results = run_bootstrap(data, mnl_results)

    # 创建评估函数
    score_fn, beta, scales = create_score_function(data, mnl_results)

    # 存储所有合成数据集
    syn_datasets = {}
    templates = None

    # 4-6. LLM生成 (需要API Key)
    if has_api_key:
        try:
            # 路线A
            syn_a, templates = run_route_a(data, score_fn)
            syn_datasets["route_a"] = syn_a

            # 路线B
            syn_b_results = run_route_b(templates)
            for tau, syn in syn_b_results.items():
                syn_datasets[f"route_b_tau{tau}"] = syn

            # 路线C
            syn_c_results = run_route_c(templates, score_fn)
            for lam, syn in syn_c_results.items():
                syn_datasets[f"route_c_lam{lam}"] = syn

        except Exception as e:
            print(f"\nLLM生成出错: {e}")
            print("继续使用已有数据...")
    else:
        print("\n跳过LLM生成步骤(未设置API Key)")

        # 创建模拟数据用于演示
        from swissmetro_llm.generation import create_templates
        templates = create_templates(
            real_train=data["df_train"],
            real_test=data["df_test"],
            N=500,
            p_unseen=0.2,
            seed=CONFIG["llm_seed"]
        )

        # 用随机选择模拟LLM输出
        rng = np.random.default_rng(42)
        templates_demo = templates.copy()
        templates_demo["CHOICE"] = rng.integers(1, 4, size=len(templates))
        syn_datasets["demo_random"] = templates_demo

    # 7. 评估
    if syn_datasets:
        eval_df = evaluate_synthetic(data, beta, scales, syn_datasets)

    # 8. 稳定性分析
    if syn_datasets:
        stab_df, metadata = run_stability_analysis(data, boot_results, syn_datasets)

    # 9. 完成
    print("\n" + "=" * 60)
    print("完整管线运行完成!")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结果目录: {output_dir.absolute()}")
    print("=" * 60)

    print("\n生成的文件:")
    for f in sorted(output_dir.glob("*.csv")):
        print(f"  - {f.name}")

if __name__ == "__main__":
    main()
