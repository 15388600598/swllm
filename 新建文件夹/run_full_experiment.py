"""
Swissmetro LLM 完整实验脚本

运行前请设置环境变量:
    Windows CMD:    set OPENAI_API_KEY=你的密钥
    Windows PS:     $env:OPENAI_API_KEY='你的密钥'

实验内容:
1. MNL模型估计 (Step1 + Step2)
2. Bootstrap推断 (B=200)
3. 三条LLM生成路线 × 3个模型对比
4. 评估所有合成数据
5. 参数稳定性分析
6. 导出汇总报告
"""
import os
print(os.environ.get("OPENAI_API_KEY"))

import sys
import os
import time
import pickle
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '.')

import numpy as np
import pandas as pd

# ============================================================
# 实验配置
# ============================================================
CONFIG = {
    # 数据路径
    "data_path": "swissmetro.dat",

    # Bootstrap配置
    "bootstrap_B": 200,
    "bootstrap_seed": 2025,

    # 合成数据配置
    "synth_N": 3000,
    "p_unseen": 0.2,
    "seed": 123,

    # 模型对比
    "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
    "repair_model": "gpt-4o",  # 修复阶段统一用强模型

    # 路线参数
    "tau_list": [0.5, 1.0, 1.5],
    "lambda_list": [0.0, 0.2, 0.3, 0.5],

    # 稳定性分析
    "stability_ratios": [0.25, 0.5, 1.0],
}

# ============================================================
# 创建输出目录结构
# ============================================================
def setup_output_dirs():
    """创建所有输出目录"""
    dirs = [
        "results/mnl",
        "results/bootstrap",
        "results/route_a",
        "results/route_b",
        "results/route_c",
        "results/evaluation",
        "results/stability",
    ]
    for model in CONFIG["models"]:
        dirs.append(f"results/route_a/{model}")

    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

    return Path("results")

# ============================================================
# 检查API Key
# ============================================================
def check_api_key():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("=" * 60)
        print("错误: 未设置 OPENAI_API_KEY")
        print("=" * 60)
        print("请设置环境变量后重新运行:")
        print("  Windows CMD:    set OPENAI_API_KEY=你的密钥")
        print("  Windows PS:     $env:OPENAI_API_KEY='你的密钥'")
        print("=" * 60)
        sys.exit(1)
    return api_key

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

    df_train, df_test, train_ids, test_ids = split_train_test_by_id(df, test_size=0.2, seed=42)
    print(f"训练集: {len(df_train)} 行, {len(train_ids)} 个受访者")
    print(f"测试集: {len(df_test)} 行, {len(test_ids)} 个受访者")

    train_data = build_matrices(df_train)
    test_data = build_matrices(df_test)

    tt_scale, co_scale, he_scale = compute_scales_from_train(train_data)
    print(f"TT缩放: {tt_scale.flatten()[:3]}")
    print(f"CO缩放: {co_scale.flatten()[:3]}")

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
def fit_mnl_models(data, output_dir):
    print("\n" + "=" * 60)
    print("2. MNL模型估计")
    print("=" * 60)

    from swissmetro_llm.models import fit_mnl_step1, fit_mnl_step2, build_X_ind, predict_step2
    from swissmetro_llm.models.utils import accuracy
    from swissmetro_llm.models.mnl import get_x_names

    train_data = data["train_data"]
    test_data = data["test_data"]

    # Step 1
    print("\n--- Step 1: 基础MNL (6参数) ---")
    res1 = fit_mnl_step1(train_data)
    print(f"收敛: {res1.success}")
    print(f"负对数似然: {res1.fun:.2f}")

    theta1 = res1.x
    param_names1 = ["B_TT", "B_CO", "B_HE", "B_SEATS", "ASC_SM", "ASC_CAR"]

    # 保存Step1参数
    step1_df = pd.DataFrame({"param": param_names1, "coef": theta1})
    step1_df.to_csv(output_dir / "mnl" / "mnl_step1_params.csv", index=False)

    # Step 2
    print("\n--- Step 2: 扩展MNL (44参数) ---")
    X_train = build_X_ind(train_data)
    X_test = build_X_ind(test_data)

    res2 = fit_mnl_step2(train_data, X_train, theta1=theta1, maxiter=20000, maxfun=20000)
    print(f"收敛: {res2.success}")
    print(f"负对数似然: {res2.fun:.2f}")
    print(f"参数数量: {len(res2.x)}")

    theta2 = res2.x

    # 生成完整参数名
    x_names = get_x_names()
    param_names2 = (
        ["B_TT", "B_CO", "B_HE", "B_SEATS", "ASC_SM", "ASC_CAR"]
        + [f"G_SM:{n}" for n in x_names]
        + [f"G_CAR:{n}" for n in x_names]
    )

    # 保存Step2参数
    step2_df = pd.DataFrame({"param": param_names2, "coef": theta2})
    step2_df.to_csv(output_dir / "mnl" / "mnl_step2_params.csv", index=False)

    # 评估
    P_train = predict_step2(theta2, train_data, X_train)
    P_test = predict_step2(theta2, test_data, X_test)
    acc_train = accuracy(P_train, train_data['y'])
    acc_test = accuracy(P_test, test_data['y'])

    print(f"\n训练集准确率: {acc_train:.4f}")
    print(f"测试集准确率: {acc_test:.4f}")

    # 保存checkpoint
    checkpoint = {
        "theta1": theta1,
        "theta2": theta2,
        "param_names2": param_names2,
        "res1_success": res1.success,
        "res2_success": res2.success,
        "acc_train": acc_train,
        "acc_test": acc_test,
    }
    with open(output_dir / "mnl" / "mnl_checkpoint.pkl", "wb") as f:
        pickle.dump(checkpoint, f)

    return {
        "theta1": theta1,
        "theta2": theta2,
        "param_names2": param_names2,
        "X_train": X_train,
        "X_test": X_test,
        "acc_train": acc_train,
        "acc_test": acc_test,
    }

# ============================================================
# 3. Bootstrap推断
# ============================================================
def run_bootstrap(data, mnl_results, output_dir):
    print("\n" + "=" * 60)
    print(f"3. Bootstrap推断 (B={CONFIG['bootstrap_B']})")
    print("=" * 60)

    from swissmetro_llm.models import cluster_bootstrap_thetas, compute_bootstrap_se, compute_bootstrap_ci
    from scipy import stats

    B = CONFIG["bootstrap_B"]
    print(f"运行 {B} 次 Bootstrap (预计需要60-90分钟)...")

    start_time = time.time()

    thetas_boot = cluster_bootstrap_thetas(
        data["df_train"],
        data["train_ids"],
        data["tt_scale"],
        data["co_scale"],
        data["he_scale"],
        theta_init=mnl_results["theta2"],
        B=B,
        seed=CONFIG["bootstrap_seed"],
        verbose_every=20,
        maxiter=20000,
        maxfun=20000
    )

    elapsed = time.time() - start_time
    print(f"\n完成 {len(thetas_boot)}/{B} 次, 耗时: {elapsed/60:.1f}分钟")

    if len(thetas_boot) == 0:
        print("警告: Bootstrap全部失败!")
        return None

    # 计算统计量
    se_boot = compute_bootstrap_se(thetas_boot)
    ci_lower, ci_upper = compute_bootstrap_ci(thetas_boot, alpha=0.05)

    theta2 = mnl_results["theta2"]
    z_values = np.zeros(len(theta2))
    p_values = np.ones(len(theta2))

    for i in range(len(theta2)):
        if se_boot[i] > 1e-10:
            z_values[i] = theta2[i] / se_boot[i]
            p_values[i] = 2 * (1 - stats.norm.cdf(abs(z_values[i])))

    # 保存结果
    boot_df = pd.DataFrame({
        "param": mnl_results["param_names2"],
        "coef": theta2,
        "boot_se": se_boot,
        "z": z_values,
        "p_value": p_values,
        "ci_2.5": ci_lower,
        "ci_97.5": ci_upper,
    })
    boot_df.to_csv(output_dir / "bootstrap" / f"bootstrap_B{B}.csv", index=False)

    # 保存原始thetas
    np.save(output_dir / "bootstrap" / "bootstrap_thetas.npy", thetas_boot)

    print(f"\n已保存: results/bootstrap/bootstrap_B{B}.csv")

    return {
        "thetas_boot": thetas_boot,
        "se_boot": se_boot,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_values": p_values,
    }

# ============================================================
# 4. 创建评估函数和模板
# ============================================================
def setup_generation(data, mnl_results):
    print("\n" + "=" * 60)
    print("4. 准备生成配置")
    print("=" * 60)

    from swissmetro_llm.evaluation import score_with_baseline_mnl
    from swissmetro_llm.generation import create_templates

    # 创建beta字典
    beta = dict(zip(mnl_results["param_names2"], mnl_results["theta2"]))
    scales = {
        "tt_scale": data["tt_scale"],
        "co_scale": data["co_scale"],
        "he_scale": data["he_scale"],
        "train_ids": data["train_ids"],
    }

    def score_fn(df):
        return score_with_baseline_mnl(df, beta, scales)

    # 创建模板
    templates = create_templates(
        real_train=data["df_train"],
        real_test=data["df_test"],
        N=CONFIG["synth_N"],
        p_unseen=CONFIG["p_unseen"],
        seed=CONFIG["seed"]
    )
    print(f"创建了 {len(templates)} 个模板")

    return {
        "beta": beta,
        "scales": scales,
        "score_fn": score_fn,
        "templates": templates,
    }

# ============================================================
# 5. 路线A: Two-stage CHOICE
# ============================================================
def run_route_a(gen_config, output_dir):
    print("\n" + "=" * 60)
    print("5A. 路线A: Two-stage CHOICE")
    print("=" * 60)

    from swissmetro_llm.generation import generate_two_stage

    results = {}

    for model in CONFIG["models"]:
        print(f"\n--- 模型: {model} ---")

        model_dir = output_dir / "route_a" / model
        model_dir.mkdir(exist_ok=True)

        # 检查是否已存在
        syn_path = model_dir / "syn_final.csv"
        if syn_path.exists():
            print(f"已存在，跳过: {syn_path}")
            results[model] = pd.read_csv(syn_path)
            continue

        try:
            start_time = time.time()
            syn = generate_two_stage(
                templates=gen_config["templates"],
                score_fn=gen_config["score_fn"],
                model_stage1=model,
                model_stage2=CONFIG["repair_model"],
                low_prob_threshold=0.01,
                jsonl_dir=str(model_dir),
                seed=CONFIG["seed"],
                cot_stage1=False,
                cot_stage2=True
            )
            elapsed = time.time() - start_time

            print(f"完成! 耗时: {elapsed/60:.1f}分钟, 样本数: {len(syn)}")
            syn.to_csv(syn_path, index=False)
            results[model] = syn

        except Exception as e:
            print(f"失败: {e}")

    return results

# ============================================================
# 6. 路线B: Utility + Softmax
# ============================================================
def run_route_b(gen_config, output_dir):
    print("\n" + "=" * 60)
    print("5B. 路线B: Utility + Softmax")
    print("=" * 60)

    from swissmetro_llm.generation import generate_from_utilities_batch

    results = {}

    for model in CONFIG["models"]:
        for tau in CONFIG["tau_list"]:
            label = f"{model}_tau{tau}"
            print(f"\n--- {label} ---")

            syn_path = output_dir / "route_b" / f"{label}.csv"
            if syn_path.exists():
                print(f"已存在，跳过")
                results[label] = pd.read_csv(syn_path)
                continue

            jsonl_path = str(output_dir / "route_b" / f"{label}.jsonl")
            out_path = str(output_dir / "route_b" / f"{label}_out.jsonl")

            try:
                start_time = time.time()
                syn = generate_from_utilities_batch(
                    templates=gen_config["templates"],
                    model=model,
                    tau=tau,
                    jsonl_path=jsonl_path,
                    out_path=out_path,
                    seed=CONFIG["seed"],
                    cot=False
                )
                elapsed = time.time() - start_time

                print(f"完成! 耗时: {elapsed/60:.1f}分钟, 样本数: {len(syn)}")
                syn.to_csv(syn_path, index=False)
                results[label] = syn

            except Exception as e:
                print(f"失败: {e}")

    return results

# ============================================================
# 7. 路线C: MNL + Residual
# ============================================================
def run_route_c(gen_config, output_dir):
    print("\n" + "=" * 60)
    print("5C. 路线C: MNL + Residual")
    print("=" * 60)

    from swissmetro_llm.generation import generate_from_residual_batch

    results = {}

    for model in CONFIG["models"]:
        for lam in CONFIG["lambda_list"]:
            label = f"{model}_lam{lam}"
            print(f"\n--- {label} ---")

            syn_path = output_dir / "route_c" / f"{label}.csv"
            if syn_path.exists():
                print(f"已存在，跳过")
                results[label] = pd.read_csv(syn_path)
                continue

            jsonl_path = str(output_dir / "route_c" / f"{label}.jsonl")
            out_path = str(output_dir / "route_c" / f"{label}_out.jsonl")

            try:
                start_time = time.time()
                syn = generate_from_residual_batch(
                    templates=gen_config["templates"],
                    score_fn=gen_config["score_fn"],
                    model=model,
                    lam=lam,
                    jsonl_path=jsonl_path,
                    out_path=out_path,
                    seed=CONFIG["seed"],
                    cot=False
                )
                elapsed = time.time() - start_time

                print(f"完成! 耗时: {elapsed/60:.1f}分钟, 样本数: {len(syn)}")
                syn.to_csv(syn_path, index=False)
                results[label] = syn

            except Exception as e:
                print(f"失败: {e}")

    return results

# ============================================================
# 8. 评估所有合成数据
# ============================================================
def evaluate_all(data, gen_config, syn_datasets, output_dir):
    print("\n" + "=" * 60)
    print("6. 评估所有合成数据")
    print("=" * 60)

    from swissmetro_llm.evaluation import evaluate_one

    eval_results = []

    for label, syn_df in syn_datasets.items():
        print(f"\n评估: {label}")

        try:
            metrics = evaluate_one(
                synth_df=syn_df,
                beta=gen_config["beta"],
                scales=gen_config["scales"],
                real_train=data["df_train"],
                real_test=data["df_test"],
                label=label
            )
            metrics["label"] = label
            eval_results.append(metrics)

            # 打印关键指标
            acc = metrics.get('ds_accuracy', metrics.get('accuracy', 'N/A'))
            avg_p = metrics.get('ds_avg_P_chosen', metrics.get('avg_P_chosen', 'N/A'))
            low_prob = metrics.get('ds_low_prob_rate(<0.01)', 'N/A')
            print(f"  准确率: {acc:.4f}" if isinstance(acc, float) else f"  准确率: {acc}")
            print(f"  平均P_chosen: {avg_p:.4f}" if isinstance(avg_p, float) else f"  平均P_chosen: {avg_p}")
            print(f"  低概率率: {low_prob:.4f}" if isinstance(low_prob, float) else f"  低概率率: {low_prob}")

        except Exception as e:
            print(f"  评估失败: {e}")

    if eval_results:
        eval_df = pd.DataFrame(eval_results)
        eval_df.to_csv(output_dir / "evaluation" / "all_evaluations.csv", index=False)
        print(f"\n已保存: results/evaluation/all_evaluations.csv")
        return eval_df

    return None

# ============================================================
# 9. 参数稳定性分析
# ============================================================
def run_stability_analysis(data, mnl_results, boot_results, syn_datasets, output_dir):
    print("\n" + "=" * 60)
    print("7. 参数稳定性分析")
    print("=" * 60)

    from swissmetro_llm.stability import stability_analysis, format_stability_for_report

    # 选择最佳合成数据集进行分析 (使用路线C lambda=0.3的gpt-4o-mini)
    best_labels = [
        "gpt-4o-mini_lam0.3",
        "gpt-4o-mini_lam0.2",
        "gpt-4o-mini",  # route_a
    ]

    syn_for_stability = None
    used_label = None

    for label in best_labels:
        if label in syn_datasets:
            syn_for_stability = syn_datasets[label]
            used_label = label
            break

    if syn_for_stability is None and syn_datasets:
        used_label = list(syn_datasets.keys())[0]
        syn_for_stability = syn_datasets[used_label]

    if syn_for_stability is None:
        print("没有可用的合成数据，跳过稳定性分析")
        return None

    print(f"\n使用 {used_label} 进行稳定性分析")
    print(f"合成数据样本数: {len(syn_for_stability)}")

    all_results = []

    for ratio in CONFIG["stability_ratios"]:
        print(f"\n--- ratio = {ratio} ---")

        stab_df, metadata = stability_analysis(
            real_train=data["df_train"],
            syn_df=syn_for_stability,
            tt_scale=data["tt_scale"],
            co_scale=data["co_scale"],
            he_scale=data["he_scale"],
            theta_init=mnl_results["theta2"],
            boot_se=boot_results["se_boot"] if boot_results else None,
            ci_lower=boot_results["ci_lower"] if boot_results else None,
            ci_upper=boot_results["ci_upper"] if boot_results else None,
            ratio=ratio,
            seed=CONFIG["seed"],
            verbose=True
        )

        stab_df["ratio"] = ratio
        stab_df["syn_source"] = used_label
        all_results.append(stab_df)

        # 保存每个ratio的结果
        format_stability_for_report(
            stab_df,
            metadata,
            output_path=str(output_dir / "stability" / f"param_stability_ratio{ratio}.csv"),
            top_n=15
        )

    # 合并所有结果
    all_df = pd.concat(all_results, ignore_index=True)
    all_df.to_csv(output_dir / "stability" / "param_stability_all_ratios.csv", index=False)

    # 汇总统计
    summary_stats = []
    for ratio in CONFIG["stability_ratios"]:
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
    summary_df.to_csv(output_dir / "stability" / "stability_summary.csv", index=False)

    print("\n稳定性分析汇总:")
    print(summary_df.to_string(index=False))

    return all_df

# ============================================================
# 10. 生成汇总报告
# ============================================================
def generate_summary_report(eval_df, stability_df, output_dir):
    print("\n" + "=" * 60)
    print("8. 生成汇总报告")
    print("=" * 60)

    report_lines = [
        f"# Swissmetro LLM 实验汇总报告",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 实验配置",
        f"- Bootstrap次数: {CONFIG['bootstrap_B']}",
        f"- 合成样本数量: {CONFIG['synth_N']}",
        f"- 对比模型: {', '.join(CONFIG['models'])}",
        f"- tau参数: {CONFIG['tau_list']}",
        f"- lambda参数: {CONFIG['lambda_list']}",
        "",
    ]

    if eval_df is not None:
        report_lines.append("## 评估结果摘要")
        # 按准确率排序
        if 'ds_accuracy' in eval_df.columns:
            top5 = eval_df.nlargest(5, 'ds_accuracy')[['label', 'ds_accuracy', 'ds_avg_P_chosen']]
            report_lines.append("\n准确率Top 5:")
            report_lines.append(top5.to_string(index=False))

    if stability_df is not None:
        report_lines.append("\n## 稳定性分析摘要")
        summary = stability_df.groupby('ratio')['abs_diff'].agg(['mean', 'median', 'max'])
        report_lines.append(summary.to_string())

    report_text = "\n".join(report_lines)

    with open(output_dir / "summary_report.md", "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n已保存: results/summary_report.md")
    print("\n" + report_text)

# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("Swissmetro LLM 完整实验")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 检查API Key
    api_key = check_api_key()
    print(f"API Key: {api_key[:8]}...{api_key[-4:]}")

    # 创建目录
    output_dir = setup_output_dirs()

    # 1. 加载数据
    data = load_data()

    # 2. MNL估计
    mnl_results = fit_mnl_models(data, output_dir)

    # 3. Bootstrap (这步最耗时)
    boot_results = run_bootstrap(data, mnl_results, output_dir)

    # 4. 准备生成配置
    gen_config = setup_generation(data, mnl_results)

    # 5-7. LLM生成三条路线
    syn_datasets = {}

    # 路线A
    route_a_results = run_route_a(gen_config, output_dir)
    for model, syn in route_a_results.items():
        syn_datasets[model] = syn

    # 路线B
    route_b_results = run_route_b(gen_config, output_dir)
    syn_datasets.update(route_b_results)

    # 路线C
    route_c_results = run_route_c(gen_config, output_dir)
    syn_datasets.update(route_c_results)

    # 8. 评估
    eval_df = evaluate_all(data, gen_config, syn_datasets, output_dir)

    # 9. 稳定性分析
    stability_df = run_stability_analysis(data, mnl_results, boot_results, syn_datasets, output_dir)

    # 10. 汇总报告
    generate_summary_report(eval_df, stability_df, output_dir)

    # 完成
    print("\n" + "=" * 60)
    print("完整实验运行完成!")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结果目录: {output_dir.absolute()}")
    print("=" * 60)

    print("\n生成的文件:")
    for f in sorted(output_dir.rglob("*.csv")):
        print(f"  - {f.relative_to(output_dir)}")

if __name__ == "__main__":
    main()
