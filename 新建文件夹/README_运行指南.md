# Swissmetro LLM 包运行指南

## 目录结构

```
新建文件夹/
├── swissmetro.dat           # 原始数据 (已存在)
├── swissmetro_llm/          # Python包
├── requirements.txt         # 依赖列表
├── run_example.py           # 运行示例
└── README_运行指南.md       # 本文件
```

## 步骤1: 安装依赖

打开 **命令提示符(CMD)** 或 **PowerShell**，进入项目目录：

```bash
cd C:\Users\12978\Desktop\新建文件夹
pip install -r requirements.txt
```

如果使用 Anaconda：
```bash
conda activate your_env_name
pip install -r requirements.txt
```

## 步骤2: 运行示例

```bash
cd C:\Users\12978\Desktop\新建文件夹
python run_example.py
```

这会执行：
1. 加载 swissmetro.dat 数据
2. 划分训练/测试集
3. 估计 MNL Step1/Step2 模型
4. 运行简单的 Bootstrap (5次)
5. 展示评估框架

## 步骤3: 使用LLM生成 (可选)

如需运行LLM合成数据生成，需要设置OpenAI API密钥：

**Windows CMD:**
```cmd
set OPENAI_API_KEY=sk-你的密钥
python run_example.py
```

**Windows PowerShell:**
```powershell
$env:OPENAI_API_KEY="sk-你的密钥"
python run_example.py
```

## 在Jupyter Notebook中使用

```python
import sys
sys.path.insert(0, 'C:/Users/12978/Desktop/新建文件夹')

from swissmetro_llm.data import load_swissmetro, build_matrices, split_train_test_by_id
from swissmetro_llm.models import fit_mnl_step1, fit_mnl_step2
from swissmetro_llm.evaluation import evaluate_one
from swissmetro_llm.generation import create_templates, generate_from_utilities_batch

# 加载数据
df = load_swissmetro("C:/Users/12978/Desktop/新建文件夹/swissmetro.dat")
```

## 主要模块说明

| 模块 | 功能 |
|------|------|
| `swissmetro_llm.data` | 数据加载、预处理、train/test划分 |
| `swissmetro_llm.models` | MNL估计、Bootstrap推断 |
| `swissmetro_llm.evaluation` | 可行性、多样性、下游一致性评估 |
| `swissmetro_llm.generation` | LLM提示、Batch API、生成管线 |

## 常用函数

```python
# 数据加载
from swissmetro_llm.data import load_swissmetro, build_matrices, split_train_test_by_id
df = load_swissmetro("swissmetro.dat")
df_train, df_test, train_ids, test_ids = split_train_test_by_id(df, test_size=0.2)

# MNL估计
from swissmetro_llm.models import fit_mnl_step1, fit_mnl_step2, build_X_ind
res1 = fit_mnl_step1(train_data)
X_train = build_X_ind(train_data)
res2 = fit_mnl_step2(train_data, X_train, theta1=res1.x)

# Bootstrap
from swissmetro_llm.models import cluster_bootstrap_thetas, compute_bootstrap_se
thetas = cluster_bootstrap_thetas(df_train, train_ids, tt_scale, co_scale, he_scale, B=30)

# 评估
from swissmetro_llm.evaluation import evaluate_one
result = evaluate_one(synth_df, beta, scales, df_train, df_test, label="my_synth")

# 生成
from swissmetro_llm.generation import create_templates, generate_from_utilities_batch
templates = create_templates(df_train, df_test, N=2000)
syn = generate_from_utilities_batch(templates, model="gpt-4o-mini", tau=1.0)
```

## 故障排除

### 1. ModuleNotFoundError
确保在项目根目录运行，或添加路径：
```python
import sys
sys.path.insert(0, 'C:/Users/12978/Desktop/新建文件夹')
```

### 2. 缺少依赖
```bash
pip install numpy pandas scipy openai
```

### 3. OpenAI API错误
- 检查API密钥是否正确
- 检查账户余额
- 确认网络连接

### 4. 数据文件找不到
确保 `swissmetro.dat` 在运行目录中，或使用绝对路径：
```python
df = load_swissmetro("C:/Users/12978/Desktop/新建文件夹/swissmetro.dat")
```
