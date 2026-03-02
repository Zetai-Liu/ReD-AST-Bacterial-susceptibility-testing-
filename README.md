# ReD-AST-Bacterial-susceptibility-testing-

Sensor data analysis + binary ML classification for ReD-AST datasets to determine bacterial antimicrobial susceptibility.

## 最小可运行模板（已提供）

本仓库现在包含一个可直接运行的最小模板，覆盖：

- 目录结构
- 示例数据
- 基线训练脚本（逻辑回归）
- 自动化测试
- 运行说明

## 目录结构

```text
.
├── configs/                  # 预留：实验配置
├── data/
│   └── sample/
│       └── reduced_ast_sample.csv
├── docs/                     # 预留：标签定义/数据字典等文档
├── reports/                  # 训练产物（模型、指标报告）
├── scripts/
│   └── train_baseline.py     # 一键训练入口
├── src/
│   └── red_ast/
│       ├── __init__.py
│       └── pipeline.py       # 数据读取、预处理、训练、评估
├── tests/
│   └── test_pipeline.py
└── requirements.txt
```

## 快速开始

### 1) 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # 仅测试依赖
```

### 2) 训练基线模型

```bash
PYTHONPATH=src python scripts/train_baseline.py
```

成功后会生成：

- `reports/model.json`：训练好的模型参数
- `reports/metrics.txt`：评估指标与分类报告

### 3) 运行测试

```bash
PYTHONPATH=src pytest -q
```

## 说明

- 当前示例数据是演示用小样本，仅用于跑通流程。
- 后续可将真实 ReD-AST 数据按同字段格式接入，并扩展 `src/red_ast/pipeline.py`。
