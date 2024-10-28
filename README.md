# BERT Performance Benchmark

实现了一个 BERT 模型性能基准测试工具，用于比较传统 Eager 模式与优化后的缩放点积注意力（Scaled Dot-Product Attention, SDPA）在不同配置下的性能表现。

## 功能特点

- 支持不同批处理大小和序列长度的性能测试
- 对比 Eager 模式和 SDPA 模式的性能差异
- 测量并比较延迟、内存使用和加速比
- 自动生成性能报告和可视化图表

## 环境要求

- Python 3.8+
- PyTorch 2.2.0+
- CUDA 11.8+
- NVIDIA GPU with compute capability 7.0+

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/bert_benchmark.git
cd bert_benchmark
```

2. 创建并激活虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 运行基准测试：
```bash
python benchmark.py
```

2. 配置选项：
```python
# 在 run_benchmarks() 函数中修改以下参数：
train_configs = [
    (4, 256), (4, 512),  # (batch_size, sequence_length)
    (8, 256), (8, 512),
    (16, 256), (16, 512),
]

infer_configs = [
    (1, 128), (1, 256),
    (2, 128), (2, 256),
    (4, 128), (4, 256),
]
```

## 结果示例

基准测试结果包括：

1. 每个 token 的处理延迟：
   - Eager 模式：0.019ms (BS=1, SL=128)
   - SDPA 模式：0.018ms (BS=1, SL=128)

2. 内存使用：
   - 基础配置：53.71MB (BS=1, SL=128)
   - 最大配置：62.71MB (BS=4, SL=256)

3. 性能提升：
   - 最佳情况：2.77% (BS=2, SL=128)
   - 平均情况：-1.84%
