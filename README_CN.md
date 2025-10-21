# 认知工作空间 - 概念验证实现

[English](README.md) | 中文版 | [📚 Wiki](https://github.com/tao-hpu/cognitive-workspace/wiki)

[![GitHub stars](https://img.shields.io/github/stars/tao-hpu/cognitive-workspace?style=social)](https://github.com/tao-hpu/cognitive-workspace/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/tao-hpu/cognitive-workspace?style=social)](https://github.com/tao-hpu/cognitive-workspace/network)
[![GitHub issues](https://img.shields.io/github/issues/tao-hpu/cognitive-workspace)](https://github.com/tao-hpu/cognitive-workspace/issues)
[![GitHub license](https://img.shields.io/github/license/tao-hpu/cognitive-workspace)](https://github.com/tao-hpu/cognitive-workspace/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2508.13171-b31b1b.svg)](https://arxiv.org/abs/2508.13171)

## 快速开始

### 1. 安装依赖

```bash
# 基础依赖
pip install numpy

# 可选：OpenAI支持
pip install openai python-dotenv

# 可选：更好的向量嵌入
pip install sentence-transformers

# 可选：增强实验（统计分析和可视化）
pip install scipy matplotlib
```

### 2. 配置环境变量

创建 `.env` 文件：

```env
# OpenAI 官方API
OPENAI_API_KEY=sk-your-key-here
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-3.5-turbo

# 或使用 Azure OpenAI
# OPENAI_API_KEY=your-azure-key
# OPENAI_API_BASE=https://your-resource.openai.azure.com
# OPENAI_MODEL=your-deployment-name

# 或使用本地模型 (如 Ollama)
# OPENAI_API_BASE=http://localhost:11434/v1
# OPENAI_MODEL=llama2
```

### 3. 运行实验

```bash
# 基础实验（4轮对话）
python cognitive_workspace_poc.py

# 增强实验（10轮对话 + 多跳推理 + 冲突解决）
python cognitive_workspace_enhanced.py
```

## 运行模式

### 模式1：完整模式（推荐）
需要OpenAI API key，展示真实的LLM行为差异：
- 任务分解质量更高
- 信息预测更准确
- 答案生成更连贯

### 模式2：模拟模式（默认）
无需任何API key，使用规则模拟：
- 仍能展示架构差异
- 适合概念验证
- 完全可重现

### 模式3：本地模式
使用Ollama等本地模型：
- 数据隐私
- 无API成本
- 性能取决于本地硬件

## 实验内容

### 实验1：单轮任务处理
对比认知工作空间与传统RAG在单个复杂问题上的表现：
- 操作数量差异（12 vs 3）
- 操作类型差异（主动vs被动）
- 记忆管理差异（分层vs扁平）
- 单轮记忆复用率：50% vs 0%

### 实验2：多轮对话（核心优势）
展示状态持久性带来的累积优势：
```
轮次  CW复用率  RAG复用率
1     50.0%    0%
2     55.0%    0%
3     56.7%    0%
4     56.4%    0%

平均复用率: 54.5% vs 0%
```

### 实验3：10轮扩展对话（增强版）
长期对话中的记忆优势：
```
平均复用率: 57.1% vs 0%
净效率提升: 17.3%
Cohen's d: 23.2（巨大效应）
P值: < 0.001（极度显著）
```

### 实验4：多跳推理（增强版）
复杂推理链中的优势：
```
平均复用率: 58.8% vs 0%
净效率提升: 17.9%
Cohen's d: 190.0（超大效应）
节省操作数: 194
```

### 实验5：信息冲突解决（增强版）
处理矛盾信息时的表现：
```
平均复用率: 59.8% vs 0%
净效率提升: 17.8%
Cohen's d: 195.7（超大效应）
节省操作数: 226
```

## 输出文件

- `cognitive_workspace_results.json`：基础实验结果
- `enhanced_results.json`：增强实验详细结果
- `cognitive_workspace_analysis.png`：实验可视化图表
- `.env.example`：环境变量模板（如果不存在.env）

## 关键指标解释

### 记忆复用率（实测数据）
- **基础实验（4轮）**：平均54.5%，从第一轮就开始复用
- **10轮对话**：平均57.1%，长期对话优势明显
- **多跳推理**：平均58.8%，复杂任务复用率更高
- **冲突解决**：平均59.8%，信息整合场景效果最佳
- **传统RAG**：始终为0%（无状态）

### 净效率提升（考虑额外开销后）
```python
净效率 = 复用率 / (1 + 额外操作比率)
```
- **10轮对话**：17.3%净提升
- **多跳推理**：17.9%净提升
- **冲突解决**：17.8%净提升

### 统计显著性
- **P值**：所有实验 < 0.001（极度显著）
- **Cohen's d效应量**：
  - 10轮对话：23.2（巨大）
  - 多跳推理：190.0（超大）
  - 冲突解决：195.7（超大）

### 操作增长模式
- **认知工作空间**：亚线性增长（通过记忆复用减少重复计算）
- **传统RAG**：线性增长（每次查询都重新开始）

### 置信度追踪
- **认知工作空间**：动态追踪任务完成度和信息充足性
- **传统RAG**：无置信度概念

## 论文支撑

此代码支撑以下论文观点：

1. **主动记忆管理优于被动检索**
   - 代码证明：任务分解、信息预测、主动准备
   
2. **状态持久性提升效率**
   - 代码证明：多轮对话中的记忆复用

3. **分层缓冲优化资源利用**
   - 代码证明：immediate→working→episodic的晋升机制

4. **元认知控制增强智能**
   - 代码证明：置信度追踪、信息gap识别

## 常见问题

### Q: 为什么模拟模式也能证明观点？
A: 因为我们证明的是架构行为差异，不是生成质量。即使用规则模拟，主动vs被动、有状态vs无状态的差异仍然明显。

### Q: 如何在论文中引用这个代码？
A: 
```latex
Code available at: \url{https://github.com/tao-hpu/cognitive-workspace}
```

### Q: 需要多少token/API调用？
A: 完整实验约需：
- 单轮实验：~10个API调用
- 多轮实验：~20个API调用
- 总成本：< $0.05（使用GPT-3.5-turbo）

### Q: 可以用其他LLM吗？
A: 可以！代码支持：
- OpenAI兼容API（通过修改OPENAI_API_BASE）
- 本地模型（Ollama、llama.cpp）
- 任何提供chat/completion接口的服务

## 扩展建议

1. **添加更长期测试（20+轮）**
   ```python
   # 修改cognitive_workspace_enhanced.py中的问题列表
   extended_questions = [...20个问题...]
   ```

2. **集成真实向量数据库**
   ```python
   # 使用ChromaDB或Pinecone
   from chromadb import Client
   ```

3. **添加更多统计测试**
   ```python
   # Mann-Whitney U test, Friedman test等
   from scipy import stats
   stats.mannwhitneyu(cw_results, rag_results)
   ```

4. **性能基准测试**
   ```python
   # 测试不同规模下的表现
   for doc_count in [10, 100, 1000]:
       test_scalability(doc_count)
   ```

## 引用

如果您使用此代码，请引用：

```bibtex
@article{an2025cognitive,
  title={Cognitive Workspace: Towards Functional Infinite Context Through Active Memory Management},
  author={Tao An},
  year={2025},
  eprint={2508.13171},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```

## 许可

MIT License - 自由使用、修改和分发
