# 认知工作空间 - 概念验证实现

[English](README.md) | 中文版 | [📚 Wiki](https://github.com/tao-hpu/cognitive-workspace/wiki)

[![GitHub stars](https://img.shields.io/github/stars/tao-hpu/cognitive-workspace?style=social)](https://github.com/tao-hpu/cognitive-workspace/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/tao-hpu/cognitive-workspace?style=social)](https://github.com/tao-hpu/cognitive-workspace/network)
[![GitHub issues](https://img.shields.io/github/issues/tao-hpu/cognitive-workspace)](https://github.com/tao-hpu/cognitive-workspace/issues)
[![GitHub license](https://img.shields.io/github/license/tao-hpu/cognitive-workspace)](https://github.com/tao-hpu/cognitive-workspace/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2508.13171-b31b1b.svg)](https://arxiv.org/abs/2508.13171)

---

## 📋 目录

- [快速开始](#快速开始)
  - [1. 安装依赖](#1-安装依赖)
  - [2. 配置环境变量](#2-配置环境变量)
  - [3. 运行实验](#3-运行实验)
- [运行模式](#运行模式)
- [实验内容](#实验内容)
- [输出文件](#输出文件)
- [关键指标解释](#关键指标解释)
- [系统架构](#系统架构)
- [实验截图](#实验截图)
- [论文支撑](#论文支撑)
- [常见问题](#常见问题)
- [故障排除](#故障排除)
- [扩展建议](#扩展建议)
- [贡献](#贡献)
- [联系与支持](#联系与支持)
- [引用](#引用)
- [致谢](#致谢)
- [Star历史](#star历史)
- [贡献者](#贡献者)
- [许可](#许可)

---

## 🚀 快速开始

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

## 🎯 运行模式

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

## 🔬 实验内容

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

## 📁 输出文件

- `cognitive_workspace_results.json`：基础实验结果
- `enhanced_results.json`：增强实验详细结果
- `cognitive_workspace_analysis.png`：实验可视化图表
- `.env.example`：环境变量模板（如果不存在.env）

## 📊 关键指标解释

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

## 🏗️ 系统架构

认知工作空间实现了一个模仿人类认知过程的分层记忆架构：

```
┌─────────────────────────────────────────────────────────┐
│              用户查询接口                                │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│         元认知控制器                                     │
│  • 任务分解                                             │
│  • 置信度追踪                                           │
│  • 信息差距分析                                         │
└────────────────┬────────────────────────────────────────┘
                 │
     ┌───────────┴───────────┐
     ▼                       ▼
┌─────────────┐      ┌──────────────────┐
│  即时缓冲   │      │  主动记忆预测    │
│  (最近)     │◄────►│   与准备         │
└──────┬──────┘      └──────────────────┘
       │
       ▼
┌─────────────┐
│  工作缓冲   │
│  (相关)     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  情景缓冲   │
│  (长期)     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│         外部知识库 (RAG)                                │
└─────────────────────────────────────────────────────────┘
```

**与传统RAG的关键差异：**
- **有状态**: 跨多个查询维护上下文
- **预测性**: 在需要之前主动准备信息
- **分层式**: 三层记忆缓冲系统，具有智能晋升机制
- **元认知**: 自我感知信息差距和任务完成度

> **提示**: 您可以通过在 `assets/` 文件夹中放置图片并在此处引用来添加自己的架构图。

## 📸 实验截图

运行实验时，您将看到类似以下的输出：

### 多轮对话对比
实验生成详细的对比图表，显示不同场景下的记忆复用率。

### 统计分析可视化
`cognitive_workspace_analysis.png` 包含以下可视化内容：
- 对话轮次中的记忆复用率趋势
- 操作数量对比（认知工作空间 vs RAG）
- 统计显著性指标（Cohen's d效应量）
- 考虑开销后的净效率提升

> **提示**: 运行 `cognitive_workspace_enhanced.py` 后，查看生成的 `cognitive_workspace_analysis.png` 文件以获取详细的可视化结果。

## 📄 论文支撑

此代码支撑以下论文观点：

1. **主动记忆管理优于被动检索**
   - 代码证明：任务分解、信息预测、主动准备
   
2. **状态持久性提升效率**
   - 代码证明：多轮对话中的记忆复用

3. **分层缓冲优化资源利用**
   - 代码证明：immediate→working→episodic的晋升机制

4. **元认知控制增强智能**
   - 代码证明：置信度追踪、信息gap识别

## ❓ 常见问题

<details>
<summary><b>Q: 为什么模拟模式也能证明观点？</b></summary>

A: 因为我们证明的是架构行为差异，不是生成质量。即使用规则模拟，主动vs被动、有状态vs无状态的差异仍然明显。
</details>

<details>
<summary><b>Q: 如何在论文中引用这个代码？</b></summary>

A: 在LaTeX中使用以下格式：
```latex
Code available at: \url{https://github.com/tao-hpu/cognitive-workspace}
```
</details>

<details>
<summary><b>Q: 需要多少token/API调用？</b></summary>

A: 完整实验约需：
- 单轮实验：~10个API调用
- 多轮实验：~20个API调用
- 总成本：< $0.05（使用GPT-3.5-turbo）
</details>

<details>
<summary><b>Q: 可以用其他LLM吗？</b></summary>

A: 可以！代码支持：
- OpenAI兼容API（通过修改OPENAI_API_BASE）
- 本地模型（Ollama、llama.cpp）
- 任何提供chat/completion接口的服务
</details>

## 🔧 故障排除

### API连接错误

**问题**: `openai.error.AuthenticationError` 或连接超时

**解决方案**:
- 检查 `.env` 中的API密钥是否正确
- 检查 `OPENAI_API_BASE` URL格式（应以 `/v1` 结尾）
- 对于Azure OpenAI，确保使用正确的端点格式
- 测试连接: `curl -H "Authorization: Bearer $OPENAI_API_KEY" $OPENAI_API_BASE/models`

### 可选依赖的导入错误

**问题**: `ModuleNotFoundError: No module named 'sentence_transformers'`

**解决方案**:
- 安装缺失的依赖: `pip install sentence-transformers`
- 完整功能安装: `pip install openai python-dotenv sentence-transformers scipy matplotlib`
- 检查Python版本（需要3.7+）

### 结果与预期值不符

**问题**: 复用率或指标与文档不匹配

**解决方案**:
- **模拟模式**（无API密钥）: 结果是确定性的但简化的
- **完整模式**（有API密钥）: 由于LLM的随机性，结果会略有不同
  - 在代码中设置temperature=0以获得更一致的结果
  - 运行多次试验以获得统计有效性
- 确保比较的是同一个实验（基础vs增强）

### 内存或性能问题

**问题**: 脚本运行缓慢或使用过多内存

**解决方案**:
- 首先运行基础实验: `python cognitive_workspace_poc.py`
- 减少测试数据中的文档数量
- 对于本地模型，确保有足够的RAM（建议8GB+）
- 检查后台进程是否占用资源

### 结果文件未生成

**问题**: 缺少 `.json` 或 `.png` 输出文件

**解决方案**:
- 检查控制台输出中的错误
- 确保当前目录有写入权限
- 对于可视化: 验证已安装matplotlib
- 使用以下命令运行: `python cognitive_workspace_enhanced.py 2>&1 | tee output.log`

## 💡 扩展建议

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

## 🤝 贡献

我们欢迎贡献来改进这个概念验证实现！以下是您可以提供帮助的方式：

### 贡献方式

- **Bug报告**: 提交issue描述问题和复现步骤
- **功能建议**: 提出新的实验或架构改进
- **代码改进**: 提交pull request修复bug或增强功能
- **文档**: 改进README、添加代码注释或创建教程
- **测试**: 添加测试用例或在不同平台上验证结果

### 贡献指南

1. **Fork本仓库** 并从 `main` 分支创建您的分支
2. **进行修改** 并使用清晰、描述性的提交信息
3. **彻底测试您的更改**（运行基础和增强实验）
4. **更新文档** 如果您更改了功能
5. **提交pull request** 并清晰描述您的更改

### 行为准则

- 在讨论中保持尊重和建设性
- 关注贡献的技术价值
- 帮助维护这个研究和教育资源

## 📬 联系与支持

### 获取帮助

- **Issues**: 对于bug报告和功能请求，使用 [GitHub Issues](https://github.com/tao-hpu/cognitive-workspace/issues)
- **讨论**: 对于问题和一般性讨论，发起 [GitHub Discussion](https://github.com/tao-hpu/cognitive-workspace/discussions)
- **文档**: 查看 [Wiki](https://github.com/tao-hpu/cognitive-workspace/wiki) 获取更多资源

### 研究合作

如果您有兴趣在认知工作空间相关研究上合作，或对论文有学术性问题：

- **作者**: Tao An
- **论文**: [arXiv:2508.13171](https://arxiv.org/abs/2508.13171)
- 研究咨询请参考论文中的联系信息

### 报告安全问题

如果您发现安全漏洞，请私下报告而不是公开issue。

## 📖 引用

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

## 🙏 致谢

此概念验证实现的开发旨在演示认知工作空间论文中描述的架构原理。我们要感谢：

- **研究社区**: 对论文和实现提供宝贵反馈
- **开源贡献者**: 所有帮助改进此代码库的贡献者
- **用户与测试者**: 所有尝试此POC并提供见解的人
- **审稿人**: 提出建设性意见，改进论文和代码质量

特别感谢更广泛的AI/ML研究社区，在记忆架构、元认知和高效LLM系统方面进行启发性讨论。

### 构建工具

本项目使用了优秀的开源工具：
- **OpenAI API** - LLM集成
- **NumPy** - 数值计算
- **Sentence Transformers** - 向量嵌入（可选）
- **SciPy & Matplotlib** - 统计分析和可视化（可选）

## ⭐ Star历史

[![Star History Chart](https://api.star-history.com/svg?repos=tao-hpu/cognitive-workspace&type=Date)](https://star-history.com/#tao-hpu/cognitive-workspace&Date)

## 👥 贡献者

感谢所有帮助改进此项目的贡献者！

[![Contributors](https://contrib.rocks/image?repo=tao-hpu/cognitive-workspace)](https://github.com/tao-hpu/cognitive-workspace/graphs/contributors)

### 如何贡献

欢迎贡献！请参阅上面的 [贡献](#-贡献) 部分了解指南。

---

## 📜 许可

MIT License - 自由使用、修改和分发

---

<div align="center">

**用 ❤️ 为AI研究社区打造**

如果您觉得这个项目有用，请考虑给它一个 ⭐！

[报告Bug](https://github.com/tao-hpu/cognitive-workspace/issues) · [请求功能](https://github.com/tao-hpu/cognitive-workspace/issues) · [查看论文](https://arxiv.org/abs/2508.13171)

</div>
