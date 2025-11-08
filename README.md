# Cognitive Workspace - Proof of Concept Implementation

[中文版](README_CN.md) | English | [📚 Wiki](https://github.com/tao-hpu/cognitive-workspace/wiki)

[![GitHub stars](https://img.shields.io/github/stars/tao-hpu/cognitive-workspace?style=social)](https://github.com/tao-hpu/cognitive-workspace/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/tao-hpu/cognitive-workspace?style=social)](https://github.com/tao-hpu/cognitive-workspace/network)
[![GitHub issues](https://img.shields.io/github/issues/tao-hpu/cognitive-workspace)](https://github.com/tao-hpu/cognitive-workspace/issues)
[![GitHub license](https://img.shields.io/github/license/tao-hpu/cognitive-workspace)](https://github.com/tao-hpu/cognitive-workspace/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2508.13171-b31b1b.svg)](https://arxiv.org/abs/2508.13171)

---

## 📋 Table of Contents

- [Quick Start](#quick-start)
  - [1. Install Dependencies](#1-install-dependencies)
  - [2. Environment Configuration](#2-environment-configuration)
  - [3. Run Experiments](#3-run-experiments)
- [Operation Modes](#operation-modes)
- [Experiment Content](#experiment-content)
- [Output Files](#output-files)
- [Key Metrics Explanation](#key-metrics-explanation)
- [System Architecture](#system-architecture)
- [Experiment Screenshots](#experiment-screenshots)
- [Paper Support](#paper-support)
- [FAQ](#faq)
- [Troubleshooting](#troubleshooting)
- [Extension Suggestions](#extension-suggestions)
- [Contributing](#contributing)
- [Contact & Support](#contact--support)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [Star History](#star-history)
- [Contributors](#contributors)
- [License](#license)

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Basic dependencies
pip install numpy

# Optional: OpenAI support
pip install openai python-dotenv

# Optional: Better vector embeddings
pip install sentence-transformers

# Optional: Enhanced experiments (statistical analysis and visualization)
pip install scipy matplotlib
```

### 2. Environment Configuration

Create a `.env` file:

```env
# OpenAI Official API
OPENAI_API_KEY=sk-your-key-here
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-3.5-turbo

# Or use Azure OpenAI
# OPENAI_API_KEY=your-azure-key
# OPENAI_API_BASE=https://your-resource.openai.azure.com
# OPENAI_MODEL=your-deployment-name

# Or use local models (e.g., Ollama)
# OPENAI_API_BASE=http://localhost:11434/v1
# OPENAI_MODEL=llama2
```

### 3. Run Experiments

```bash
# Basic experiment (4-round dialogue)
python cognitive_workspace_poc.py

# Enhanced experiments (10-round dialogue + multi-hop reasoning + conflict resolution)
python cognitive_workspace_enhanced.py
```

## 🎯 Operation Modes

### Mode 1: Full Mode (Recommended)
Requires OpenAI API key, demonstrates real LLM behavioral differences:
- Higher quality task decomposition
- More accurate information prediction
- More coherent answer generation

### Mode 2: Simulation Mode (Default)
No API key required, uses rule-based simulation:
- Still demonstrates architectural differences
- Suitable for proof-of-concept
- Fully reproducible

### Mode 3: Local Mode
Uses local models like Ollama:
- Data privacy
- No API costs
- Performance depends on local hardware

## 🔬 Experiment Content

### Experiment 1: Single-turn Task Processing
Compares Cognitive Workspace vs traditional RAG on single complex questions:
- Operation count difference (12 vs 3)
- Operation type difference (active vs passive)
- Memory management difference (hierarchical vs flat)
- Single-turn memory reuse rate: 50% vs 0%

### Experiment 2: Multi-turn Dialogue (Core Advantage)
Demonstrates cumulative advantages from state persistence:
```
Round  CW Reuse Rate  RAG Reuse Rate
1      50.0%         0%
2      55.0%         0%
3      56.7%         0%
4      56.4%         0%

Average reuse rate: 54.5% vs 0%
```

### Experiment 3: 10-round Extended Dialogue (Enhanced)
Memory advantages in long-term conversations:
```
Average reuse rate: 57.1% vs 0%
Net efficiency gain: 17.3%
Cohen's d: 23.2 (huge effect)
P-value: < 0.001 (extremely significant)
```

### Experiment 4: Multi-hop Reasoning (Enhanced)
Advantages in complex reasoning chains:
```
Average reuse rate: 58.8% vs 0%
Net efficiency gain: 17.9%
Cohen's d: 190.0 (extremely large effect)
Operations saved: 194
```

### Experiment 5: Information Conflict Resolution (Enhanced)
Performance when handling contradictory information:
```
Average reuse rate: 59.8% vs 0%
Net efficiency gain: 17.8%
Cohen's d: 195.7 (extremely large effect)
Operations saved: 226
```

## 📁 Output Files

- `cognitive_workspace_results.json`: Basic experiment results
- `enhanced_results.json`: Enhanced experiment detailed results
- `cognitive_workspace_analysis.png`: Experiment visualization charts
- `.env.example`: Environment variable template (if .env doesn't exist)

## 📊 Key Metrics Explanation

### Memory Reuse Rate (Measured Data)
- **Basic experiment (4 rounds)**: Average 54.5%, reuse starts from round 1
- **10-round dialogue**: Average 57.1%, long-term dialogue advantage clear
- **Multi-hop reasoning**: Average 58.8%, higher reuse rate for complex tasks
- **Conflict resolution**: Average 59.8%, best performance in information integration scenarios
- **Traditional RAG**: Always 0% (stateless)

### Net Efficiency Gain (After considering extra overhead)
```python
Net efficiency = Reuse rate / (1 + Extra operation ratio)
```
- **10-round dialogue**: 17.3% net improvement
- **Multi-hop reasoning**: 17.9% net improvement
- **Conflict resolution**: 17.8% net improvement

### Statistical Significance
- **P-values**: All experiments < 0.001 (extremely significant)
- **Cohen's d effect size**:
  - 10-round dialogue: 23.2 (huge)
  - Multi-hop reasoning: 190.0 (extremely large)
  - Conflict resolution: 195.7 (extremely large)

### Operation Growth Patterns
- **Cognitive Workspace**: Sub-linear growth (reduces redundant computation through memory reuse)
- **Traditional RAG**: Linear growth (starts fresh for each query)

### Confidence Tracking
- **Cognitive Workspace**: Dynamically tracks task completion and information sufficiency
- **Traditional RAG**: No confidence concept

## 🏗️ System Architecture

The Cognitive Workspace implements a hierarchical memory architecture that mimics human cognitive processes:

```
┌─────────────────────────────────────────────────────────┐
│              User Query Interface                       │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│         Metacognitive Controller                        │
│  • Task Decomposition                                   │
│  • Confidence Tracking                                  │
│  • Information Gap Analysis                             │
└────────────────┬────────────────────────────────────────┘
                 │
     ┌───────────┴───────────┐
     ▼                       ▼
┌─────────────┐      ┌──────────────────┐
│  Immediate  │      │  Active Memory   │
│   Buffer    │◄────►│   Prediction     │
│  (Recent)   │      │   & Preparation  │
└──────┬──────┘      └──────────────────┘
       │
       ▼
┌─────────────┐
│   Working   │
│   Buffer    │
│ (Relevant)  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Episodic   │
│   Buffer    │
│(Long-term)  │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│         External Knowledge Base (RAG)                   │
└─────────────────────────────────────────────────────────┘
```

**Key Differences from Traditional RAG:**
- **Stateful**: Maintains context across multiple queries
- **Predictive**: Proactively prepares information before it's needed
- **Hierarchical**: Three-tier memory buffer system with intelligent promotion
- **Metacognitive**: Self-aware of information gaps and task completion

> **Note**: You can add your own architecture diagrams by placing images in an `assets/` folder and referencing them here.

## 📸 Experiment Screenshots

When you run the experiments, you'll see outputs similar to these:

### Multi-turn Dialogue Comparison
The experiments generate detailed comparison charts showing memory reuse rates across different scenarios.

### Statistical Analysis Visualization
`cognitive_workspace_analysis.png` contains visualizations of:
- Memory reuse rate trends across dialogue rounds
- Operation count comparisons (Cognitive Workspace vs RAG)
- Statistical significance indicators (Cohen's d effect sizes)
- Net efficiency gains accounting for overhead

> **Tip**: After running `cognitive_workspace_enhanced.py`, check the generated `cognitive_workspace_analysis.png` file for detailed visual results.

## 📄 Paper Support

This code supports the following paper arguments:

1. **Active memory management outperforms passive retrieval**
   - Code proof: Task decomposition, information prediction, active preparation
   
2. **State persistence improves efficiency**
   - Code proof: Memory reuse in multi-turn dialogues

3. **Hierarchical buffers optimize resource utilization**
   - Code proof: immediate→working→episodic promotion mechanism

4. **Metacognitive control enhances intelligence**
   - Code proof: Confidence tracking, information gap identification

## ❓ FAQ

<details>
<summary><b>Q: Why can simulation mode also prove the points?</b></summary>

A: Because we prove architectural behavioral differences, not generation quality. Even with rule simulation, the differences between active vs passive, stateful vs stateless are still obvious.
</details>

<details>
<summary><b>Q: How to cite this code in papers?</b></summary>

A: Use the following format in your LaTeX:
```latex
Code available at: \url{https://github.com/tao-hpu/cognitive-workspace}
```
</details>

<details>
<summary><b>Q: How many tokens/API calls are needed?</b></summary>

A: Full experiments require approximately:
- Single-turn experiment: ~10 API calls
- Multi-turn experiment: ~20 API calls
- Total cost: < $0.05 (using GPT-3.5-turbo)
</details>

<details>
<summary><b>Q: Can other LLMs be used?</b></summary>

A: Yes! The code supports:
- OpenAI-compatible APIs (by modifying OPENAI_API_BASE)
- Local models (Ollama, llama.cpp)
- Any service providing chat/completion interfaces
</details>

## 🔧 Troubleshooting

### API Connection Errors

**Problem**: `openai.error.AuthenticationError` or connection timeout

**Solutions**:
- Verify your API key is correct in `.env`
- Check `OPENAI_API_BASE` URL format (should end with `/v1`)
- For Azure OpenAI, ensure you're using the correct endpoint format
- Test connection: `curl -H "Authorization: Bearer $OPENAI_API_KEY" $OPENAI_API_BASE/models`

### Import Errors for Optional Dependencies

**Problem**: `ModuleNotFoundError: No module named 'sentence_transformers'`

**Solutions**:
- Install missing dependencies: `pip install sentence-transformers`
- For full functionality: `pip install openai python-dotenv sentence-transformers scipy matplotlib`
- Check Python version (requires 3.7+)

### Results Differ from Expected Values

**Problem**: Reuse rates or metrics don't match documentation

**Solutions**:
- **Simulation mode** (no API key): Results are deterministic but simplified
- **Full mode** (with API key): Results vary slightly due to LLM randomness
  - Set temperature=0 in code for more consistent results
  - Run multiple trials for statistical validity
- Ensure you're comparing same experiment (basic vs enhanced)

### Memory or Performance Issues

**Problem**: Script runs slowly or uses too much memory

**Solutions**:
- Start with basic experiment first: `python cognitive_workspace_poc.py`
- Reduce number of documents in test data
- For local models, ensure adequate RAM (8GB+ recommended)
- Check if background processes are consuming resources

### Results Files Not Generated

**Problem**: Missing `.json` or `.png` output files

**Solutions**:
- Check for errors in console output
- Ensure write permissions in current directory
- For visualization: verify matplotlib is installed
- Run with: `python cognitive_workspace_enhanced.py 2>&1 | tee output.log`

## 💡 Extension Suggestions

1. **Add longer-term tests (20+ rounds)**
   ```python
   # Modify question list in cognitive_workspace_enhanced.py
   extended_questions = [...20 questions...]
   ```

2. **Integrate real vector databases**
   ```python
   # Use ChromaDB or Pinecone
   from chromadb import Client
   ```

3. **Add more statistical tests**
   ```python
   # Mann-Whitney U test, Friedman test, etc.
   from scipy import stats
   stats.mannwhitneyu(cw_results, rag_results)
   ```

4. **Performance benchmarking**
   ```python
   # Test performance at different scales
   for doc_count in [10, 100, 1000]:
       test_scalability(doc_count)
   ```

## 🤝 Contributing

We welcome contributions to improve this proof-of-concept implementation! Here's how you can help:

### Ways to Contribute

- **Bug Reports**: Open an issue describing the problem with steps to reproduce
- **Feature Suggestions**: Propose new experiments or architectural improvements
- **Code Improvements**: Submit pull requests for bug fixes or enhancements
- **Documentation**: Improve README, add code comments, or create tutorials
- **Testing**: Add test cases or validate results on different platforms

### Contribution Guidelines

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** with clear, descriptive commit messages
3. **Test your changes** thoroughly (run both basic and enhanced experiments)
4. **Update documentation** if you change functionality
5. **Submit a pull request** with a clear description of your changes

### Code of Conduct

- Be respectful and constructive in discussions
- Focus on the technical merits of contributions
- Help maintain this as a research and educational resource

## 📬 Contact & Support

### Getting Help

- **Issues**: For bug reports and feature requests, use [GitHub Issues](https://github.com/tao-hpu/cognitive-workspace/issues)
- **Discussions**: For questions and general discussion, start a [GitHub Discussion](https://github.com/tao-hpu/cognitive-workspace/discussions)
- **Documentation**: Check the [Wiki](https://github.com/tao-hpu/cognitive-workspace/wiki) for additional resources

### Research Collaboration

If you're interested in collaborating on research related to Cognitive Workspace or have academic questions about the paper:

- **Author**: Tao An
- **Paper**: [arXiv:2508.13171](https://arxiv.org/abs/2508.13171)
- For research inquiries, please reference the paper for contact information

### Reporting Security Issues

If you discover a security vulnerability, please report it privately rather than opening a public issue.

## 📖 Citation

If you use this code, please cite:

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

## 🙏 Acknowledgments

This proof-of-concept implementation was developed to demonstrate the architectural principles described in the Cognitive Workspace paper. We'd like to thank:

- **Research Community**: For valuable feedback on the paper and implementation
- **Open Source Contributors**: All contributors who have helped improve this codebase
- **Users & Testers**: Everyone who has experimented with this POC and provided insights
- **Reviewers**: For constructive comments that improved both the paper and code quality

Special thanks to the broader AI/ML research community for inspiring discussions on memory architectures, metacognition, and efficient LLM systems.

### Built With

This project leverages excellent open-source tools:
- **OpenAI API** - LLM integration
- **NumPy** - Numerical computing
- **Sentence Transformers** - Vector embeddings (optional)
- **SciPy & Matplotlib** - Statistical analysis and visualization (optional)

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=tao-hpu/cognitive-workspace&type=Date)](https://star-history.com/#tao-hpu/cognitive-workspace&Date)

## 👥 Contributors

Thanks to all the contributors who have helped improve this project!

[![Contributors](https://contrib.rocks/image?repo=tao-hpu/cognitive-workspace)](https://github.com/tao-hpu/cognitive-workspace/graphs/contributors)

### How to Contribute

We welcome contributions! See the [Contributing](#-contributing) section above for guidelines.

---

## 📜 License

MIT License - Free to use, modify and distribute

---

<div align="center">

**Made with ❤️ for the AI Research Community**

If you find this project useful, please consider giving it a ⭐!

[Report Bug](https://github.com/tao-hpu/cognitive-workspace/issues) · [Request Feature](https://github.com/tao-hpu/cognitive-workspace/issues) · [View Paper](https://arxiv.org/abs/2508.13171)

</div>
