# Cognitive Workspace - Proof of Concept Implementation

## Quick Start

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

## Operation Modes

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

## Experiment Content

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

## Output Files

- `cognitive_workspace_results.json`: Basic experiment results
- `enhanced_results.json`: Enhanced experiment detailed results
- `cognitive_workspace_analysis.png`: Experiment visualization charts
- `.env.example`: Environment variable template (if .env doesn't exist)

## Key Metrics Explanation

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

## Paper Support

This code supports the following paper arguments:

1. **Active memory management outperforms passive retrieval**
   - Code proof: Task decomposition, information prediction, active preparation
   
2. **State persistence improves efficiency**
   - Code proof: Memory reuse in multi-turn dialogues

3. **Hierarchical buffers optimize resource utilization**
   - Code proof: immediate→working→episodic promotion mechanism

4. **Metacognitive control enhances intelligence**
   - Code proof: Confidence tracking, information gap identification

## FAQ

### Q: Why can simulation mode also prove the points?
A: Because we prove architectural behavioral differences, not generation quality. Even with rule simulation, the differences between active vs passive, stateful vs stateless are still obvious.

### Q: How to cite this code in papers?
A: 
```latex
Code available at: \url{https://github.com/tao-hpu/cognitive-workspace}
```

### Q: How many tokens/API calls are needed?
A: Full experiments require approximately:
- Single-turn experiment: ~10 API calls
- Multi-turn experiment: ~20 API calls
- Total cost: < $0.05 (using GPT-3.5-turbo)

### Q: Can other LLMs be used?
A: Yes! The code supports:
- OpenAI-compatible APIs (by modifying OPENAI_API_BASE)
- Local models (Ollama, llama.cpp)
- Any service providing chat/completion interfaces

## Extension Suggestions

1. **Add longer-term tests (20+ rounds)**
   ```python
   # Modify question list in enhanced_experiment.py
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

## Citation

If you use this code, please cite:

```bibtex
@article{cognitive-workspace-2025,
  title={Cognitive Workspace: Towards Functional Infinite Context Through Active Memory Management},
  author={Tao An},
  journal={arXiv preprint arXiv:2025.xxxxx},
  year={2025}
}
```

## License

MIT License - Free to use, modify and distribute