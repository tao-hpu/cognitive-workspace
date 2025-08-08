#!/usr/bin/env python3
"""
认知工作空间 - 增强实验版本
包含更复杂的测试案例和统计分析
"""

import json
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import os
from enum import Enum
import hashlib

# 尝试导入高级分析库
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("提示: 安装scipy以启用统计显著性测试 (pip install scipy)")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("提示: 安装matplotlib以启用可视化 (pip install matplotlib)")

# 复用原始代码的核心类
import sys
sys.path.append('.')
from cognitive_workspace_poc import (
    CognitiveWorkspace,
    TraditionalRAG,
    MemoryItem,
    CognitiveState,
    LLMInterface,
    VectorStore,
    LLMBackend  # 使用POC版本的枚举！
)


# ============================================
# 增强的实验类
# ============================================

class EnhancedExperiment:
    """增强实验：更复杂的测试案例和深度分析"""
    
    def __init__(self):
        self.cw = CognitiveWorkspace(use_vectors=True)
        self.rag = TraditionalRAG()
        self.results = {
            "experiments": [],
            "statistics": {},
            "efficiency_analysis": {}
        }
    
    def run_extended_conversation(self, questions: List[str], documents: List[str], 
                                 experiment_name: str = "extended") -> Dict:
        """运行扩展的多轮对话（支持10+轮）"""
        
        print(f"\n运行实验: {experiment_name}")
        print("=" * 60)
        
        cw_results = []
        rag_results = []
        cw_reuse_rates = []
        rag_reuse_rates = []
        cw_ops_counts = []
        rag_ops_counts = []
        
        for i, question in enumerate(questions):
            print(f"\n轮次 {i+1}/{len(questions)}: {question[:50]}...")
            
            # 认知工作空间处理
            cw_start = time.time()
            cw_result = self.cw.process_task(question, documents)
            cw_time = time.time() - cw_start
            
            cw_results.append(cw_result)
            cw_reuse_rates.append(cw_result.get('memory_reuse_rate', 0))
            cw_ops_counts.append(len(cw_result.get('operations', [])))
            
            # 传统RAG处理
            rag_start = time.time()
            rag_result = self.rag.process_task(question, documents)
            rag_time = time.time() - rag_start
            
            rag_results.append(rag_result)
            rag_reuse_rates.append(0.0)  # RAG永远是0
            rag_ops_counts.append(len(rag_result.get('operations', [])))
            
            # 打印实时对比
            print(f"  CW: 复用率={cw_reuse_rates[-1]:.1%}, 操作数={cw_ops_counts[-1]}, 时间={cw_time:.2f}s")
            print(f"  RAG: 复用率=0.0%, 操作数={rag_ops_counts[-1]}, 时间={rag_time:.2f}s")
        
        # 计算统计数据
        experiment_result = {
            "name": experiment_name,
            "rounds": len(questions),
            "cw_reuse_rates": cw_reuse_rates,
            "rag_reuse_rates": rag_reuse_rates,
            "cw_ops_counts": cw_ops_counts,
            "rag_ops_counts": rag_ops_counts,
            "cw_avg_reuse": np.mean(cw_reuse_rates),
            "cw_std_reuse": np.std(cw_reuse_rates),
            "cw_total_ops": sum(cw_ops_counts),
            "rag_total_ops": sum(rag_ops_counts),
            "ops_ratio": sum(cw_ops_counts) / sum(rag_ops_counts) if sum(rag_ops_counts) > 0 else 0
        }
        
        self.results["experiments"].append(experiment_result)
        return experiment_result
    
    def run_multi_hop_reasoning(self, documents: List[str]) -> Dict:
        """多跳推理测试案例"""
        
        # 构建需要多跳推理的问题序列
        multi_hop_questions = [
            "What is machine learning?",
            "How does deep learning differ from traditional machine learning?",
            "What role do neural networks play in deep learning?",
            "How do attention mechanisms improve neural networks?",
            "Can transfer learning be combined with attention mechanisms?",
            "What are the computational requirements for these combined approaches?",
            "How do these requirements compare to simpler models?",
            "What trade-offs exist between model complexity and performance?",
            "In what scenarios would simpler models be preferred?",
            "How can we optimize the balance between complexity and efficiency?"
        ]
        
        return self.run_extended_conversation(
            multi_hop_questions, 
            documents,
            "multi_hop_reasoning"
        )
    
    def run_information_conflict_resolution(self, documents: List[str]) -> Dict:
        """信息冲突解决测试案例"""
        
        # 添加一些包含冲突信息的文档
        conflicting_docs = documents + [
            "Some experts argue that deep learning is overhyped and traditional methods are more reliable.",
            "Recent studies show that simple linear models outperform complex neural networks in many cases.",
            "Deep learning has revolutionized AI and made traditional methods obsolete.",
            "The effectiveness of AI techniques heavily depends on the specific problem domain."
        ]
        
        conflict_questions = [
            "Is deep learning always better than traditional methods?",
            "What evidence supports the superiority of neural networks?",
            "What are the counterarguments against deep learning?",
            "How can we reconcile these conflicting viewpoints?",
            "What factors determine the best approach for a given problem?",
            "Can you synthesize a balanced perspective on this debate?"
        ]
        
        return self.run_extended_conversation(
            conflict_questions,
            conflicting_docs,
            "conflict_resolution"
        )
    
    def calculate_efficiency_metrics(self) -> Dict:
        """计算详细的效率指标"""
        
        print("\n\n" + "=" * 60)
        print("效率分析")
        print("=" * 60)
        
        for exp in self.results["experiments"]:
            name = exp["name"]
            
            # 基础指标
            avg_reuse = exp["cw_avg_reuse"]
            ops_ratio = exp["ops_ratio"]
            
            # 计算净效率提升
            # efficiency_gain = reuse_rate / (1 + extra_ops_ratio)
            extra_ops_ratio = ops_ratio - 1  # CW相对于RAG的额外操作比例
            net_efficiency = avg_reuse / (1 + extra_ops_ratio) if ops_ratio > 0 else 0
            
            # 计算累积优势
            cumulative_savings = 0
            for i, reuse_rate in enumerate(exp["cw_reuse_rates"]):
                # 每轮节省的操作数 = 基础操作数 * 复用率
                base_ops = exp["rag_ops_counts"][0] if exp["rag_ops_counts"] else 3
                saved_ops = base_ops * reuse_rate
                cumulative_savings += saved_ops
            
            efficiency_metrics = {
                "average_reuse_rate": avg_reuse,
                "operations_ratio": ops_ratio,
                "extra_operations_ratio": extra_ops_ratio,
                "net_efficiency_gain": net_efficiency,
                "cumulative_operations_saved": cumulative_savings,
                "break_even_round": self._calculate_break_even(exp)
            }
            
            self.results["efficiency_analysis"][name] = efficiency_metrics
            
            # 打印结果
            print(f"\n实验: {name}")
            print("-" * 40)
            print(f"平均复用率: {avg_reuse:.1%}")
            print(f"操作比率 (CW/RAG): {ops_ratio:.2f}")
            print(f"净效率提升: {net_efficiency:.1%}")
            print(f"累积节省操作数: {cumulative_savings:.0f}")
            print(f"收益平衡点: 第{efficiency_metrics['break_even_round']}轮")
        
        return self.results["efficiency_analysis"]
    
    def _calculate_break_even(self, experiment: Dict) -> int:
        """计算CW开始产生净收益的轮次"""
        
        cumulative_cw = 0
        cumulative_rag = 0
        
        for i in range(len(experiment["cw_ops_counts"])):
            cumulative_cw += experiment["cw_ops_counts"][i]
            cumulative_rag += experiment["rag_ops_counts"][i]
            
            # 考虑复用带来的实际价值
            effective_cw = cumulative_cw * (1 - experiment["cw_reuse_rates"][i] * 0.5)
            
            if effective_cw <= cumulative_rag:
                return i + 1
        
        return len(experiment["cw_ops_counts"])
    
    def run_statistical_tests(self) -> Dict:
        """运行统计显著性测试"""
        
        if not HAS_SCIPY:
            print("\n⚠ 需要scipy库来运行统计测试")
            return {}
        
        print("\n\n" + "=" * 60)
        print("统计显著性分析")
        print("=" * 60)
        
        statistical_results = {}
        
        for exp in self.results["experiments"]:
            name = exp["name"]
            
            # T-test: CW复用率是否显著大于0
            cw_reuse = exp["cw_reuse_rates"]
            t_stat, p_value = stats.ttest_1samp(cw_reuse, 0)
            
            # Wilcoxon signed-rank test (非参数检验)
            # 比较CW和RAG的性能差异
            if len(cw_reuse) > 1:
                wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
                    cw_reuse,
                    [0] * len(cw_reuse)
                )
            else:
                wilcoxon_stat, wilcoxon_p = None, None
            
            # Cohen's d 效应量
            cohens_d = np.mean(cw_reuse) / np.std(cw_reuse) if np.std(cw_reuse) > 0 else 0
            
            statistical_results[name] = {
                "t_statistic": t_stat,
                "p_value": p_value,
                "wilcoxon_statistic": wilcoxon_stat,
                "wilcoxon_p_value": wilcoxon_p,
                "cohens_d": cohens_d,
                "significance": "显著" if p_value < 0.05 else "不显著"
            }
            
            print(f"\n实验: {name}")
            print("-" * 40)
            print(f"T检验统计量: {t_stat:.4f}")
            print(f"P值: {p_value:.6f}")
            print(f"Cohen's d: {cohens_d:.4f}")
            print(f"结论: {statistical_results[name]['significance']} (α=0.05)")
            
            if cohens_d > 0.8:
                print("效应量: 大")
            elif cohens_d > 0.5:
                print("效应量: 中等")
            elif cohens_d > 0.2:
                print("效应量: 小")
            else:
                print("效应量: 极小")
        
        self.results["statistics"] = statistical_results
        return statistical_results
    
    def generate_visualizations(self):
        """生成可视化图表"""
        
        if not HAS_MATPLOTLIB:
            print("\n⚠ 需要matplotlib库来生成可视化")
            return
        
        print("\n生成可视化图表...")
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 复用率随轮次变化
        ax1 = axes[0, 0]
        for exp in self.results["experiments"]:
            rounds = range(1, len(exp["cw_reuse_rates"]) + 1)
            ax1.plot(rounds, exp["cw_reuse_rates"], marker='o', label=f"CW-{exp['name']}")
        ax1.axhline(y=0, color='r', linestyle='--', label='RAG (always 0)')
        ax1.set_xlabel('Conversation Round')
        ax1.set_ylabel('Memory Reuse Rate')
        ax1.set_title('Memory Reuse Rate Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 累积操作数对比
        ax2 = axes[0, 1]
        for exp in self.results["experiments"]:
            cw_cumsum = np.cumsum(exp["cw_ops_counts"])
            rag_cumsum = np.cumsum(exp["rag_ops_counts"])
            rounds = range(1, len(cw_cumsum) + 1)
            ax2.plot(rounds, cw_cumsum, marker='s', label=f"CW-{exp['name']}")
            ax2.plot(rounds, rag_cumsum, marker='^', label=f"RAG-{exp['name']}")
        ax2.set_xlabel('Conversation Round')
        ax2.set_ylabel('Cumulative Operations')
        ax2.set_title('Operations Growth Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 效率对比柱状图
        ax3 = axes[1, 0]
        if self.results["efficiency_analysis"]:
            names = list(self.results["efficiency_analysis"].keys())
            net_efficiency = [self.results["efficiency_analysis"][n]["net_efficiency_gain"] 
                            for n in names]
            ax3.bar(names, net_efficiency)
            ax3.set_ylabel('Net Efficiency Gain')
            ax3.set_title('Net Efficiency Comparison')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. 统计显著性热图
        ax4 = axes[1, 1]
        if self.results["statistics"]:
            exp_names = list(self.results["statistics"].keys())
            p_values = [self.results["statistics"][n]["p_value"] for n in exp_names]
            cohens_d = [self.results["statistics"][n]["cohens_d"] for n in exp_names]
            
            # 创建热图数据
            heatmap_data = np.array([p_values, cohens_d])
            im = ax4.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto')
            
            ax4.set_xticks(range(len(exp_names)))
            ax4.set_xticklabels(exp_names, rotation=45)
            ax4.set_yticks([0, 1])
            ax4.set_yticklabels(['P-value', "Cohen's d"])
            ax4.set_title('Statistical Metrics Heatmap')
            
            # 添加数值标签
            for i in range(2):
                for j in range(len(exp_names)):
                    text = ax4.text(j, i, f'{heatmap_data[i, j]:.3f}',
                                  ha="center", va="center", color="black")
            
            plt.colorbar(im, ax=ax4)
        
        plt.tight_layout()
        plt.savefig('cognitive_workspace_analysis.png', dpi=150, bbox_inches='tight')
        print("图表已保存为: cognitive_workspace_analysis.png")
        plt.show()
    
    def save_detailed_results(self, filename: str = "enhanced_results.json"):
        """保存详细结果"""
        
        # 确保所有numpy类型转换为Python原生类型
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            return obj
        
        # 递归转换结果
        def clean_results(d):
            if isinstance(d, dict):
                return {k: clean_results(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [clean_results(item) for item in d]
            else:
                return convert_numpy(d)
        
        cleaned_results = clean_results(self.results)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(cleaned_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n详细结果已保存到: {filename}")


# ============================================
# 主程序
# ============================================

def main():
    """运行增强实验套件"""
    
    print("=" * 60)
    print("认知工作空间 - 增强实验套件")
    print("=" * 60)
    
    # 准备测试文档
    documents = [
        "Machine learning algorithms learn patterns from data without explicit programming.",
        "Deep learning uses neural networks with multiple layers to process complex patterns.",
        "Natural language processing enables computers to understand and generate human language.",
        "Computer vision allows machines to interpret and analyze visual information.",
        "Reinforcement learning trains agents through reward and punishment signals.",
        "Transfer learning reuses knowledge from one task to improve performance on another.",
        "Attention mechanisms help models focus on relevant parts of the input.",
        "Generative models can create new data similar to their training data.",
        "Federated learning enables training on distributed data while preserving privacy.",
        "Meta-learning helps models learn how to learn more efficiently.",
        "Ensemble methods combine multiple models to improve prediction accuracy.",
        "Active learning selects the most informative samples for labeling.",
        "Few-shot learning enables models to learn from very few examples.",
        "Continual learning allows models to learn new tasks without forgetting old ones.",
        "Explainable AI makes model decisions interpretable to humans."
    ]
    
    # 初始化实验
    experiment = EnhancedExperiment()
    
    # 配置后端
    try:
        from dotenv import load_dotenv
        load_dotenv()
        if os.getenv("OPENAI_API_KEY"):
            backend = LLMBackend.OPENAI  # 使用正确的枚举类型！
            print("✓ 使用OpenAI后端")
        else:
            backend = LLMBackend.MOCK
            print("✓ 使用模拟后端")
    except:
        backend = LLMBackend.MOCK
        print("✓ 使用模拟后端")
    
    experiment.cw.llm = LLMInterface(backend=backend)
    experiment.rag.llm = LLMInterface(backend=backend)
    
    # 运行实验1：标准10轮对话
    print("\n" + "=" * 60)
    print("实验1: 10轮扩展对话")
    print("=" * 60)
    
    extended_questions = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are neural networks?",
        "Explain deep learning",
        "What is natural language processing?",
        "How does computer vision work?",
        "What is reinforcement learning?",
        "Explain transfer learning",
        "What are attention mechanisms?",
        "How do these techniques work together?"
    ]
    
    experiment.run_extended_conversation(extended_questions, documents, "10_round_conversation")
    
    # 运行实验2：多跳推理
    print("\n" + "=" * 60)
    print("实验2: 多跳推理测试")
    print("=" * 60)
    
    experiment.run_multi_hop_reasoning(documents)
    
    # 运行实验3：信息冲突解决
    print("\n" + "=" * 60)
    print("实验3: 信息冲突解决")
    print("=" * 60)
    
    experiment.run_information_conflict_resolution(documents)
    
    # 计算效率指标
    experiment.calculate_efficiency_metrics()
    
    # 运行统计测试
    experiment.run_statistical_tests()
    
    # 生成可视化
    experiment.generate_visualizations()
    
    # 保存结果
    experiment.save_detailed_results()
    
    # 打印总结
    print("\n" + "=" * 60)
    print("实验总结")
    print("=" * 60)
    
    for exp_name, metrics in experiment.results["efficiency_analysis"].items():
        print(f"\n{exp_name}:")
        print(f"  - 平均复用率: {metrics['average_reuse_rate']:.1%}")
        print(f"  - 净效率提升: {metrics['net_efficiency_gain']:.1%}")
        print(f"  - 节省操作数: {metrics['cumulative_operations_saved']:.0f}")
    
    if experiment.results["statistics"]:
        print("\n统计显著性:")
        for exp_name, stats in experiment.results["statistics"].items():
            print(f"  {exp_name}: p={stats['p_value']:.4f} ({stats['significance']})")
    
    print("\n✓ 所有实验完成！")


if __name__ == "__main__":
    main()