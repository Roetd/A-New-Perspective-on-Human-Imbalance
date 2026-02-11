#!/usr/bin/env python3
"""
拓扑分析与模拟脚本
实现 homology group 理论和 Metropolis 算法，构建三阶段拓扑修复机制
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime

# 导入同调群计算模块
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from 同调群计算 import calculate_betti_numbers, calculate_imbalance_index, calculate_topology_energy, calculate_resonance_consistency

# 数据路径
RAW_DATA_DIR = r'D:\桌面\拓扑波动AI架构\人类失衡研究\原始资料'
PROCESSED_DATA_DIR = r'D:\桌面\拓扑波动AI架构\人类失衡研究\数据处理'
RESULTS_DIR = r'D:\桌面\拓扑波动AI架构\人类失衡研究\数据结论'

# 确保目录存在
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 加载拓扑网络数据
def load_topology_data():
    """加载拓扑网络数据"""
    nodes_path = os.path.join(RAW_DATA_DIR, "topology_nodes_simulation_data.csv")
    edges_path = os.path.join(RAW_DATA_DIR, "topology_edges_simulation_data.csv")
    
    nodes_df = pd.read_csv(nodes_path, encoding='utf-8-sig')
    edges_df = pd.read_csv(edges_path, encoding='utf-8-sig')
    
    return nodes_df, edges_df

# 创建网络x图
def create_graph(nodes_df, edges_df):
    """创建网络x图"""
    G = nx.Graph()
    
    # 添加节点
    for _, node in nodes_df.iterrows():
        G.add_node(node['node_id'], 
                   type=node['node_type'],
                   strength=node['strength'],
                   stability=node['stability'],
                   imbalance=node['imbalance'])
    
    # 添加边
    for _, edge in edges_df.iterrows():
        G.add_edge(edge['source'], edge['target'],
                   weight=edge['weight'],
                   stability=edge['stability'],
                   imbalance=edge['imbalance'])
    
    return G

# 计算拓扑指标
def calculate_topology_metrics(G):
    """计算拓扑指标"""
    metrics = {
        "nodes": len(G.nodes()),
        "edges": len(G.edges()),
        "density": nx.density(G),
        "average_clustering": nx.average_clustering(G),
        "average_degree": sum(dict(G.degree()).values()) / len(G.nodes())
    }
    
    # 计算失衡相关指标
    node_imbalance = [data['imbalance'] for _, data in G.nodes(data=True)]
    edge_imbalance = [data['imbalance'] for _, _, data in G.edges(data=True)]
    
    metrics["avg_node_imbalance"] = np.mean(node_imbalance)
    metrics["avg_edge_imbalance"] = np.mean(edge_imbalance)
    metrics["total_imbalance"] = np.mean(node_imbalance + edge_imbalance)
    
    return metrics

# 实现 homology group 理论分析
def homology_analysis(G):
    """基于 homology group 理论的拓扑分析"""
    print("执行 homology group 理论分析...")
    
    # 计算连通分量（0阶同调群）
    components = list(nx.connected_components(G))
    h0 = len(components)
    
    # 计算环的数量（1阶同调群的近似）
    # 对于平面图，使用 Euler 公式: V - E + F = C + H1
    # 这里使用近似方法
    V = len(G.nodes())
    E = len(G.edges())
    C = len(components)
    
    # 近似计算1阶同调群的秩
    # 对于连通图，H1 = E - V + 1
    h1 = max(0, E - V + C)
    
    # 计算 Betti 数
    betti_numbers = {
        "h0": h0,  # 连通分量数
        "h1": h1,  # 环的数量
        "h2": 0    # 对于平面图，h2=0
    }
    
    print(f"Betti 数: h0={h0}, h1={h1}, h2=0")
    
    # 计算拓扑链接的完整性
    # 基于连通性和环的分布
    connectivity = nx.average_node_connectivity(G)
    integrity = {
        "connectivity": connectivity,
        "component_size_distribution": [len(comp) for comp in components],
        "betti_numbers": betti_numbers
    }
    
    return integrity

# 实现 Metropolis 算法模拟拓扑波动
def metropolis_simulation(G, iterations=5000, temperature=0.5, convergence_threshold=1e-6, intervention_interval=1000):
    """使用 Metropolis 算法模拟拓扑波动
    
    参数:
        G: NetworkX图对象
        iterations: 迭代次数
        temperature: 热波动因子 (k_BT)
        convergence_threshold: 收敛阈值
        intervention_interval: 干预间隔
    
    返回:
        current_G: 模拟后的图
        energy_history: 能量历史
        imbalance_history: 失衡历史
        betti_history: 贝蒂数历史
        resonance_history: 共振一致性历史
    """
    print("执行 Metropolis 算法模拟拓扑波动...")
    print(f"参数设置: 迭代次数={iterations}, 温度={temperature}, 收敛阈值={convergence_threshold}")
    
    # 初始化
    current_G = G.copy()
    current_energy = calculate_energy(current_G)
    energy_history = [current_energy]
    
    # 计算初始失衡值
    avg_imbalance = np.mean([data['imbalance'] for _, data in current_G.nodes(data=True)])
    imbalance_history = [avg_imbalance]
    
    # 计算初始贝蒂数
    betti = calculate_betti_numbers(current_G)
    betti_history = [{"h0": betti['h0'], "h1": betti['h1'], "h2": betti['h2']}]
    
    # 计算初始共振一致性
    resonance = calculate_resonance_consistency(current_G)
    resonance_history = [resonance]
    
    # 模拟过程
    for i in range(iterations):
        # 生成新状态
        new_G = propose_new_state(current_G)
        
        # 计算新状态的能量
        new_energy = calculate_energy(new_G)
        
        # 计算能量差
        delta_energy = new_energy - current_energy
        
        # Metropolis 准则
        if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / temperature):
            current_G = new_G
            current_energy = new_energy
        
        # 定期干预
        if (i + 1) % intervention_interval == 0:
            print(f"迭代 {i+1}: 执行干预...")
            # 识别关键节点并修复
            key_nodes = identify_key_nodes(current_G, top_k=5)
            current_G = repair_connections(current_G, key_nodes)
            # 重新计算能量
            current_energy = calculate_energy(current_G)
        
        # 记录历史（每100次迭代）
        if i % 100 == 0:
            energy_history.append(current_energy)
            avg_imbalance = np.mean([data['imbalance'] for _, data in current_G.nodes(data=True)])
            imbalance_history.append(avg_imbalance)
            
            # 计算贝蒂数
            betti = calculate_betti_numbers(current_G)
            betti_history.append({"h0": betti['h0'], "h1": betti['h1'], "h2": betti['h2']})
            
            # 计算共振一致性
            resonance = calculate_resonance_consistency(current_G)
            resonance_history.append(resonance)
        
        # 检查收敛
        if len(energy_history) > 10:
            recent_energies = energy_history[-10:]
            energy_std = np.std(recent_energies)
            if energy_std < convergence_threshold:
                print(f"模拟在迭代 {i} 处收敛")
                break
    
    # 保存模拟结果
    save_simulation_results(energy_history, imbalance_history, betti_history, resonance_history)
    
    return current_G, energy_history, imbalance_history, betti_history, resonance_history

# 计算系统能量
def calculate_energy(G):
    """计算系统能量"""
    # 使用拓扑能量E的计算公式
    energy = calculate_topology_energy(G)
    # 转换为能量值（低能量表示更稳定）
    # 拓扑能量E范围是0-1，其中1表示完全平衡
    # 我们需要将其转换为能量值，使得平衡状态能量更低
    energy = 1 - energy
    return energy

# 保存模拟结果
def save_simulation_results(energy_history, imbalance_history, betti_history, resonance_history):
    """保存模拟结果到文件"""
    print("保存模拟结果...")
    
    # 创建结果目录
    results_dir = os.path.join(RESULTS_DIR, "模拟结果")
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存能量历史
    energy_df = pd.DataFrame({
        "iteration": range(len(energy_history)),
        "energy": energy_history
    })
    energy_df.to_csv(os.path.join(results_dir, 'energy_history.csv'), index=False, encoding='utf-8-sig')
    
    # 保存失衡历史
    imbalance_df = pd.DataFrame({
        "iteration": range(len(imbalance_history)),
        "imbalance": imbalance_history
    })
    imbalance_df.to_csv(os.path.join(results_dir, 'imbalance_history.csv'), index=False, encoding='utf-8-sig')
    
    # 保存贝蒂数历史
    betti_df = pd.DataFrame(betti_history)
    betti_df['iteration'] = range(len(betti_history))
    betti_df.to_csv(os.path.join(results_dir, 'betti_history.csv'), index=False, encoding='utf-8-sig')
    
    # 保存共振一致性历史
    resonance_df = pd.DataFrame({
        "iteration": range(len(resonance_history)),
        "resonance": resonance_history
    })
    resonance_df.to_csv(os.path.join(results_dir, 'resonance_history.csv'), index=False, encoding='utf-8-sig')
    
    print(f"模拟结果保存完成，保存到: {results_dir}")

# 生成新状态
def propose_new_state(G):
    """生成新状态"""
    new_G = G.copy()
    
    # 随机选择一个操作：修改节点或边
    if np.random.rand() < 0.5:
        # 修改节点
        node_id = np.random.choice(list(new_G.nodes()))
        current_imbalance = new_G.nodes[node_id]['imbalance']
        # 小幅度随机调整失衡程度
        new_imbalance = max(0, min(1, current_imbalance + np.random.normal(0, 0.1)))
        new_G.nodes[node_id]['imbalance'] = new_imbalance
    else:
        # 修改边
        if len(new_G.edges()) > 0:
            edges_list = list(new_G.edges())
            edge_idx = np.random.randint(0, len(edges_list))
            edge = edges_list[edge_idx]
            current_imbalance = new_G.edges[edge]['imbalance']
            # 小幅度随机调整失衡程度
            new_imbalance = max(0, min(1, current_imbalance + np.random.normal(0, 0.1)))
            new_G.edges[edge]['imbalance'] = new_imbalance
    
    return new_G

# 构建三阶段拓扑修复机制
def topological_repair_mechanism(G):
    """构建三阶段拓扑修复机制"""
    print("执行三阶段拓扑修复机制...")
    
    # 阶段1：意识锚定（识别关键节点）
    print("阶段1：意识锚定")
    key_nodes = identify_key_nodes(G)
    
    # 阶段2：链接缝合（修复断裂的连接）
    print("阶段2：链接缝合")
    repaired_G = repair_connections(G, key_nodes)
    
    # 阶段3：共振再平衡（系统再平衡）
    print("阶段3：共振再平衡")
    balanced_G, energy_history, imbalance_history, betti_history, resonance_history = metropolis_simulation(
        repaired_G, 
        iterations=5000, 
        temperature=0.5, 
        convergence_threshold=1e-6, 
        intervention_interval=1000
    )
    
    return balanced_G, energy_history, imbalance_history, key_nodes

# 识别关键节点
def identify_key_nodes(G, top_k=5):
    """识别关键节点"""
    # 使用多种中心性指标综合评估
    betweenness = nx.betweenness_centrality(G, weight='weight')
    degree = dict(G.degree(weight='weight'))
    strength = {node: data['strength'] for node, data in G.nodes(data=True)}
    
    # 综合得分
    scores = {}
    for node in G.nodes():
        scores[node] = (betweenness[node] * 0.4 + 
                       degree[node] / max(degree.values()) * 0.3 + 
                       strength[node] / max(strength.values()) * 0.3)
    
    # 排序并选择前k个
    key_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    key_nodes = [node for node, _ in key_nodes]
    
    print(f"识别的关键节点: {key_nodes}")
    return key_nodes

# 修复连接
def repair_connections(G, key_nodes):
    """修复断裂的连接"""
    repaired_G = G.copy()
    
    # 为关键节点之间添加连接
    for i in range(len(key_nodes)):
        for j in range(i+1, len(key_nodes)):
            node1 = key_nodes[i]
            node2 = key_nodes[j]
            if not repaired_G.has_edge(node1, node2):
                # 添加新边
                weight = np.random.randint(5, 10)
                stability = np.random.uniform(0.7, 0.9)
                imbalance = np.random.uniform(0.1, 0.3)
                repaired_G.add_edge(node1, node2, 
                                   weight=weight,
                                   stability=stability,
                                   imbalance=imbalance)
    
    print(f"修复后边数: {len(repaired_G.edges())}")
    return repaired_G

# 系统再平衡
def balance_system(G, iterations=500, temperature=0.5):
    """系统再平衡"""
    # 使用降温的 Metropolis 算法
    current_G = G.copy()
    energy_history = []
    imbalance_history = []
    
    for i in range(iterations):
        # 逐渐降低温度
        current_temp = temperature * (1 - i / iterations)
        
        # 生成新状态
        new_G = propose_new_state(current_G)
        
        # 计算能量
        current_energy = calculate_energy(current_G)
        new_energy = calculate_energy(new_G)
        
        # Metropolis 准则
        if new_energy < current_energy or np.random.rand() < np.exp(-(new_energy - current_energy) / current_temp):
            current_G = new_G
        
        # 记录历史
        if i % 10 == 0:
            energy_history.append(calculate_energy(current_G))
            avg_imbalance = np.mean([data['imbalance'] for _, data in current_G.nodes(data=True)])
            imbalance_history.append(avg_imbalance)
    
    return current_G, energy_history, imbalance_history

# 生成图表
def generate_plots(energy_history, imbalance_history, initial_metrics, final_metrics, integrity, key_nodes, G, balanced_G):
    """生成图表"""
    print("生成分析图表...")
    
    # 创建结果目录
    plot_dir = os.path.join(RESULTS_DIR, "图表")
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. 能量和失衡历史
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.plot(energy_history)
    plt.title('Energy History')
    plt.xlabel('Iterations')
    plt.ylabel('Energy')
    
    plt.subplot(122)
    plt.plot(imbalance_history)
    plt.title('Imbalance History')
    plt.xlabel('Iterations')
    plt.ylabel('Average Imbalance')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'energy_imbalance_history.png'))
    plt.close()
    
    # 2. 拓扑指标对比
    metrics_to_compare = ['density', 'average_clustering', 'average_degree', 'avg_node_imbalance', 'avg_edge_imbalance', 'total_imbalance']
    labels = ['Density', 'Clustering', 'Average Degree', 'Node Imbalance', 'Edge Imbalance', 'Total Imbalance']
    
    initial_values = [initial_metrics[metric] for metric in metrics_to_compare]
    final_values = [final_metrics[metric] for metric in metrics_to_compare]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.figure(figsize=(14, 8))
    plt.bar(x - width/2, initial_values, width, label='Before Repair')
    plt.bar(x + width/2, final_values, width, label='After Repair')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Topology Metrics Comparison')
    plt.xticks(x, labels, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'topology_metrics_comparison.png'))
    plt.close()
    
    # 3. 网络结构图
    # 只绘制部分节点以避免过于密集
    sample_size = min(30, len(G.nodes()))
    sample_nodes = list(G.nodes())[:sample_size]
    
    # 修复前的网络结构
    subG = G.subgraph(sample_nodes)
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(subG, k=0.3)
    node_colors = [data['imbalance'] for _, data in subG.nodes(data=True)]
    edge_widths = [data['weight']/10 for _, _, data in subG.edges(data=True)]
    
    # 创建节点并获取mappable对象
    nodes = nx.draw_networkx_nodes(subG, pos, node_size=300, cmap=plt.cm.Reds, node_color=node_colors)
    nx.draw_networkx_edges(subG, pos, width=edge_widths, alpha=0.5)
    nx.draw_networkx_labels(subG, pos, font_size=8)
    plt.title('Network Structure Before Repair')
    plt.colorbar(nodes, label='Imbalance')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'network_structure_before.png'))
    plt.close()
    
    # 修复后的网络结构
    sub_balanced_G = balanced_G.subgraph(sample_nodes)
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(sub_balanced_G, k=0.3)
    node_colors = [data['imbalance'] for _, data in sub_balanced_G.nodes(data=True)]
    edge_widths = [data['weight']/10 for _, _, data in sub_balanced_G.edges(data=True)]
    
    # 创建节点并获取mappable对象
    nodes = nx.draw_networkx_nodes(sub_balanced_G, pos, node_size=300, cmap=plt.cm.Reds, node_color=node_colors)
    nx.draw_networkx_edges(sub_balanced_G, pos, width=edge_widths, alpha=0.5)
    nx.draw_networkx_labels(sub_balanced_G, pos, font_size=8)
    plt.title('Network Structure After Repair')
    plt.colorbar(nodes, label='Imbalance')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'network_structure_after.png'))
    plt.close()
    
    # 4. Betti 数可视化
    betti_values = list(integrity['betti_numbers'].values())
    betti_labels = ['h0 (Components)', 'h1 (Cycles)', 'h2 (Void)']
    
    plt.figure(figsize=(10, 6))
    plt.bar(betti_labels, betti_values, color=['blue', 'green', 'red'])
    plt.xlabel('Betti Numbers')
    plt.ylabel('Values')
    plt.title('Betti Numbers Visualization')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'betti_numbers_visualization.png'))
    plt.close()
    
    # 5. 拓扑指标
    metrics_df = pd.DataFrame([final_metrics])
    metrics_df.to_csv(os.path.join(RESULTS_DIR, 'topology_metrics.csv'), index=False, encoding='utf-8-sig')
    
    # 6. Betti 数
    betti_df = pd.DataFrame([integrity['betti_numbers']])
    betti_df.to_csv(os.path.join(RESULTS_DIR, 'betti_numbers.csv'), index=False, encoding='utf-8-sig')
    
    # 7. 关键节点分析
    key_nodes_df = pd.DataFrame({'key_node': key_nodes})
    key_nodes_df.to_csv(os.path.join(RESULTS_DIR, 'key_nodes.csv'), index=False, encoding='utf-8-sig')
    
    print(f"图表生成完成，保存到: {plot_dir}")

# 主函数
def main():
    print("开始拓扑分析与模拟...")
    
    # 加载数据
    nodes_df, edges_df = load_topology_data()
    
    # 创建图
    G = create_graph(nodes_df, edges_df)
    
    # 计算初始拓扑指标
    initial_metrics = calculate_topology_metrics(G)
    print(f"初始拓扑指标: {initial_metrics}")
    
    # homology group 分析
    integrity = homology_analysis(G)
    
    # 执行三阶段拓扑修复
    balanced_G, energy_history, imbalance_history, key_nodes = topological_repair_mechanism(G)
    
    # 计算修复后的指标
    final_metrics = calculate_topology_metrics(balanced_G)
    print(f"修复后拓扑指标: {final_metrics}")
    
    # 生成图表
    generate_plots(energy_history, imbalance_history, initial_metrics, final_metrics, integrity, key_nodes, G, balanced_G)
    
    # 保存修复后的网络
    nx.write_graphml(balanced_G, os.path.join(RESULTS_DIR, 'balanced_network.graphml'))
    
    print("\n拓扑分析与模拟完成！")
    print(f"结果保存到: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
