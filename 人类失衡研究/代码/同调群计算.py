#!/usr/bin/env python3
"""
同调群计算脚本
实现贝蒂数β0/β1和失衡指数II的计算逻辑
"""

import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import svds

# 计算贝蒂数
def calculate_betti_numbers(G):
    """
    计算网络的贝蒂数
    
    参数:
        G: NetworkX图对象
    
    返回:
        dict: 包含h0和h1的贝蒂数
    """
    print("计算贝蒂数...")
    
    # 计算0阶贝蒂数（连通分量数）
    h0 = nx.number_connected_components(G)
    
    # 计算1阶贝蒂数（环的数量）
    # 对于平面图，使用欧拉公式的变形：h1 = E - V + C
    V = G.number_of_nodes()
    E = G.number_of_edges()
    C = h0
    h1 = max(0, E - V + C)
    
    betti_numbers = {
        "h0": h0,
        "h1": h1,
        "h2": 0  # 对于平面图，h2=0
    }
    
    print(f"贝蒂数计算完成: h0={h0}, h1={h1}, h2=0")
    return betti_numbers

# 计算失衡指数II
def calculate_imbalance_index(G):
    """
    计算网络的失衡指数II
    
    参数:
        G: NetworkX图对象
    
    返回:
        float: 失衡指数II
    """
    print("计算失衡指数II...")
    
    # 获取所有节点的失衡值
    node_imbalance = [data.get('imbalance', 0.5) for _, data in G.nodes(data=True)]
    
    # 获取所有边的失衡值
    edge_imbalance = []
    for u, v, data in G.edges(data=True):
        edge_imbalance.append(data.get('imbalance', 0.5))
    
    # 计算节点失衡平均值和标准差
    if node_imbalance:
        avg_node_imbalance = np.mean(node_imbalance)
        std_node_imbalance = np.std(node_imbalance)
    else:
        avg_node_imbalance = 0.5
        std_node_imbalance = 0
    
    # 计算边失衡平均值和标准差
    if edge_imbalance:
        avg_edge_imbalance = np.mean(edge_imbalance)
        std_edge_imbalance = np.std(edge_imbalance)
    else:
        avg_edge_imbalance = 0.5
        std_edge_imbalance = 0
    
    # 计算拓扑结构失衡（基于贝蒂数）
    betti = calculate_betti_numbers(G)
    h0, h1 = betti['h0'], betti['h1']
    
    # 拓扑结构失衡指标
    # 连通分量过多或环过多都表示结构失衡
    structural_imbalance = min(1.0, (h0 - 1) / 10 + h1 / (G.number_of_nodes() * 2))
    
    # 计算综合失衡指数II
    # 权重分配：节点失衡40%，边失衡30%，结构失衡30%
    II = (0.4 * (avg_node_imbalance + std_node_imbalance) +
          0.3 * (avg_edge_imbalance + std_edge_imbalance) +
          0.3 * structural_imbalance)
    
    # 归一化到[0, 1]范围
    II = max(0, min(1, II))
    
    print(f"失衡指数II计算完成: {II:.4f}")
    return II

# 基于矩阵的同调群计算（更准确的方法）
def calculate_homology_matrix(G):
    """
    基于边界矩阵计算同调群
    
    参数:
        G: NetworkX图对象
    
    返回:
        dict: 包含贝蒂数和同调群信息
    """
    print("基于矩阵计算同调群...")
    
    # 获取节点和边
    nodes = list(G.nodes())
    edges = list(G.edges())
    
    V = len(nodes)
    E = len(edges)
    
    if E == 0:
        return {"h0": V, "h1": 0, "h2": 0}
    
    # 构建边界矩阵 ∂1: C1 → C0
    # 每列对应一条边，每行对应一个节点
    boundary_matrix = sp.lil_matrix((V, E), dtype=int)
    
    for i, (u, v) in enumerate(edges):
        u_idx = nodes.index(u)
        v_idx = nodes.index(v)
        boundary_matrix[u_idx, i] = 1
        boundary_matrix[v_idx, i] = -1
    
    # 转换为CSR格式以提高计算效率
    boundary_matrix = boundary_matrix.tocsr()
    
    # 计算边界矩阵的秩（即同态的像的维度）
    # 使用SVD计算秩
    try:
        # 对于大型矩阵，使用部分SVD
        if V * E > 1000000:
            _, s, _ = svds(boundary_matrix.asfptype(), k=min(V, E, 100))
        else:
            _, s, _ = svds(boundary_matrix.asfptype())
        
        # 计算非零奇异值的数量（秩）
        rank_d1 = np.sum(s > 1e-10)
    except:
        # 如果SVD失败，使用简化方法
        rank_d1 = min(V-1, E)
    
    # 计算0阶同调群的秩: H0 = Z^V / im(∂1)
    h0 = V - rank_d1
    
    # 计算1阶同调群的秩: H1 = ker(∂1) / im(∂2)
    # 对于1维复形，∂2=0，所以 H1 = ker(∂1)
    # ker(∂1)的维度 = E - rank(∂1)
    h1 = max(0, E - rank_d1)
    
    homology_info = {
        "h0": h0,
        "h1": h1,
        "h2": 0,
        "rank_d1": rank_d1,
        "explanation": "H0 = V - rank(∂1), H1 = E - rank(∂1)"
    }
    
    print(f"矩阵同调群计算完成: h0={h0}, h1={h1}, h2=0")
    return homology_info

# 计算拓扑能量E
def calculate_topology_energy(G):
    """
    计算拓扑能量E
    公式: E = (alpha * 结构完整性 + beta * 本源规则一致性) / (alpha + beta)
    
    参数:
        G: NetworkX图对象
    
    返回:
        float: 拓扑能量E
    """
    print("计算拓扑能量E...")
    
    # 参数设置
    alpha = 0.6  # 结构完整性权重
    beta = 0.4  # 本源规则一致性权重
    
    # 计算结构完整性
    # 基于贝蒂数和连通性
    betti = calculate_betti_numbers(G)
    h0, h1 = betti['h0'], betti['h1']
    
    # 理想情况下，h0=1（一个连通分量），h1适中
    V = G.number_of_nodes()
    ideal_h1 = V // 2  # 理想环数
    
    # 结构完整性得分（0-1）
    connectivity_score = 1.0 if h0 == 1 else max(0, 1 - (h0 - 1) / V)
    cycle_score = max(0, 1 - abs(h1 - ideal_h1) / max(V, 1))
    structure_integrity = (connectivity_score + cycle_score) / 2
    
    # 计算本源规则一致性
    # 基于节点和边的失衡值
    node_imbalance = [data.get('imbalance', 0.5) for _, data in G.nodes(data=True)]
    edge_imbalance = [data.get('imbalance', 0.5) for _, _, data in G.edges(data=True)]
    
    if node_imbalance:
        avg_node_balance = 1 - np.mean(node_imbalance)
    else:
        avg_node_balance = 0.5
    
    if edge_imbalance:
        avg_edge_balance = 1 - np.mean(edge_imbalance)
    else:
        avg_edge_balance = 0.5
    
    # 本源规则一致性得分（0-1）
    rule_consistency = (avg_node_balance + avg_edge_balance) / 2
    
    # 计算拓扑能量E
    E = (alpha * structure_integrity + beta * rule_consistency) / (alpha + beta)
    
    # 能量值范围：0（完全失衡）到1（完全平衡）
    E = max(0, min(1, E))
    
    print(f"拓扑能量E计算完成: {E:.4f}")
    return E

# 计算共振一致性
def calculate_resonance_consistency(G):
    """
    计算共振一致性
    个体与社会拓扑系统波动频率的相关系数
    
    参数:
        G: NetworkX图对象
    
    返回:
        float: 共振一致性值
    """
    print("计算共振一致性...")
    
    # 模拟个体和系统的波动频率
    # 基于节点强度和失衡值
    node_strengths = []
    node_imbalances = []
    
    for _, data in G.nodes(data=True):
        strength = data.get('strength', 1.0)
        imbalance = data.get('imbalance', 0.5)
        node_strengths.append(strength)
        node_imbalances.append(imbalance)
    
    if len(node_strengths) < 2:
        return 0.5
    
    # 计算个体波动频率
    # 强度越高，波动频率越高；失衡越高，波动频率越不稳定
    individual_frequencies = []
    for s, i in zip(node_strengths, node_imbalances):
        # 基础频率基于强度
        base_freq = s
        # 失衡导致频率波动
        freq_variation = i * 0.5
        # 实际频率
        freq = base_freq * (1 + np.random.normal(0, freq_variation))
        individual_frequencies.append(max(0.1, freq))  # 确保频率为正
    
    # 计算系统波动频率
    # 系统频率是个体频率的加权平均
    weights = node_strengths / np.sum(node_strengths)
    system_frequency = np.average(individual_frequencies, weights=weights)
    
    # 计算共振一致性（相关系数）
    # 个体频率与系统频率的偏差越小，共振一致性越高
    deviations = [abs(f - system_frequency) / system_frequency for f in individual_frequencies]
    avg_deviation = np.mean(deviations)
    
    # 共振一致性得分（0-1）
    resonance_consistency = max(0, 1 - avg_deviation)
    
    print(f"共振一致性计算完成: {resonance_consistency:.4f}")
    return resonance_consistency

# 测试函数
def test_homology_calculations():
    """
    测试同调群计算功能
    """
    print("\n=== 测试同调群计算功能 ===")
    
    # 创建一个测试图
    G = nx.Graph()
    
    # 添加节点
    for i in range(10):
        G.add_node(i, 
                   strength=np.random.uniform(0.5, 1.5),
                   imbalance=np.random.uniform(0.1, 0.9))
    
    # 添加边
    for i in range(10):
        for j in range(i+1, 10):
            if np.random.rand() < 0.3:
                G.add_edge(i, j, 
                           weight=np.random.randint(1, 10),
                           imbalance=np.random.uniform(0.1, 0.9))
    
    # 测试贝蒂数计算
    betti = calculate_betti_numbers(G)
    print(f"测试结果 - 贝蒂数: {betti}")
    
    # 测试失衡指数计算
    ii = calculate_imbalance_index(G)
    print(f"测试结果 - 失衡指数II: {ii:.4f}")
    
    # 测试矩阵同调群计算
    homology = calculate_homology_matrix(G)
    print(f"测试结果 - 矩阵同调群: {homology}")
    
    # 测试拓扑能量计算
    energy = calculate_topology_energy(G)
    print(f"测试结果 - 拓扑能量E: {energy:.4f}")
    
    # 测试共振一致性计算
    resonance = calculate_resonance_consistency(G)
    print(f"测试结果 - 共振一致性: {resonance:.4f}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_homology_calculations()
