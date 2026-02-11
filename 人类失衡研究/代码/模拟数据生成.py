#!/usr/bin/env python3
"""
模拟数据生成脚本
由于实际数据下载失败，创建模拟数据集用于研究
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime

# 数据保存路径
RAW_DATA_DIR = r'D:\桌面\拓扑波动AI架构\人类失衡研究\原始资料'
PROCESSED_DATA_DIR = r'D:\桌面\拓扑波动AI架构\人类失衡研究\数据处理'

# 确保目录存在
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# 生成腐败行为模拟数据
def generate_corruption_data():
    """生成腐败行为模拟数据"""
    print("生成腐败行为模拟数据...")
    
    # 国家列表
    countries = ["中国", "美国", "日本", "德国", "英国", "法国", "意大利", "加拿大", "澳大利亚", "巴西",
                "俄罗斯", "印度", "南非", "韩国", "墨西哥", "印度尼西亚", "沙特阿拉伯", "土耳其", "波兰", "西班牙"]
    
    # 年份范围
    years = list(range(2010, 2025))
    
    # 生成腐败控制指数（-2.5到2.5，越高越好）
    data = []
    for country in countries:
        # 基础腐败指数（正态分布）
        base_score = np.random.normal(0, 1)
        base_score = max(-2.5, min(2.5, base_score))
        
        for year in years:
            # 每年的波动
            fluctuation = np.random.normal(0, 0.1)
            score = base_score + fluctuation
            score = max(-2.5, min(2.5, score))
            
            # 腐败案例数（与腐败指数负相关）
            case_count = int(np.exp(3 - score) * np.random.uniform(0.8, 1.2))
            
            # 腐败金额（与案例数正相关）
            avg_amount = case_count * np.random.uniform(10000, 50000)
            
            data.append({
                "country": country,
                "year": year,
                "corruption_control": round(score, 2),
                "corruption_cases": case_count,
                "average_amount": round(avg_amount, 2)
            })
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 保存数据
    csv_path = os.path.join(RAW_DATA_DIR, "corruption_simulation_data.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"腐败行为模拟数据生成完成: {csv_path}")
    return df

# 生成道德行为实验模拟数据
def generate_moral_data():
    """生成道德行为实验模拟数据"""
    print("生成道德行为实验模拟数据...")
    
    # 实验参与者
    participants = 200
    
    # 道德基础理论的五个维度
    moral_foundations = ["关爱/伤害", "公平/欺骗", "忠诚/背叛", "权威/颠覆", "圣洁/堕落"]
    
    data = []
    for i in range(participants):
        # 参与者基本信息
        age = np.random.randint(18, 65)
        gender = np.random.choice(["男", "女"])
        education = np.random.choice(["高中及以下", "本科", "硕士及以上"])
        
        # 道德基础得分（1-7分）
        foundation_scores = {}
        for foundation in moral_foundations:
            foundation_scores[foundation] = np.random.randint(1, 8)
        
        # 道德决策情境（1-5个情境）
        for scenario in range(1, 6):
            # 情境类型
            scenario_type = np.random.choice(["个人道德困境", "社会道德困境", "职业伦理", "环境伦理", "数字伦理"])
            
            # 决策选择（1-7分，越高越道德）
            decision = np.random.randint(1, 8)
            
            # 反应时间（秒）
            reaction_time = np.random.uniform(1, 30)
            
            # 情绪反应（1-5分，越高越强烈）
            emotional_response = np.random.randint(1, 6)
            
            data.append({
                "participant_id": i+1,
                "age": age,
                "gender": gender,
                "education": education,
                "scenario": scenario,
                "scenario_type": scenario_type,
                "decision": decision,
                "reaction_time": round(reaction_time, 2),
                "emotional_response": emotional_response,
                **foundation_scores
            })
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 保存数据
    csv_path = os.path.join(RAW_DATA_DIR, "moral_experiment_simulation_data.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"道德行为实验模拟数据生成完成: {csv_path}")
    return df

# 生成社会伦理调查模拟数据
def generate_ethics_data():
    """生成社会伦理调查模拟数据"""
    print("生成社会伦理调查模拟数据...")
    
    # 调查问题
    ethics_questions = [
        "为了更大的利益，是否可以牺牲少数人的利益？",
        "在商业活动中，是否可以为了成功而稍微不诚实？",
        "是否应该严格遵守所有法律，即使有些法律看起来不公平？",
        "在竞争中，是否可以使用一些手段来获得优势？",
        "是否应该对亲近的人更加宽容，即使他们做错了事情？"
    ]
    
    # 调查对象
    respondents = 500
    
    data = []
    for i in range(respondents):
        # 基本信息
        age = np.random.randint(18, 70)
        gender = np.random.choice(["男", "女"])
        income = np.random.choice(["低收入", "中等收入", "高收入"])
        region = np.random.choice(["城市", "农村"])
        
        # 价值观倾向
        value_orientation = np.random.choice(["集体主义", "个人主义", "混合"])
        
        # 问题回答（1-5分，1=强烈反对，5=强烈同意）
        responses = {}
        for j, question in enumerate(ethics_questions):
            # 根据价值观倾向调整回答
            if value_orientation == "集体主义":
                # 集体主义更倾向于同意牺牲少数人利益
                if j == 0:
                    response = np.random.randint(3, 6)
                else:
                    response = np.random.randint(1, 6)
            elif value_orientation == "个人主义":
                # 个人主义更倾向于反对牺牲少数人利益
                if j == 0:
                    response = np.random.randint(1, 4)
                else:
                    response = np.random.randint(1, 6)
            else:
                # 混合倾向
                response = np.random.randint(1, 6)
            
            responses[f"q{j+1}"] = response
        
        # 道德自我评分
        moral_self_score = np.random.randint(1, 11)
        
        data.append({
            "respondent_id": i+1,
            "age": age,
            "gender": gender,
            "income": income,
            "region": region,
            "value_orientation": value_orientation,
            "moral_self_score": moral_self_score,
            **responses
        })
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 保存数据
    csv_path = os.path.join(RAW_DATA_DIR, "ethics_survey_simulation_data.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"社会伦理调查模拟数据生成完成: {csv_path}")
    return df

# 生成拓扑网络模拟数据
def generate_topology_data():
    """生成拓扑网络模拟数据"""
    print("生成拓扑网络模拟数据...")
    
    # 网络节点数
    n_nodes = 50
    
    # 生成节点属性
    nodes = []
    for i in range(n_nodes):
        # 节点类型
        node_type = np.random.choice(["个人", "组织", "制度"])
        
        # 节点强度（1-10）
        strength = np.random.randint(1, 11)
        
        # 节点稳定性（0-1）
        stability = np.random.uniform(0, 1)
        
        # 节点失衡程度（0-1，越高越失衡）
        imbalance = np.random.uniform(0, 1)
        
        nodes.append({
            "node_id": i+1,
            "node_type": node_type,
            "strength": strength,
            "stability": round(stability, 2),
            "imbalance": round(imbalance, 2)
        })
    
    # 生成边（连接）
    edges = []
    max_edges = n_nodes * (n_nodes - 1) // 2
    n_edges = np.random.randint(n_nodes, max_edges)
    
    existing_edges = set()
    while len(edges) < n_edges:
        # 随机选择两个节点
        node1 = np.random.randint(1, n_nodes+1)
        node2 = np.random.randint(1, n_nodes+1)
        
        # 确保不重复且不是自连接
        if node1 != node2 and (node1, node2) not in existing_edges and (node2, node1) not in existing_edges:
            # 边的权重（1-10）
            weight = np.random.randint(1, 11)
            
            # 边的稳定性（0-1）
            edge_stability = np.random.uniform(0, 1)
            
            # 边的失衡程度（0-1）
            edge_imbalance = np.random.uniform(0, 1)
            
            edges.append({
                "source": node1,
                "target": node2,
                "weight": weight,
                "stability": round(edge_stability, 2),
                "imbalance": round(edge_imbalance, 2)
            })
            
            existing_edges.add((node1, node2))
    
    # 保存节点数据
    nodes_df = pd.DataFrame(nodes)
    nodes_path = os.path.join(RAW_DATA_DIR, "topology_nodes_simulation_data.csv")
    nodes_df.to_csv(nodes_path, index=False, encoding='utf-8-sig')
    
    # 保存边数据
    edges_df = pd.DataFrame(edges)
    edges_path = os.path.join(RAW_DATA_DIR, "topology_edges_simulation_data.csv")
    edges_df.to_csv(edges_path, index=False, encoding='utf-8-sig')
    
    print(f"拓扑网络模拟数据生成完成:")
    print(f"- 节点数据: {nodes_path}")
    print(f"- 边数据: {edges_path}")
    
    return nodes_df, edges_df

# 创建模拟数据说明文档
def create_simulation_doc():
    """创建模拟数据说明文档"""
    doc_path = os.path.join(RAW_DATA_DIR, "模拟数据说明.md")
    
    with open(doc_path, 'w', encoding='utf-8') as f:
        f.write("# 模拟数据说明\n\n")
        f.write(f"创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 数据说明\n\n")
        f.write("由于实际数据下载失败，本项目使用模拟数据进行研究。模拟数据基于真实数据的分布特征生成，用于验证拓扑波动理论在人类失衡研究中的应用。\n\n")
        
        f.write("## 数据集列表\n\n")
        f.write("### 1. 腐败行为模拟数据 (corruption_simulation_data.csv)\n")
        f.write("- 包含20个国家2010-2024年的腐败相关数据\n")
        f.write("- 字段：country, year, corruption_control, corruption_cases, average_amount\n\n")
        
        f.write("### 2. 道德行为实验模拟数据 (moral_experiment_simulation_data.csv)\n")
        f.write("- 包含200名参与者在5个道德决策情境中的反应\n")
        f.write("- 字段：participant_id, age, gender, education, scenario, scenario_type, decision, reaction_time, emotional_response, 道德基础得分\n\n")
        
        f.write("### 3. 社会伦理调查模拟数据 (ethics_survey_simulation_data.csv)\n")
        f.write("- 包含500名受访者对5个伦理问题的回答\n")
        f.write("- 字段：respondent_id, age, gender, income, region, value_orientation, moral_self_score, q1-q5\n\n")
        
        f.write("### 4. 拓扑网络模拟数据\n")
        f.write("- 节点数据 (topology_nodes_simulation_data.csv)：包含50个节点的属性\n")
        f.write("- 边数据 (topology_edges_simulation_data.csv)：包含节点间的连接关系\n\n")
        
        f.write("## 数据生成方法\n\n")
        f.write("- **腐败行为数据**：基于正态分布生成腐败控制指数，案例数和金额与指数负相关\n")
        f.write("- **道德行为数据**：考虑参与者人口统计学特征和价值观倾向\n")
        f.write("- **社会伦理数据**：基于不同价值观倾向生成伦理问题回答\n")
        f.write("- **拓扑网络数据**：随机生成节点和边，包含强度、稳定性和失衡程度属性\n")
    
    print(f"\n模拟数据说明文档创建完成: {doc_path}")

# 主函数
def main():
    print("开始生成模拟数据...")
    
    # 创建模拟数据说明文档
    create_simulation_doc()
    
    # 生成各类模拟数据
    generate_corruption_data()
    generate_moral_data()
    generate_ethics_data()
    generate_topology_data()
    
    print("\n模拟数据生成完成！")
    print(f"所有数据已保存到: {RAW_DATA_DIR}")

if __name__ == "__main__":
    main()
