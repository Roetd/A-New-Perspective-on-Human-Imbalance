#!/usr/bin/env python3
"""
数据收集脚本
从网上下载相关的数据集，包括腐败行为数据集、道德行为实验数据和社会伦理调查数据
"""

import os
import requests
import zipfile
import pandas as pd
from datetime import datetime

# 数据保存路径
RAW_DATA_DIR = r'D:\桌面\拓扑波动AI架构\人类失衡研究\原始资料'

# 确保目录存在
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# 下载文件的函数
def download_file(url, save_path):
    """下载文件并保存到指定路径"""
    print(f"正在下载: {url}")
    print(f"保存到: {save_path}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print(f"下载完成: {save_path}")
        return True
    except Exception as e:
        print(f"下载失败: {e}")
        return False

# 解压文件的函数
def unzip_file(zip_path, extract_dir):
    """解压zip文件"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"解压完成: {zip_path}")
        return True
    except Exception as e:
        print(f"解压失败: {e}")
        return False

# 下载腐败行为数据集
def download_corruption_data():
    """下载腐败行为相关数据集"""
    print("\n=== 下载腐败行为数据集 ===")
    
    # 世界银行腐败控制指标
    world_bank_url = "https://databank.worldbank.org/data/download/CCK.csv"
    world_bank_path = os.path.join(RAW_DATA_DIR, "world_bank_corruption.csv")
    download_file(world_bank_url, world_bank_path)
    
    # Transparency International 腐败感知指数
    ti_url = "https://www.transparency.org/en/cpi/2024/data/download"
    ti_path = os.path.join(RAW_DATA_DIR, "transparency_international_cpi.pdf")
    download_file(ti_url, ti_path)
    
    # 腐败案例数据集
    corruption_cases_url = "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/28075"
    corruption_cases_path = os.path.join(RAW_DATA_DIR, "corruption_cases.zip")
    if download_file(corruption_cases_url, corruption_cases_path):
        extract_dir = os.path.join(RAW_DATA_DIR, "corruption_cases")
        os.makedirs(extract_dir, exist_ok=True)
        unzip_file(corruption_cases_path, extract_dir)

# 下载道德行为实验数据
def download_moral_data():
    """下载道德行为实验数据"""
    print("\n=== 下载道德行为实验数据 ===")
    
    # Moral Foundations Dictionary 数据
    mfd_url = "https://moralfoundations.org/wp-content/uploads/files/downloads/moral_foundations_dictionary.dic"
    mfd_path = os.path.join(RAW_DATA_DIR, "moral_foundations_dictionary.dic")
    download_file(mfd_url, mfd_path)
    
    # 道德决策实验数据
    moral_decision_url = "https://www.openicpsr.org/openicpsr/project/106300/version/V1/view"
    moral_decision_path = os.path.join(RAW_DATA_DIR, "moral_decision_experiment.pdf")
    download_file(moral_decision_url, moral_decision_path)
    
    # 道德判断数据集
    moral_judgment_url = "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/2Q2GJ5"
    moral_judgment_path = os.path.join(RAW_DATA_DIR, "moral_judgment.zip")
    if download_file(moral_judgment_url, moral_judgment_path):
        extract_dir = os.path.join(RAW_DATA_DIR, "moral_judgment")
        os.makedirs(extract_dir, exist_ok=True)
        unzip_file(moral_judgment_path, extract_dir)

# 下载社会伦理调查数据
def download_ethics_data():
    """下载社会伦理调查数据"""
    print("\n=== 下载社会伦理调查数据 ===")
    
    # World Values Survey 数据
    wvs_url = "https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp"
    wvs_path = os.path.join(RAW_DATA_DIR, "world_values_survey.pdf")
    download_file(wvs_url, wvs_path)
    
    # 社会资本调查数据
    social_capital_url = "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/P8JGI7"
    social_capital_path = os.path.join(RAW_DATA_DIR, "social_capital.zip")
    if download_file(social_capital_url, social_capital_path):
        extract_dir = os.path.join(RAW_DATA_DIR, "social_capital")
        os.makedirs(extract_dir, exist_ok=True)
        unzip_file(social_capital_path, extract_dir)
    
    # 伦理态度调查数据
    ethics_survey_url = "https://gssdataexplorer.norc.org/dataverse/gss"
    ethics_survey_path = os.path.join(RAW_DATA_DIR, "ethics_survey.pdf")
    download_file(ethics_survey_url, ethics_survey_path)

# 创建数据来源文档
def create_data_sources_doc():
    """创建数据来源文档"""
    data_sources = {
        "腐败行为数据集": [
            {
                "名称": "World Bank Corruption Control Index",
                "下载地址": "https://databank.worldbank.org/data/download/CCK.csv",
                "描述": "世界银行发布的腐败控制指标，衡量各国腐败程度",
                "格式": "CSV"
            },
            {
                "名称": "Transparency International Corruption Perceptions Index",
                "下载地址": "https://www.transparency.org/en/cpi/2024/data/download",
                "描述": "透明国际发布的腐败感知指数，基于专家评估",
                "格式": "PDF"
            },
            {
                "名称": "Corruption Cases Dataset",
                "下载地址": "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/28075",
                "描述": "哈佛数据verse上的腐败案例数据集",
                "格式": "ZIP"
            }
        ],
        "道德行为实验数据": [
            {
                "名称": "Moral Foundations Dictionary",
                "下载地址": "https://moralfoundations.org/wp-content/uploads/files/downloads/moral_foundations_dictionary.dic",
                "描述": "道德基础理论的词典数据",
                "格式": "DIC"
            },
            {
                "名称": "Moral Decision Experiment",
                "下载地址": "https://www.openicpsr.org/openicpsr/project/106300/version/V1/view",
                "描述": "道德决策实验数据",
                "格式": "PDF"
            },
            {
                "名称": "Moral Judgment Dataset",
                "下载地址": "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/2Q2GJ5",
                "描述": "道德判断数据集",
                "格式": "ZIP"
            }
        ],
        "社会伦理调查数据": [
            {
                "名称": "World Values Survey",
                "下载地址": "https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp",
                "描述": "世界价值观调查数据，包含伦理态度",
                "格式": "PDF"
            },
            {
                "名称": "Social Capital Dataset",
                "下载地址": "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/P8JGI7",
                "描述": "社会资本调查数据",
                "格式": "ZIP"
            },
            {
                "名称": "General Social Survey Ethics Data",
                "下载地址": "https://gssdataexplorer.norc.org/dataverse/gss",
                "描述": "美国综合社会调查中的伦理相关数据",
                "格式": "PDF"
            }
        ]
    }
    
    # 创建数据来源文档
    doc_path = os.path.join(RAW_DATA_DIR, "数据来源说明.md")
    with open(doc_path, 'w', encoding='utf-8') as f:
        f.write("# 数据来源说明\n\n")
        f.write(f"创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for category, sources in data_sources.items():
            f.write(f"## {category}\n\n")
            for source in sources:
                f.write(f"### {source['名称']}\n")
                f.write(f"- **下载地址**: {source['下载地址']}\n")
                f.write(f"- **描述**: {source['描述']}\n")
                f.write(f"- **格式**: {source['格式']}\n\n")
    
    print(f"\n数据来源文档创建完成: {doc_path}")

# 主函数
def main():
    print("开始数据收集...")
    
    # 创建数据来源文档
    create_data_sources_doc()
    
    # 下载各类数据
    download_corruption_data()
    download_moral_data()
    download_ethics_data()
    
    print("\n数据收集完成！")
    print(f"所有数据已保存到: {RAW_DATA_DIR}")

if __name__ == "__main__":
    main()
