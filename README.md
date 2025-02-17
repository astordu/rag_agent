![image](https://github.com/user-attachments/assets/4c1ec2ad-aebb-491c-97c2-3faf6f8c64c3)

#  RAG_Agent 

这是一个基于西游记白话文版本构建的 RAG_Agent (检索增强生成) 智能问答系统。该项目使用西游记白话文文本作为知识库，通过 RAG_AGENT 技术实现对西游记相关问题的智能回答。

对比了传统RAG与RAG_Agent的不同.

## 使用说明

导入openrouter的key到环境变量:
  
  ```
    export OPENROUTER_API_KEY=xxx
  ```

运行rag_naive.py和rag_agent.py直接可以看到对比效果了.

## 数据集说明

本项目使用改编版《西游记》白话文作为基础数据集. 

## 功能特性

项目主要分为三个部分

- 文本预处理 (etl.py)
  - 文本清洗
  - 格式标准化
  - 批量处理文件

- 文本分块 (split_chunk.py) 
  - 按语义分块
  - 生成检索单元

- RAG 问答 (rag_naive.py)
  - 基于传统RAG的构建方式

- RAG 问答 (rag_agent.py)
  - 基于检索的问答系统
  - 智能理解西游记相关问题
  - 准确检索相关内容
  - 生成连贯答案

## 项目结构 

    ├── 西游记白话文/ # 原始文本数据
    ├── output/ # 处理后的文本
    ├── etl.py # 文本预处理脚本
    ├── split_chunk.py # 文本分块脚本
    └── rag_agent.py # RAG Agent
    └── rag_naive.py # 传统RAG

## 测试问题

孙悟空有两个师傅,他们分别是谁?
