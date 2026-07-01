# 🚀 Decision Evaluation System for New Retail Industry Based on BERT+PET

(BERT-PET based New Retail Decision Evaluation System)

## 📖 Project Introduction
With the rapid development of technology and the ubiquity of smart devices, AI technology has been widely applied in the new retail industry. This project aims to build a deep learning-based e-commerce review text classification system.
In intelligent recommendation systems, user text reviews contain rich semantic information (such as preferences and product features). Compared to explicit ratings, text information can effectively compensate for rating sparsity and enhance the explainability of recommendations. This project utilizes the BERT + PET (Pattern-Exploiting Training) method to accurately classify e-commerce platform user reviews, helping platforms respond to user needs quickly, improve products and services, and lay the foundation for personalized recommendations.
## 💡 Core Technology: BERT + PET
This project adopts the PET (Pattern-Exploiting Training) paradigm. Its core idea is to transform traditional classification tasks into Cloze tasks, which pre-trained models (MLM) excel at.
Template Construction: Artificially define templates based on prior knowledge to wrap the input text. For example: This is a {MASK} review: {textA}.
Verbalizer: Establish a mapping relationship between real labels and predicted words (Label Words). For example, mapping the label "Sports" to a word like "Football" which is easier for the model to predict.
Fine-tuning: Fine-tune the MLM task parameters to make the model perform better on specific classification tasks, which is especially suitable for few-shot scenarios or situations requiring high semantic understanding.
### 🛠️ Environment Preparation
This project is implemented based on PyTorch and HuggingFace Transformers. Please ensure the following dependencies are installed before running:
```
pip install torch
pip install transformers==4.22.1
pip install datasets==2.4.0
pip install evaluate==0.2.2
pip install matplotlib==3.6.0
pip install rich==12.5.1
pip install scikit-learn==1.1.2
pip install requests==2.28.1
```
## 📂 Project Architecture and Directory Description
```
/
├── bert-base-chinese/      # Pre-trained model files (BERT-base-chinese)
├── checkpoints/            # Model storage location (model_400, model_best, etc.)
├── data/                   # Data storage location
│   ├── train.txt           # Training set (Format: label \t text)
│   ├── dev.txt             # Validation set
│   ├── prompt.txt          # Prompt template (containing placeholders like {MASK}, {textA})
│   └── verbalizer.txt      # Label mapping file (Real label -> Predicted word)
├── data_handle/            # Data preprocessing module
│   ├── template.py         # Template construction and Text2ID conversion
│   ├── data_preprocess.py  # Convert sample data to model Input IDs
│   └── data_loader.py      # Define DataLoader, encapsulate batch reading
├── utils/                  # Utility package
│   ├── verbalizer.py       # Label mapping logic (including Longest Common Substring fuzzy matching)
│   ├── metric_utils.py     # Evaluation metric calculation (Acc, Precision, Recall, F1)
│   └── common_utils.py     # Loss function (MLM Loss) and Logits to ID tools
├── pet_config.py           # Project configuration file (paths, hyperparameters, etc.)
├── train.py                # Model training and validation entry point
├── inference.py            # Model inference entry point
└── nohup.out               # Training log file
```
## 🔍 Key Module Analysis
### Data Processing (data_handle)
Template Construction: template.py is responsible for parsing custom parameters in prompt.txt (such as {textA}, {MASK}) and concatenating the processed text into a tensor format acceptable by the model.
Data Loader: data_loader.py implements an efficient data iterator, automatically completing the conversion from raw text to Token IDs and loading data by Batch.
### Label Mapping Strategy (utils/verbalizer.py)
During the inference phase, the predicted words output by the model may not exactly match the defined labels. This project implements a fallback strategy based on Longest Common Substring fuzzy matching:
Logic: 
  Calculate the similarity between the model's predicted words and the sub-labels in the label dictionary.
Optimization Suggestion: 
  Although string matching is intuitive, it may present performance bottlenecks with large data volumes. For industrial deployment, it is recommended to switch to direct matching based on Token IDs, using integer comparison instead of dynamic programming algorithms to significantly improve inference speed and align with the model's vocabulary space.
### Model Evaluation (utils/metric_utils.py)
Implements a standard classification evaluator ClassEvaluator:
Streaming Evaluation: 
  Supports accumulating prediction results in batches, eliminating the need to load all test data at once.
Multi-dimensional Metrics: 
  alculates not only global Accuracy and Weighted F1 but also outputs Precision, Recall, and F1 for each category, facilitating the analysis of model performance shortcomings in specific categories (e.g., "Clothing", "Fruit").
## 🚀 Quick Start
### Dataset Preparation
Prepare data in the data/ directory.
train.txt / dev.txt: Each line is separated by \t, with the first part being the label and the second part being the review text.
Fruit   Crispy, sweet taste is okay, but maybe it's been a while, not very juicy.
Tablet  Huawei devices are definitely good, but I encountered the worst service on JD.com for the first time...
prompt.txt: Define the template, e.g., This is a {MASK} review: {textA}.
### Project Configuration
Modify pet_config.py to set model paths, data paths, Batch Size, Learning Rate, and other hyperparameters.
Train Model
Run the training script:
```
python train.py
```
During training, Loss changes will be output, and Precision, Recall, and F1 will be evaluated on the validation set.
### Model Inference
Use the trained model for prediction:
```
python inference.py
```
### 📊 Expected Results
With the introduction of the PET method, the model can understand the review context more accurately.
Example Input: "China upset South Korea 2-1 in a shock win"
Model Prediction: The [MASK] position predicts a word related to "Sports", and it is finally classified as Sports news.
### 📝 Future Optimization Directions
Data Augmentation: Increase specific sample data for training categories with low F1 scores (such as certain long-tail product reviews).
Hard Template Search: Try different Prompt templates to find the expression that works best for the current dataset.
Engineering Deployment: Optimize the string matching part of inference into Tensor operations to improve online service response speed.
### 🔗 Reference
Pre-trained model (bert-base-chinese), download address: The file is shared via Baidu Netdisk. Link: ```
```
https://pan.baidu.com/s/1UnHXBJUpib1m7-LIE4oNjA?pwd=cywd, Access Code: cywd
```
#### For a detailed step-by-step guide on this project, please refer to the original blog post:
```
https://blog.csdn.net/zhoupenghui168/article/details/162371694
```

基于 BERT+PET 的新零售行业决策评价系统
📖 项目简介
随着科技的迅速发展和智能设备的普及，AI 技术在新零售行业中得到了广泛应用。本项目旨在构建一个基于深度学习的电商评论文本分类系统。
在智能推荐系统中，用户的文本评论蕴含着丰富的语义信息（如偏好、商品特征等）。相比于显式评分，文本信息能有效弥补评分稀疏性问题，并提升推荐的可解释性。本项目利用 BERT + PET (Pattern-Exploiting Training) 方法，对电商平台用户评论进行精准分类，帮助平台快速回应用户需求，改进产品服务，并为个性化推荐奠定基础。
💡 核心技术：BERT + PET
本项目采用 PET (Pattern-Exploiting Training) 范式，其核心思想是将传统的分类任务转化为预训练模型（MLM）擅长的完形填空（Cloze）任务。
模版构建 (Template)：根据先验知识人工定义模版，将输入文本包裹起来。例如：这是一条 {MASK} 评论：{textA}。
标签映射 (Verbalizer)：建立真实标签与预测词（Label Words）之间的映射关系。例如将标签“体育”映射为更容易被模型预测的词汇“足球”。
微调 (Fine-tuning)：通过微调 MLM 任务参数，使模型在特定分类任务上表现更佳，特别适合小样本或语义理解要求高的场景。
🛠️ 环境准备
本项目基于 PyTorch 和 HuggingFace Transformers 实现。运行前请确保安装以下依赖包：
pip install torch
pip install transformers==4.22.1
pip install datasets==2.4.0
pip install evaluate==0.2.2
pip install matplotlib==3.6.0
pip install rich==12.5.1
pip install scikit-learn==1.1.2
pip install requests==2.28.1

📂 项目架构与目录说明
/
├── bert-base-chinese/      # 预训练模型文件 (BERT-base-chinese)
├── checkpoints/            # 模型存放位置 (model_400, model_best等)
├── data/                   # 数据存放位置
│   ├── train.txt           # 训练集 (格式: label \t text)
│   ├── dev.txt             # 验证集
│   ├── prompt.txt          # 提示模版 (包含 {MASK}, {textA} 等占位符)
│   └── verbalizer.txt      # 标签映射文件 (真实标签 -> 预测词)
├── data_handle/            # 数据预处理模块
│   ├── template.py         # 模版构建与 Text2ID 转换
│   ├── data_preprocess.py  # 样本数据转换为模型 Input IDs
│   └── data_loader.py      # 定义 DataLoader，封装批次读取
├── utils/                  # 工具包
│   ├── verbalizer.py       # 标签映射逻辑 (含最长公共子串模糊匹配)
│   ├── metric_utils.py     # 评估指标计算 (Acc, Precision, Recall, F1)
│   └── common_utils.py     # 损失函数(MLM Loss)及 Logits 转 ID 工具
├── pet_config.py           # 项目配置文件 (路径、超参数等)
├── train.py                # 模型训练与验证入口
├── inference.py            # 模型预测入口
└── nohup.out               # 训练日志文件
🔍 关键模块解析
1. 数据处理 (data_handle)
Template 构建：template.py 负责解析 prompt.txt 中的自定义参数（如 {textA}, {MASK}），并将处理后的文本拼接成模型可接受的张量格式。
Data Loader：data_loader.py 实现了高效的数据迭代器，自动完成从原始文本到 Token ID 的转换，并按 Batch 加载数据。
2. 标签映射策略 (utils/verbalizer.py)
在推理阶段，模型输出的预测词可能与定义的标签不完全一致。本项目实现了一个基于最长公共子串 (Longest Common Substring) 的模糊匹配策略作为兜底：
逻辑：计算模型预测词与标签字典中子标签的相似度。
优化建议：虽然字符串匹配直观，但在大数据量下可能存在性能瓶颈。工业级落地建议改为基于 Token ID 的直接匹配，利用整数比较代替动态规划算法，大幅提升推理速度并对齐模型词表空间。
3. 模型评估 (utils/metric_utils.py)
实现了标准的分类评估器 ClassEvaluator：
流式评估：支持分批累积预测结果，无需一次性加载所有测试数据。
多维度指标：不仅计算全局 Accuracy 和 Weighted F1，还能输出每个类别的 Precision、Recall 和 F1，便于分析模型在特定类别（如“衣服”、“水果”）上的表现短板。
🚀 快速开始
1. 数据集准备
在 data/ 目录下准备数据。
train.txt / dev.txt: 每一行用 \t 分隔，前半部分为标签，后半部分为评论文本。
水果    脆脆的，甜味可以，可能时间有点长了，水分不是很足。
平板    华为机器肯定不错，但第一次碰上京东最糟糕的服务...
prompt.txt: 定义模版，例如 这是一条 {MASK} 评论：{textA}。
2. 配置项目
修改 pet_config.py，设置模型路径、数据路径、Batch Size、Learning Rate 等超参数。
3. 训练模型
运行训练脚本：
python train.py
训练过程中会输出 Loss 变化，并在验证集上评估 Precision, Recall, F1。
4. 模型推理
使用训练好的模型进行预测：
python inference.py
📊 预期效果
通过 PET 方法的引入，模型能够更准确地理解评论语境。
示例输入：“中国爆冷2-1战胜韩国”
模型预测：[MASK] 处预测为“体育”相关的词汇，最终归类为 体育 新闻。
📝 后续优化方向
数据增强：针对 F1 值较低的类别（如某些长尾商品评论），增加特定样本数据进行训练。
硬模版搜索：尝试不同的 Prompt 模版，寻找对当前数据集效果最佳的表达方式。
工程化部署：将推理部分的字符串匹配优化为 Tensor 操作，提升线上服务响应速

项目中的预训练模型(bert-base-chinese), 下载地址: 通过网盘分享的文件,链接: https://pan.baidu.com/s/1UnHXBJUpib1m7-LIE4oNjA?pwd=cywd 提取码: cywd

##### 具体步骤详解博客: https://blog.csdn.net/zhoupenghui168/article/details/162371694
