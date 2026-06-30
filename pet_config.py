# coding:utf-8
import torch
import sys
import faulthandler
faulthandler.enable(all_threads=True)

class ProjectConfig(object):
    def __init__(self):
        # 是否使用GPU
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # windows电脑/linux服务器
        # self.device = "mps:0" # MAC电脑
        # 预训练模型bert路径
        self.pre_model = '/www/PET/bert-base-chinese'
        # 训练集和验证集的数据文件路径
        self.train_path = '/www/PET/data/train.txt'
        self.dev_path = '/www/PET/data/dev.txt'
        # 定义 Prompt 模板的文件路径
        self.prompt_file = '/www/PET/data/prompt.txt'
        # 定义标签词映射（将模型预测的 token 映射回实际分类标签）的文件路径
        self.verbalizer = '/www/PET/data/verbalizer.txt'
        # 输入文本的最大序列长度，超过此长度的文本将被截断
        self.max_seq_len = 256
        # 每次训练迭代送入模型的样本数量
        self.batch_size = 8
        # 学习率
        self.learning_rate = 5e-5
        # 权重衰减系数，用于防止过拟合，这里设为 0 表示不使
        self.weight_decay = 0
        # 预热学习率(用来定义预热的步数), 学习率预热比例
        # 在训练初期，学习率会从 0 线性增加到设定值，防止模型在初期因梯度不稳定而崩溃。
        # 这里表示预热的步数占总训练步数的 6%
        self.warmup_ratio = 0.06
        # 标签的最大长度（在 PET 中，标签通常由一个或几个词组成
        self.max_label_len = 2
        #  整个数据集被遍历训练的轮数
        self.epochs = 20
        # 每训练 2 个 step 打印一次训练日志（如 loss 等）
        self.logging_steps = 2
        # 每训练 20 个 step 在验证集上进行一次评估
        self.valid_steps = 20
        # 训练过程中保存模型权重（checkpoints）的本地目录路径
        self.save_dir = '/www/PET/checkpoints'


if __name__ == '__main__':
    pc = ProjectConfig()
    print(pc.prompt_file)
    print(pc.pre_model)
