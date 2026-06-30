# coding='utf-8'
"""

（多）分类问题下的指标评估（acc, precision, recall, f1）。

"""
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix

class ClassEvaluator(object):
    def __init__(self):
        self.goldens = [] # 真实的结果
        self.predictions = [] # 预测的结果

    def add_batch(self, pred_batch: List[List], gold_batch: List[List]):
        """
        添加一个batch中的prediction和gold列表，用于后续统一计算。

        Args:
            pred_batch (list): 模型预测标签列表, e.g. ->  [['体', '育'], ['财', '经'], ...]
            gold_batch (list): 真实标签标签列表, e.g. ->  [['体', '育'], ['财', '经'], ...]
        """
        assert len(pred_batch) == len(gold_batch) # 保证预测值样本长度和真实值样本长度一致, 这样才能进行逻辑处理
        # pred_batch0--》[['财', '经'], ['财', '经'], ['体', '育'], ['体', '育'], ['计', '算', '机']]
        # print(f'pred_batch0--》{pred_batch}')
        # gold_batch0--》[['体', '育'], ['财', '经'], ['体', '育'], ['计', '算', '机'], ['计', '算', '机']]
        # print(f'gold_batch0--》{gold_batch}')

        if type(gold_batch[0]) in [list, tuple]:  # 若遇到多个子标签构成一个标签的情况, 取第一个元素来判断类型即可
            pred_batch = [''.join([str(e) for e in ele]) for ele in pred_batch]  # 将所有的label拼接为一个整label: ['体', '育'] -> '体育'
            gold_batch = [''.join([str(e) for e in ele]) for ele in gold_batch]

        # pred_batch1--》['财经', '财经', '体育', '体育', '计算机']
        # print(f'pred_batch1--》{pred_batch}')
        # gold_batch1--》['体育', '财经', '体育', '计算机', '计算机']
        # print(f'gold_batch1--》{gold_batch}')

        self.goldens.extend(gold_batch) # 将当前批次的真实值数据填充进goldens去
        # self.goldens--》['体育', '财经', '体育', '计算机', '计算机']
        print(f'self.goldens--》{self.goldens}')

        self.predictions.extend(pred_batch) # 将当前批次的预测值数据填充进predictions去
        # self.predictions--->['财经', '财经', '体育', '体育', '计算机']
        print(f' self.predictions--->{ self.predictions}')

    def compute(self, round_num=2) -> dict:
        """
        根据当前类中累积的变量值，计算当前的P, R, F1, 能够快速地抓取模型哪里有问题, 可以通过增加/减少数据进行优化等
        Args:
            round_num (int): 计算结果保留小数点后几位, 默认小数点后2位。
        Returns:
            dict -> {
                'accuracy': 准确率,
                'precision': 精准率,
                'recall': 召回率,
                'f1': f1值,
                'class_metrics': {
                    '0': {
                            'precision': 该类别下的precision,
                            'recall': 该类别下的recall,
                            'f1': 该类别下的f1
                        },
                    ...
                }
            }
        """
        # self.goldens--》['体育', '财经', '体育', '计算机', '计算机']
        # print(f'self.goldens--》{self.goldens}')

        # self.predictions--->['财经', '财经', '体育', '体育', '计算机']
        # print(f'self.predictions--》{self.predictions}')

        # set(self.goldens) | set(self.predictions  ==> | 表示: 并集;  set: 集合(天然去重)
        # sorted: 排序
        # sorted(list(set(self.goldens) | set(self.predictions))): 表示把goldens,predictions数据混合到一起, 去重后再进行排序, 得到所有的类别
        classes, class_metrics, res = sorted(list(set(self.goldens) | set(self.predictions))), {}, {}

        # classes -->['体育', '计算机', '财经']
        print(f'classes-->{classes}')

        res['accuracy'] = round(accuracy_score(self.goldens, self.predictions), round_num)  # 构建全局指标
        res['precision'] = round(precision_score(self.goldens, self.predictions, average='weighted'), round_num)
        res['recall'] = round(recall_score(self.goldens, self.predictions, average='weighted'), round_num)
        res['f1'] = round(f1_score(self.goldens, self.predictions, average='weighted'), round_num)

        # res-->{'accuracy': 0.6, 'precision': 0.7, 'recall': 0.6, 'f1': 0.6}
        # print(f'res-->{res}') # 得到当前例子的 准确率, 精确率, 召回率, f1值, 考虑到weighted权重, 故里面的数值是不一样的

        # classes--》['体育', '计算机', '财经']
        # print(f'classes--》{classes}')

        # self.goldens--》['体育', '财经', '体育', '计算机', '计算机']
        # print(f'self.goldens--》{self.goldens}')

        # self.predictions - -》['财经', '财经', '体育', '体育', '计算机']
        # print(f'self.predictions--》{self.predictions}')

        try:
            # 混淆矩阵: 矩阵的行代表真实值（Actual），列代表预测值（Predicted）。对角线上的值代表预测正确的数量
            conf_matrix = np.array(confusion_matrix(self.goldens, self.predictions))  # (n_class, n_class)
            """
            conf_matrix-->[[1 0 1]  
                           [1 1 0]
                           [0 0 1]]
                    根据上面classes sorted可知, 混淆矩阵表示: 
                        体育 计算机 财经    预测
                    体育 [ 1    0    1 ]  
            真实   计算机 [ 1    1    0 ]  
                    财经 [ 0    0    1 ]  
            """
            # print(f'conf_matrix-->{conf_matrix}')

            assert conf_matrix.shape[0] == len(classes)
            for i in range(conf_matrix.shape[0]):  # 构建每个class的指标
                # 单独计算每个类别(0 体育, 1 计算机, 2 财经)的精确率, 召回率, f1值
                # Precision = TP / (TP + FP) -> 预测为 i 的样本中，有多少是真的 i
                precision = 0 if sum(conf_matrix[:, i]) == 0 else conf_matrix[i, i] / sum(conf_matrix[:, i])
                # Recall = TP / (TP + FN) -> 真实的 i 样本中，有多少被成功预测为 i
                recall = 0 if sum(conf_matrix[i, :]) == 0 else conf_matrix[i, i] / sum(conf_matrix[i, :])
                # F1 = 2 * P * R / (P + R)
                f1 = 0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
                class_metrics[classes[i]] = {
                    'precision': round(precision, round_num),
                    'recall': round(recall, round_num),
                    'f1': round(f1, round_num)
                }
            res['class_metrics'] = class_metrics
        except Exception as e:
            print(f'[Warning] Something wrong when calculate class_metrics: {e}')
            print(f'-> goldens: {set(self.goldens)}')
            print(f'-> predictions: {set(self.predictions)}')
            print(f'-> diff elements: {set(self.predictions) - set(self.goldens)}')
            res['class_metrics'] = {}

        return res

    def reset(self):
        """
        重置积累的数值
        """
        self.goldens = []
        self.predictions = []


if __name__ == '__main__':
    from rich import print

    metric = ClassEvaluator()
    metric.add_batch(
        [['财', '经'], ['财', '经'], ['体', '育'], ['体', '育'], ['计', '算', '机']],
        [['体', '育'], ['财', '经'], ['体', '育'], ['计', '算', '机'], ['计', '算', '机']],
    )
    # metric.add_batch(
    #     [0, 0, 1, 1, 0],
    #     [1, 1, 1, 0, 0]
    # )
    resu = metric.compute()
    """
    {
        'accuracy': 0.6,
        'precision': 0.7,
        'recall': 0.6,
        'f1': 0.6,
        'class_metrics': {
            '体育': {'precision': 0.5, 'recall': 0.5, 'f1': 0.5},
            '计算机': {'precision': 1.0, 'recall': 0.5, 'f1': 0.67},
            '财经': {'precision': 0.5, 'recall': 1.0, 'f1': 0.67}
        }
    }
    """
    print(resu)
