# -*- coding:utf-8 -*-
import os
from typing import Union, List
# Union 是 typing 模块中定义的一个类,用于表示多个类型中的任意一种类型
from pet_config import *

pc = ProjectConfig()


class Verbalizer(object):
    """
    Verbalizer类，用于将一个Label对应到其子Label的映射。
    """

    def __init__(self, verbalizer_file: str, tokenizer, max_label_len: int):
        """
        Args:
            verbalizer_file (str): verbalizer文件存放地址。
            tokenizer: 用于文本和id之间的转换。
            max_label_len (int): 标签长度，若大于则截断，若小于则补齐
        """
        self.tokenizer = tokenizer
        self.label_dict = self.load_label_dict(verbalizer_file)
        self.max_label_len = max_label_len

    def load_label_dict(self, verbalizer_file: str):
        """
        读取本地文件，构建verbalizer字典。

        Args:
            verbalizer_file (str): verbalizer文件存放地址。

        Returns:
            dict -> {
                '体育': ['篮球', '足球','网球', '排球',  ...],
                '酒店': ['宾馆', '旅馆', '旅店', '酒店', ...],
                ...
            }
        """
        label_dict = {}
        with open(verbalizer_file, 'r', encoding='utf8') as f:
            for line in f.readlines():
                # print(f'line-->{line}') # 电脑	电脑
                label, sub_labels = line.strip().split('\t') # 以制表符进行拆分
                label_dict[label] = list(set(sub_labels.split(','))) # 子标签如果有多个,按照逗号进行拆分, 然后使用set元组(去重)存放到list中

        return label_dict

    def find_sub_labels(self, label: Union[list, str]):
        """
        通过主标签找到所有的子标签。

        Args:
            label (Union[list, str]): 标签, 文本型 或 id_list, e.g. -> '体育' or [860, 5509]
        
        Returns:
            dict -> {
                'sub_labels': ['足球', '网球'], # 子标签
                'token_ids': [[6639, 4413], [5381, 4413]] # 子标签对应的token id
            }
        """
        if type(label) == list:  # 如果传入为id_list, 则通过tokenizer转回来
            while self.tokenizer.pad_token_id in label:
                label.remove(self.tokenizer.pad_token_id) # 删除补齐的标签 pad token id
            label = ''.join(self.tokenizer.convert_ids_to_tokens(label)) # 把标签token id 转换成标签字符串
        # print(f'label-->{label}') #  label--> 电脑

        if label not in self.label_dict: # 判断标签是否在字典中
            raise ValueError(f'Lable Error: "{label}" not in label_dict {list(self.label_dict)}.')

        sub_labels = self.label_dict[label] # 通过主标签获取子标签

        # print(f'sub_labels-->{sub_labels}') # sub_labels: 电脑
        ret = {'sub_labels': sub_labels}

        # self.tokenizer(sub_labels)--> [[101, 4510, 5554, 102]], 其中101表示[CLS], 102表示[SEP], 中间的[1:-1]表示子标签input ids
        # print(f"self.tokenizer(sub_labels)-->{self.tokenizer(sub_labels)}")

        # 列表推导式: self.tokenizer(sub_labels)['input_ids' => 获取子标签input_ids
        token_ids = [_id[1:-1] for _id in self.tokenizer(sub_labels)['input_ids']]
        # token_ids--> [[4510, 5554]]
        # print(f'token_ids-->{token_ids}')

        for i in range(len(token_ids)):
            token_ids[i] = token_ids[i][:self.max_label_len]  # 对标签进行截断与补齐
            if len(token_ids[i]) < self.max_label_len:
                token_ids[i] = token_ids[i] + [self.tokenizer.pad_token_id] * (self.max_label_len - len(token_ids[i]))
        # ret中有sub_labels, 表示子标签list, 还有token_ids, 表示对应子标签的token id
        ret['token_ids'] = token_ids
        return ret

    def batch_find_sub_labels(self, label: List[Union[list, str]]):
        """
        批量找到子标签。

        Args:
            label (List[list, str]): 标签列表, [[4510, 5554], [860, 5509]] or ['体育', '电脑']

        Returns:
            list -> [
                        {
                            'sub_labels': ['笔记本', '电脑'], 
                            'token_ids': [[5011, 6381, 3315], [4510, 5554]]
                        },
                        ...
                    ]
        """
        # 通过列表生成式调用find_sub_labels获取数据
        return [self.find_sub_labels(l) for l in label]

    def get_common_sub_str(self, str1: str, str2: str):
        """
        解决“最长公共子串”问题的经典动态规划实现。它通过构建一个二维表格来记录中间状态，从而高效地找到两个字符串中最长的、连续的公共部分
        寻找最大公共子串(连续子序列)。
        str1:abcd
        str2:abadbcdba
        结果: bcd
        """
        lstr1, lstr2 = len(str1), len(str2)
        # 生成0矩阵，为方便后续计算，比字符串长度多了一列
        """
        它创建了一个 (lstr1 + 1) x (lstr2 + 1) 的二维矩阵，并全部初始化为 0。
        为什么尺寸要 +1？ 这是为了处理边界情况。record[i][j] 将代表 str1 的前 i 个字符和 str2 的前 j 个字符的匹配情况。
        多出的一行和一列（索引为0）作为“哨兵”，可以简化后续的逻辑，避免数组越界检查
        """
        record = [[0 for i in range(lstr2 + 1)] for j in range(lstr1 + 1)]
        p = 0  # 最长匹配对应在str1中的最后一位: 一个指针，用于记录当前找到的最长公共子串在 str1 中结束位置的下一个位置
        maxNum = 0  # 最长匹配长度: 一个变量，用于记录当前找到的最长公共子串的长度

        # 两个嵌套循环: 用于遍历两个字符串的每一个字符组合，逐一进行比较
        for i in range(lstr1):
            for j in range(lstr2):
                if str1[i] == str2[j]: # 核心判断: 如果 str1 的第 i 个字符和 str2 的第 j 个字符相等，说明找到了一个匹配的字符
                    """
                    状态转移方程: 这是整个算法的灵魂。
                        如果 str1[i] == str2[j]，那么以这两个字符结尾的公共子串长度，就等于它们前一个字符结尾的公共子串长度加 1。
                        record[i][j] 存储的是 str1[0...i-1] 和 str2[0...j-1] 的匹配信息。
                        因此，record[i+1][j+1] (代表考虑 str1[i] 和 str2[j]) 的值就是 record[i][j] + 1。
                        如果字符不相等，record[i+1][j+1] 会保持初始值 0，表示以这两个字符结尾没有公共子串
                    """
                    record[i + 1][j + 1] = record[i][j] + 1
                    """
                    更新最优解:
                        每次计算出新的匹配长度 record[i + 1][j + 1] 后，都与当前记录的最大长度 maxNum 进行比较。
                        如果新长度更长，就更新 maxNum。
                        同时，更新指针 p。因为 record 矩阵的索引比字符串索引大 1，所以 str1 中匹配结束的位置是 i，而 p 被设为 i + 1，这恰好是切片操作的结束索引
                    """
                    if record[i + 1][j + 1] > maxNum:
                        maxNum = record[i + 1][j + 1]
                        p = i + 1
        """
        返回结果:
            循环结束后，maxNum 是最长公共子串的长度，p 是该子串在 str1 中结束位置的下一个索引。
            通过 Python 的切片 str1[p - maxNum:p]，我们可以精确地截取出这个最长的公共子串。
            函数最终返回子串本身和它的长度
        """
        return str1[p - maxNum:p], maxNum

    def hard_mapping(self, sub_label: str):
        """
        强匹配函数: 代码是 PET (Pattern-Exploiting Training) 框架中推理阶段（Inference）的一个关键容错机制,
            在 PET 模型中，模型通过预测 [MASK] 位置的词来进行分类,
            但有时候，模型预测出的词可能不在我们预定义的标签字典（label_dict）中,
            这个 hard_mapping 函数的作用就是：当模型“猜错”或生成了未知的词时，通过计算字符串相似度，
            将其“强行映射”回最接近的正确主标签（main_label）

            当模型生成的子label不存在时，通过最大公共子串找到重合度最高的主label。

        Args:
            sub_label (str): 子label。

        Returns:
            str: 主label。
        """
        # 设计意图：接收模型预测出的 sub_label，准备在字典中寻找最匹配的项
        label, max_overlap_str = '', 0 # max_overlap_str:记录当前最高的重叠度分数（初始为0）
        # print(self.label_dict.items())

        for main_label, sub_labels in self.label_dict.items(): # main_label: 主标签（如 "电脑", "手机"）, sub_labels: 该主标签下所有可能的子标签列表（如 ["笔记本", "电脑", "计算机"]）
            overlap_num = 0 # 字符串重复长度, 初始值为0
            for s_label in sub_labels:  # 求所有子label与当前推理label之间的最长公共子串长度
                # print(f'self.get_common_sub_str(sub_label, s_label)-->{self.get_common_sub_str(sub_label, s_label)}')
                # 字符串重复长度进行累加, 后面就可以判断每一个主label对应的子label的最长公共子串长度, 找到最大最大重叠度字符串的那个主label就是需要的label
                # 计算模型预测词与当前子标签的最长公共子串长度，并累加到 overlap_num 中
                overlap_num += self.get_common_sub_str(sub_label, s_label)[1]
            # print(f'main_label, sub_labels, overlap_num--》{main_label, sub_labels, overlap_num}')

            # 更新最优解
            # 设计意图：如果当前主标签的累计重叠度大于或等于历史最高分，就更新最高分，并将当前主标签作为最佳候选。
            # 细节注意：这里使用了 >=。这意味着如果两个主标签的重叠度完全一样，代码会返回在字典中遍历到较晚的那个主标签
            if overlap_num >= max_overlap_str:
                max_overlap_str = overlap_num
                label = main_label

        return label

    def find_main_label(self, sub_label: Union[list, str], hard_mapping=True):
        """
        通过子标签找到父标签。

        Args:
            sub_label (Union[list, str]): 子标签, 文本型 或 id_list, e.g. -> '苹果' or [5741, 3362]
            hard_mapping (bool): 当生成的词语不存在时，是否一定要匹配到一个最相似的label。

        Returns:
            dict -> {
                'label': '水果',  #主标签
                'token_ids': [3717, 3362] # 主标签token id
            }
        """
        if type(sub_label) == list:  # 如果传入为id_list, 则通过tokenizer转回来
            pad_token_id = self.tokenizer.pad_token_id
            while pad_token_id in sub_label:  # 移除[PAD]token
                sub_label.remove(pad_token_id)
            sub_label = ''.join(self.tokenizer.convert_ids_to_tokens(sub_label))
        # print(f'sub_label-->{sub_label}')
        main_label = '无'
        for label, s_labels in self.label_dict.items(): # 循环标签dict
            if sub_label in s_labels: # 如果子标签在s_labels,说明找到主标签
                main_label = label
                break
        # print(f'main_label--》{main_label}')
        if main_label == '无' and hard_mapping:
            main_label = self.hard_mapping(sub_label)
        # print('强匹配', main_label)
        ret = {
            'label': main_label,
            'token_ids': self.tokenizer(main_label)['input_ids'][1:-1]
        }
        # 返回主标签以及对应的标签input id
        return ret

    def batch_find_main_label(self, sub_label: List[Union[list, str]], hard_mapping=True):
        """
        批量通过子标签找父标签。

        Args:
            sub_label (List[Union[list, str]]): 子标签列表, ['苹果', ...] or [[5741, 3362], ...]

        Returns:
            list: [
                    {
                    'label': '水果', 
                    'token_ids': [3717, 3362]
                    },
                    ...
            ]
        """
        return [self.find_main_label(l, hard_mapping) for l in sub_label]


if __name__ == '__main__':
    from rich import print
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(pc.pre_model)
    verbalizer = Verbalizer(
        verbalizer_file=pc.verbalizer,
        tokenizer=tokenizer,
        max_label_len=2
    )
    # print(verbalizer.label_dict)

    # 测试 find_sub_labels方法
    # label = [4510, 5554]
    # ret = verbalizer.find_sub_labels(label)
    # # {'sub_labels': ['电脑'], 'token_ids': [[4510, 5554]]}
    # # print(ret)
    # label = '电脑'
    # ret = verbalizer.find_sub_labels(label)
    # # {'sub_labels': ['电脑'], 'token_ids': [[4510, 5554]]}
    # print(ret)

    # print('*' * 80)
    # labels = ['电脑', '衣服']
    # labels = [[4510, 5554], [6132, 3302]]
    # result = verbalizer.batch_find_sub_labels(labels)
    """
    result--》[
        {'sub_labels': ['电脑'], 'token_ids': [[4510, 5554]]}, 
        {'sub_labels': ['衣服'], 'token_ids': [[6132, 3302]]}
    ]
    """
    # print(f'result--》{result}')

    # sub_label = [6132, 4510]
    # sub_label = [4510, 5554]
    # sub_label = "洗发水"
    # a = verbalizer.tokenizer.convert_ids_to_tokens(sub_label)
    # print(a)
    # ret = verbalizer.find_main_label(sub_label)
    # {'label': '洗浴', 'token_ids': [3819, 3861]}
    # print(ret)

    sub_label = ['衣服', '牛奶']
    # sub_label = [[2506, 2506]]
    ret = verbalizer.batch_find_main_label(sub_label, hard_mapping=True)
    # [
    #     {'label': '衣服', 'token_ids': [6132, 3302]},
    #     {'label': '蒙牛', 'token_ids': [5885, 4281]}
    # ]
    print(ret)
