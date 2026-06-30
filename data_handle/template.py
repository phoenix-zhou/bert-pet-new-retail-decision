# -*- coding:utf-8 -*-
from rich import print
from transformers import AutoTokenizer
import numpy as np
from pet_config import *

class HardTemplate(object):
    """
    硬模板，人工定义句子和[MASK]之间的位置关系。
    """

    def __init__(self, prompt: str):
        """
        Args:
            prompt (str): prompt格式定义字符串, e.g. -> "这是一条{MASK}评论：{textA}。"
        """
        self.prompt = prompt
        self.inputs_list = []                       # 根据文字prompt拆分为各part的列表:  一个列表，用于存储解析后的模板元素（包含普通字符和占位符
        self.custom_tokens = set(['MASK'])          # 从prompt中解析出的自定义token集合(天然去重), 这里后面就是prompt.txt中的{MASK}, {textA}
        # {'MASK'}
        self.prompt_analysis()                         # 解析prompt模板:在初始化时立即调用解析方法，完成模板的拆解


    def prompt_analysis(self):
        """
        将prompt文字模板拆解为可映射的数据结构。
        字符串状态机解析逻辑，通过单指针 idx 遍历整个模板字符串,
        在后续的代码中，程序会遍历 inputs_list：遇到普通字符就保留，遇到 MASK 就替换为真实的 [MASK] token，
        遇到 textA 就替换为用户传入的真实文本内容，最终拼接成完整的句子并送入 Tokenizer 进行编码。
        这种设计使得模板的修改极其灵活，只需更改一行配置字符串即可
        Examples:
            prompt -> "这是一条{MASK}评论：{textA}。"
            inputs_list -> ['这', '是', '一', '条', 'MASK', '评', '论', '：', 'textA', '。']
            custom_tokens -> {'textA', 'MASK'}
        """
        # print(f'prompt-->{self.prompt}')
        idx = 0
        while idx < len(self.prompt):
            str_part = ''
            # 1. 处理普通字符
            if self.prompt[idx] not in ['{', '}']:
                self.inputs_list.append(self.prompt[idx])
            # 2. 处理自定义字段（遇到 '{' 开始捕获）
            if self.prompt[idx] == '{': # 进入自定义字段
                # 占位符捕获：当遇到 { 时，进入内部 while 循环，不断拼接字符直到遇到 }，
                # 从而提取出完整的占位符名称（如 MASK 或 textA）
                idx += 1
                while self.prompt[idx] != '}':
                    str_part += self.prompt[idx] # 拼接该自定义字段的值
                    idx += 1
            # 3. 异常处理（遇到 '}' 但没有前置的 '{'）
            elif self.prompt[idx] == '}':
                raise ValueError("Unmatched bracket '}', check your prompt.")
            # 4. 记录捕获到的自定义字段
            if str_part:
                # 更新数据结构：如果成功提取到了 str_part（即占位符名称），将其追加到 inputs_list 中，并加入 custom_tokens 集合中
                self.inputs_list.append(str_part)
                self.custom_tokens.add(str_part) # 将所有自定义字段存储，后续会检测输入信息是否完整
            idx += 1

    # 方法签名与初始化
    def __call__(self,
                 inputs_dict: dict,
                 tokenizer,
                 mask_length,
                 max_seq_len=512):
        """
        输入一个样本，转换为符合模板的格式。

        Args:
            inputs_dict (dict): 包含真实文本和 MASK 标记的字典，prompt中的参数字典, e.g. -> {
                                                            "textA": "这个手机也太卡了", 
                                                            "MASK": "[MASK]"
                                                        }
            tokenizer: Hugging Face 的分词器对象，用于将字符串转为 Token ID, 用于encoding文本
            mask_length (int): MASK token 的长度, [MASK] 标记重复的次数（用于多词标签预测）
            max_seq_len: 序列最大长度，用于截断和 Padding

        初始化输出字典：预定义了返回的数据结构，包含原始文本、Token ID、Token 类型、注意力掩码以及 MASK 的位置索引
        Returns:
            dict -> {
                'text': '[CLS]这是一条[MASK]评论：这个手机也太卡了。[SEP]',
                'input_ids': [1, 47, 10, 7, 304, 3, 480, 279, 74, 47, 27, 247, 98, 105, 512, 777, 15, 12043, 2],
                'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                'mask_position': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            }
        """
        # 定义输出格式
        outputs = {
            'text': '',
            'input_ids': [],
            'token_type_ids': [],
            'attention_mask': [],
            'mask_position': []
        }

        str_formated = ''
        """
        eg: 
            inputs_list-->['这', '是', '一', '条', 'MASK', '评', '论', '：', 'textA', '。']
            custom_tokens--》{'textA', 'MASK'}
        """
        """
        这是整个类最核心的逻辑，它遍历了 prompt_analysis 阶段解析出的 inputs_list：
        如果是普通字符（如 '这', '是'）：直接拼接到 str_formated 中。
        如果是自定义占位符（如 'textA'）：从 inputs_dict 中提取真实的文本并拼接。
        如果是 MASK 占位符：从字典中提取 [MASK]，并根据 mask_length 参数进行乘法重复。
        例如 mask_length=2 时，会拼接成 [MASK][MASK]。这对于 PET 预测多字标签（如“非常满意”）至关重
        """
        for value in self.inputs_list:
            if value in self.custom_tokens:
                if value == 'MASK':
                    str_formated += inputs_dict[value] * mask_length
                else:
                    str_formated += inputs_dict[value]
            else:
                str_formated += value
        # str_formated-->这是一条[MASK][MASK]评论：包装不错，苹果挺甜的，个头也大。。
        # print(f'str_formated-->{str_formated}')

        # 对输入的数据进行编码
        """
        调用 tokenizer 对拼接好的完整字符串进行编码。
        truncation=True: 如果文本超过 max_seq_len，自动截断。
        padding='max_length': 如果文本长度不足，自动在末尾填充 [PAD]（对应 ID 为 0），直到达到 max_seq_len。
        这保证了输入 Batch 的形状一致。
        将编码后的三个核心张量存入 outputs 字典
        """
        encoded = tokenizer(text=str_formated,
                            truncation=True,
                            max_length=max_seq_len,
                            padding='max_length')
        # print('*'*80)
        """
        encoded--->{'input_ids': [101, 6821, 3221, 671, 3340, 103, 103, 6397, 6389, 
            8038, 1259, 6163, 679, 7231, 8024, 5741, 3362, 2923, 4494, 4638, 8024, 702, 
            1928, 738, 1920, 511, 511, 102, 0, 0], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 1, 1, 1, 0, 0]}
        input_ids:输入句子转换成token_id:  input_ids, input_ids 中为0则表示补齐
        token_type_ids: token_type_ids类型
        attention_mask: 掩码
        """
        # print(f'encoded--->{encoded}')
        outputs['input_ids'] = encoded['input_ids']
        outputs['token_type_ids'] = encoded['token_type_ids']
        outputs['attention_mask'] = encoded['attention_mask']

        # 还原可读文本:
        #       将 Token ID 列表反向转换回 Token 字符串列表，再拼接成一个完整的字符串。
        #       这通常用于调试或日志打印，让你能直观看到包含 [PAD] 的最终输入长什么样
        # [ '[CLS]', '这','是','一','条','[MASK]','[MASK]','评','论','：','包','装','不','错','，','苹','果','挺','甜','的','，',  '个','头', '也','大', '。',  '。', '[SEP]','[PAD]''[PAD]']
        # print(tokenizer.convert_ids_to_tokens(encoded['input_ids']))
        # outputs['text']:[CLS]这是一条[MASK][MASK]评论：包装不错，苹果挺甜的，个头也大。。[SEP][PAD][PAD]
        outputs['text'] = ''.join(tokenizer.convert_ids_to_tokens(encoded['input_ids']))
        """
        outputs-->{'text': 
            '[CLS]这是一条[MASK][MASK]评论：包装不错，苹果挺甜的，个头也大。。[SEP][PAD][PA
            D]', 'input_ids': [101, 6821, 3221, 671, 3340, 103, 103, 6397, 6389, 8038, 
            1259, 6163, 679, 7231, 8024, 5741, 3362, 2923, 4494, 4638, 8024, 702, 1928, 
            738, 1920, 511, 511, 102, 0, 0], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 1, 1, 1, 0, 0], 'mask_position': []}
        """
        # print(f'outputs-->{outputs}')
        # print('*' * 80)

        # 定位 [MASK] 的位置
        """
        获取 MASK 的 ID：在 BERT 中文模型中，[MASK] 的 ID 通常是 103。
        NumPy 掩码查找：将 input_ids 转为 NumPy 数组，利用 np.where 找出所有等于 mask_token_id 的索引位置。
        保存位置：将找到的位置列表（如 ``）存入 outputs['mask_position']。
        为什么需要这个？ 
            在 PET 推理/训练时，模型会输出整个序列的词表概率分布（Logits）。
            只需要提取 [MASK] 所在位置的 Logits，然后通过 Verbalizer 映射到具体的分类标签上。
            因此，精确知道 MASK 的位置是 PET 算法的必经之路
        """
        # print(tokenizer.convert_tokens_to_ids(['[MASK]']))
        # 找到MASK的token_id
        mask_token_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
        # print(f'mask_token_id-->{mask_token_id}') # mask_token_id-->103
        # print('*' * 80)
        # [False False False False False  True  True False False False False False
        #  False False False False False False False False False False False False
        #  False False False False False False]
        # print(np.array(outputs['input_ids']) == mask_token_id)
        # print('*'*80)
        print(np.where(np.array(outputs['input_ids']) == mask_token_id)) # (array([5, 6], dtype=int64),)
        mask_position = np.where(np.array(outputs['input_ids']) == mask_token_id)[0].tolist()
        # print(f'mask_position--》{mask_position}') # mask_position--》[5, 6]
        outputs['mask_position'] = mask_position

        return outputs


if __name__ == '__main__':
    pc = ProjectConfig()
    tokenizer = AutoTokenizer.from_pretrained(pc.pre_model)
    hard_template = HardTemplate(prompt='这是一条{MASK}评论：{textA}。')

    # inputs_list -->['这', '是', '一', '条', 'MASK', '评', '论', '：', 'textA', '。']
    # print(f'inputs_list-->{hard_template.inputs_list}')

    # custom_tokens - -》{'textA', 'MASK'}
    # print(f'custom_tokens--》{hard_template.custom_tokens}')

    # print(f"inputs_dict--》{{'textA': '包装不错，苹果挺甜的，个头也大。', 'MASK': '[MASK]'}}")
    tep = hard_template(
                inputs_dict={'textA': '包装不错，苹果挺甜的，个头也大。', 'MASK': '[MASK]'},
                tokenizer=tokenizer,
                max_seq_len=30,
                mask_length=2)
    # print(tep)

    print(tokenizer("这是一条[MASK][MASK]评论：包装不错，苹果挺甜的，个头也大。。"))
    print(tokenizer.convert_tokens_to_ids(['网', '球'])) # 5381, 4413]
