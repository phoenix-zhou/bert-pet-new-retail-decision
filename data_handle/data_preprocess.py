# 导入必备工具包
import torch
import numpy as np
from .template import *
from rich import print
from datasets import load_dataset
from functools import partial # partial是对函数进行再次封装，便于使用
from pet_config import *
import faulthandler
faulthandler.enable(all_threads=True)

def convert_example(
        examples: dict,
        tokenizer,
        max_seq_len: int,
        max_label_len: int,
        hard_template: HardTemplate,
        train_mode=True,
        return_tensor=False) -> dict:
    # print("*"*80)
    """
    将样本数据转换为模型接收的输入数据。

    Args:
        examples (dict): 训练数据样本, e.g. -> {
                                                "text": [
                                                            '手机	这个手机也太卡了。',
                                                            '体育	世界杯为何迟迟不见宣传',
                                                            ...
                                                ]
                                            }
        max_seq_len (int): 句子的最大长度，若没有达到最大长度，则padding为最大长度
        max_label_len (int): 最大label长度，若没有达到最大长度，则padding为最大长度
        hard_template (HardTemplate): 模板类。
        train_mode (bool): 训练阶段 or 推理阶段。
        return_tensor (bool): 是否返回tensor类型，如不是，则返回numpy类型。

    Returns:
        dict (str: np.array) -> tokenized_output = {
                            'input_ids': [[1, 47, 10, 7, 304, 3, 3, 3, 3, 47, 27, 247, 98, 105, 512, 777, 15, 12043, 2], ...],
                            'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ...],
                            'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], ...],
                            'mask_positions': [[5, 6, 7, 8], ...],
                            'mask_labels': [[2372, 3442, 0, 0], [2643, 4434, 2334, 0], ...]
                        }
    """
    tokenized_output = {
        'input_ids': [],
        'token_type_ids': [],
        'attention_mask': [],
        'mask_positions': [],
        'mask_labels': []
    }
    """
        examples--》{'text': ['手机\t这个手机也太卡了。', 
        '体育\t世界杯为何迟迟不见宣传']}
    """
    # print(f'examples--》{examples}')
    for i, example in enumerate(examples['text']):
        if train_mode:
            # print(f'example-->{example}') # example-->手机  这个手机也太卡了。
            label, content = example.strip().split('\t')
            # print(f'label-->{label}') # label-->手机
            # print(f'content-->{content}') #content-->这个手机也太卡了。
        else:
            content = example.strip()

        inputs_dict = {
            'textA': content,
            'MASK': '[MASK]'
        }
        # print(f'inputs_dict-->{inputs_dict}') # inputs_dict-->{'textA': '这个手机也太卡了。', 'MASK': '[MASK]'}

        encoded_inputs = hard_template(
            inputs_dict=inputs_dict,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            mask_length=max_label_len)
        """
                encoded_inputs--》{'text': 
        '[CLS]这是一条[MASK][MASK]评论：这个手机也太卡了。[SEP][PAD][PAD][PAD][PAD][PAD
        ][PAD][PAD][PAD][PAD][PAD]', 'input_ids': [101, 6821, 3221, 671, 3340, 103, 
        103, 6397, 6389, 8038, 6821, 702, 2797, 3322, 738, 1922, 1305, 749, 511, 102, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': 
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0], 'mask_position': [5, 6]}
        """
        # print(f'encoded_inputs--》{encoded_inputs}')
        # print('*'*80)
        tokenized_output['input_ids'].append(encoded_inputs["input_ids"])
        tokenized_output['token_type_ids'].append(encoded_inputs["token_type_ids"])
        tokenized_output['attention_mask'].append(encoded_inputs["attention_mask"])
        tokenized_output['mask_positions'].append(encoded_inputs["mask_position"])
        """
        tokenized_output-->{'input_ids': [[101, 6821, 3221, 671, 3340, 103, 103, 6397, 
            6389, 8038, 6821, 702, 2797, 3322, 738, 1922, 1305, 749, 511, 102, 0, 0, 0, 0, 
            0, 0, 0, 0, 0, 0]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
            0, 0]], 'mask_positions': [[5, 6]], 'mask_labels': []}
        """
        # print(f'tokenized_output-->{tokenized_output}')

        # mask_labels: [] 为空, 通过下面代码处理
        if train_mode: # 训练模式
            # print(f'label--》{label}') # label--》手机
            # 将文本标签（如“电脑”）也通过 tokenizer 编码成 ID（如 [2797, 3322]）
            label_encoded = tokenizer(text=[label])  # 对label(也就是abel--》手机)进行编码, 将label补到最大长度
            # label_encoded-->{'input_ids': [[101, 2797, 3322, 102]], 'token_type_ids': [[0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1]]}
            # print(f'label_encoded-->{label_encoded}')

            label_encoded = label_encoded['input_ids'][0][1:-1] #  [0][1:-1]: 表示获取[[101, 2797, 3322, 102]]中手机对应的编码,101,102表示开始[CLS],[SEP]
            # print(f'label_encoded-->{label_encoded}') # label_encoded-->[2797, 3322]

            # 截断或填充：确保标签长度符合 max_label_len，不足补 0，超出截断
            label_encoded = label_encoded[:max_label_len] # 获取最大数标签长度, 超过截断,不足补齐(用0补齐)
            # print(f'tokenizer.pad_token_id-->{tokenizer.pad_token_id}') # tokenizer.pad_token_id-->0
            label_encoded = label_encoded + [tokenizer.pad_token_id] * (max_label_len - len(label_encoded))

            tokenized_output['mask_labels'].append(label_encoded)
    for k, v in tokenized_output.items():
        if return_tensor: # 是否返回tensor张量格式
            tokenized_output[k] = torch.LongTensor(v)
        else:
            tokenized_output[k] = np.array(v) # 返回array

    return tokenized_output


if __name__ == '__main__':
    pc = ProjectConfig()
    # load_dataset加载对应文件类型数据
    train_dataset = load_dataset('text', data_files=pc.train_path)
    """
    train_dataset-->DatasetDict({
    train: Dataset({
        features: ['text'],
        num_rows: 63
        })
    })
    """
    # print(f'train_dataset-->{train_dataset}') # 获取训练集数据格式
    # print('*'*80)
    """
    Dataset({
        features: ['text'],
        num_rows: 63
    })
    """
    # print(train_dataset['train'])
    # print('*'*80)
    """
        Column(['电脑\t(1)这款笔记本外观感觉挺漂亮的，分量吗，对我来说不算沉。 
    (2)安装了WindowsXP系统后，运行的速度挺快。发热量没有想象中那么大。可能尚未运行
    很耗资源的程序，没有感到内存的弊病。不过，1G的内存确实有点小。 
    (3)附赠的包很不错，挺有手感的。但是附赠的鼠标实在是太小了，幸好同时订了一个双飞
    燕的鼠标哟。', 
    '水果\t什么苹果啊，都没有苹果味，怪怪的味道，而且一点都不甜，超级难吃！', 
    '平板\t价格便宜送货快，质量挺好的', '书籍\t当时是因为要写读书报告 
    所以上网到处抄 
    就发现这本书感觉以前自己没看过中文版的骆驼祥子就看过电影所以决定买个英文的看看
    本来以为 是中文翻译过来的 不会太地道结果 
    令我惊讶翻译的非常好！忠实了原著！很多很多复杂的复合句 而且 用词也很得体觉得 
    是个 非常不错的拓宽知识面的一本小说推荐！', 
    '水果\t很差的果，表面色泽就已经看到不新鲜啦，跟图片相比，简直一个天一个地，还有
    烂果，果大细不一致，小到哭。', ...])
    """
    # print(train_dataset['train']['text']) # 获取训练集里面的样本列表数据

    tokenizer = AutoTokenizer.from_pretrained(pc.pre_model)
    hard_template = HardTemplate(prompt='这是一条{MASK}评论：{textA}')
    # 举例examples, 使用convert_example来进行处理样本,返回 tokenized_output
    # examples = {"text": ['手机	这个手机也太卡了。','体育	世界杯为何迟迟不见宣传']}
    #
    # tokenized_output = convert_example(examples=examples,
    #                                     tokenizer=tokenizer,
    #                                     max_seq_len=30,
    #                                     max_label_len=2,
    #                                     hard_template=hard_template,
    #                                     train_mode=True,
    #                                     return_tensor=False)

    """
    返回的是input_ids两个二维数组, 因为样本是两个, 见变量examples
    tokenized_output-->{'input_ids': array([[ 101, 6821, 3221,  671, 3340,  103,  
        103, 6397, 6389, 8038, 6821,
                 702, 2797, 3322,  738, 1922, 1305,  749,  511,  102,    0,    0,
                   0,    0,    0,    0,    0,    0,    0,    0],
               [ 101, 6821, 3221,  671, 3340,  103,  103, 6397, 6389, 8038,  686,
                4518, 3344,  711,  862, 6826, 6826,  679, 6224, 2146,  837,  102,
                   0,    0,    0,    0,    0,    0,    0,    0]]), 'token_type_ids': 
        array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': array([[1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0],
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 0]]), 'mask_positions': array([[5, 6],
               [5, 6]]), 'mask_labels': array([[2797, 3322],
               [ 860, 5509]])}
    """
    # print(f'tokenized_output-->{tokenized_output}')

    # 使用 functools.partial 包装 convert_example 函数，预先填入 tokenizer、template 等固定参数，生成 convert_fun
    # 实际上是使用partial集成方法来处理样本数据, 使用partial把convert_example函数再次封装,进行下面处理
    convert_func = partial(convert_example,
                           tokenizer=tokenizer,
                           hard_template=hard_template,
                           max_seq_len=30,
                           max_label_len=2,
                           )

    # 调用 train_dataset.map(convert_func, batched=True)：这是 Hugging Face datasets 库的高效操作。
    # 它会批量地将 convert_example 应用到数据集的每一行，自动完成所有数据的向量化处理
    # batched=True相当于将train_dataset看成一个批次的样本直接对数据进行处理，节省时间
    dataset = train_dataset.map(convert_func, batched=True)
    """
    dataset-->DatasetDict({
            train: Dataset({
                features: ['text', 'input_ids', 'token_type_ids', 'attention_mask', 
        'mask_positions', 'mask_labels'],
                num_rows: 63
            })
        })
    """
    # print(f"dataset-->{dataset}")
    """
     dataset['train']--》Dataset({
        features: ['text', 'input_ids', 'token_type_ids', 'attention_mask', 
        'mask_positions', 'mask_labels'],
        num_rows: 63
    })
    """
    # print(f"dataset['train']--》{dataset['train']}")
    # print("*"*80)
    # print(len(dataset['train'])) # 63
    # print("*" * 80)
    print(dataset["train"][0])
    for value in dataset['train']:
        """
        {
            'text': '电脑\t(1)这款笔记本外观感觉挺漂亮的，分量吗，对我来说不算沉。 
        (2)安装了WindowsXP系统后，运行的速度挺快。发热量没有想象中那么大。可能尚未运行
        很耗资源的程序，没有感到内存的弊病。不过，1G的内存确实有点小。 
        (3)附赠的包很不错，挺有手感的。但是附赠的鼠标实在是太小了，幸好同时订了一个双飞
        燕的鼠标哟。',
            'input_ids': [101,6821,321,671,3340,103,103,6397,6389,8038,113,122,114,6821,3621,5011,6381,3315,1912,6225,2697,6230,2923,4023,778,4638,8024,1146,7030,102],
            'token_type_ids': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            'attention_mask': [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            'mask_positions': [5, 6],
            'mask_labels': [4510, 5554]
        }
        """
        print(value)
        print(len(value['input_ids'])) # 0
        print(type(value['input_ids'])) # <class 'list'>
        break
