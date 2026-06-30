# coding:utf-8
# 导入 PyTorch 的核心数据加载器。它的作用是自动将数据集分批、打乱顺序，并提供多进程加载，是训练循环的标准组件
from torch.utils.data import DataLoader
# 导入 Hugging Face 提供的默认数据整理函数。它的作用是将一个批次（batch）的样本（通常是字典或列表）自动堆叠（stack）成一个张量（Tensor），并处理 padding
from transformers import default_data_collator
# 导入上面分析的 convert_example 函数和 HardTemplate 类。这是数据处理的逻辑核心
from .data_preprocess import *
# 导入项目配置类 ProjectConfig，用于统一管理文件路径、超参数等
from pet_config import *

pc = ProjectConfig()  # 实例化项目配置文件
# 根据配置中的模型路径，加载对应的分词器。这个分词器与模型必须匹配，用于将文本转换为 ID
tokenizer = AutoTokenizer.from_pretrained(pc.pre_model)

def get_data():
    # 1.加载模板与原始数据
    # 获取prompt.txt
    prompt = open(pc.prompt_file, 'r', encoding='utf8').readlines()[0].strip()  # prompt定义
    print(f'prompt--》{prompt}') # prompt--》这是一条{MASK}评论：{textA}。
    hard_template = HardTemplate(prompt=prompt)  # 模板转换器定义
    # load_dataset(...): 使用 Hugging Face 的 datasets 库一次性加载训练集和验证集
    # 为什么这样写？ datasets 库非常高效，支持内存映射，能处理超大文件。
    #   通过字典 {'train': ..., 'dev': ...} 的形式传入，它会返回一个 DatasetDict 对象，
    #   内部自动区分了训练和验证数据，结构清晰
    dataset = load_dataset('text', data_files={'train': pc.train_path, 'dev': pc.dev_path})
    """
    dataset-->DatasetDict({
        train: Dataset({
            features: ['text'],
            num_rows: 63
        })
        dev: Dataset({
            features: ['text'],
            num_rows: 590
        })
    })
    """
    # print(f'dataset-->{dataset}')
    # print(f'Prompt is -> {prompt}') # Prompt is -> 这是一条{MASK}评论：{textA}。

    # 2. 应用数据处理函数
    new_func = partial(convert_example,
                       tokenizer=tokenizer,
                       hard_template=hard_template,
                       max_seq_len=pc.max_seq_len,
                       max_label_len=pc.max_label_len)
    # print("*"*80)
    # functools.partial: 这是一个非常巧妙的用法。convert_example 函数有很多参数，
    # 但 dataset.map() 只会传入数据本身。partial 的作用是“冻结”住 tokenizer、template 等固定参数，
    # 返回一个新的函数 new_func
    """
    .map(new_func, batched=True): 这是 datasets 库最强大的功能之一。
        它会将 new_func（即 convert_example）应用到数据集的每一个批次上
    
    为什么这样写？ 
        batched=True 意味着函数会一次性处理一个批次的数据（例如 1000 条），而不是逐条处理。
        这利用了底层向量化操作，速度极快，是处理大规模数据集的标准做法。处理后，dataset 中除了原始的 text 列，
        还会新增 input_ids, mask_labels 等列
    """
    dataset = dataset.map(new_func, batched=True)
    """
    dataset改变之后的-->DatasetDict({
    train: Dataset({
        features: ['text', 'input_ids', 'token_type_ids', 'attention_mask', 
            'mask_positions', 'mask_labels'],
            num_rows: 63
        })
    dev: Dataset({
            features: ['text', 'input_ids', 'token_type_ids', 'attention_mask', 
            'mask_positions', 'mask_labels'],
            num_rows: 590
        })
    })
    """
    # print(f'dataset改变之后的-->{dataset}')

    # 3. 拆分数据集
    train_dataset = dataset["train"]
    """
    train_dataset[0]--》{'text': 
        '电脑\t(1)这款笔记本外观感觉挺漂亮的，分量吗，对我来说不算沉。 
        (2)安装了WindowsXP系统后，运行的速度挺快。发热量没有想象中那么大。可能尚未运行
        很耗资源的程序，没有感到内存的弊病。不过，1G的内存确实有点小。 
        (3)附赠的包很不错，挺有手感的。但是附赠的鼠标实在是太小了，幸好同时订了一个双飞
        燕的鼠标哟。', 'input_ids': [101, 6821, 3221, 671, 3340, 103, 103, 6397, 6389, 
        8038, 113, 122, 114, 6821, 3621, 5011, 6381, 3315, 1912, 6225, 2697, 6230, 
        2923, 4023, 778, 4638, 8024, 1146, 7030, 1408, 8024, 2190, 2769, 3341, 6432, 
        679, 5050, 3756, 511, 113, 123, 114, 2128, 6163, 749, 100, 5143, 5320, 1400, 
        8024, 6817, 6121, 4638, 6862, 2428, 2923, 2571, 511, 1355, 4178, 7030, 3766, 
        3300, 2682, 6496, 704, 6929, 720, 1920, 511, 1377, 5543, 2213, 3313, 6817, 
        6121, 2523, 5450, 6598, 3975, 4638, 4923, 2415, 8024, 3766, 3300, 2697, 1168, 
        1079, 2100, 4638, 2464, 4567, 511, 679, 6814, 8024, 100, 4638, 1079, 2100, 
        4802, 2141, 3300, 4157, 2207, 511, 113, 124, 114, 7353, 6615, 4638, 1259, 2523,
        679, 7231, 8024, 2923, 3300, 2797, 2697, 4638, 511, 852, 3221, 7353, 6615, 
        4638, 7962, 3403, 2141, 1762, 3221, 1922, 2207, 749, 8024, 2401, 1962, 1398, 
        3198, 6370, 749, 671, 702, 1352, 7607, 4242, 4638, 7962, 3403, 1518, 511, 511, 
        102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0], 'mask_positions': [5, 6], 'mask_labels': [4510, 5554]}
    """
    # print(f'train_dataset[0]--》{train_dataset[0]}')
    dev_dataset = dataset["dev"]
    """
    dev_dataset--》Dataset({
    features: ['text', 'input_ids', 'token_type_ids', 'attention_mask', 
        'mask_positions', 'mask_labels'],
        num_rows: 590
    })
    """
    # print(f'dev_dataset--》{dev_dataset}')
    """
    dev_dataset
    {
        'text': 
    '水果\t买回来才知道潘苹果原来是这个意思，没想到与金融大鳄也有个亲密接触。包装不
    错，苹果挺甜的，个头也大。',
    'input_ids': [
        101,
        6821,
        ,...
        102,
        0,
        0,
        0,
        0,
        ...
    ],
    'token_type_ids': [
        0,
        1,...
    ],
    'attention_mask': [
        1,
        1,
        0,...
    ],
        'mask_positions': [5, 6],
        'mask_labels': [3717, 3362]
    }
    """
    # print('dev_dataset', dev_dataset[0])
    # print('*'*80)
    # default_data_collator作用，转换为tensor数据类型

    # 4. 创建 DataLoader: 为训练集创建数据加载器
    """
    shuffle=True: 至关重要。在每个训练周期（epoch）开始时打乱数据顺序，防止模型记住数据的顺序，提高泛化能力。
c   collate_fn=default_data_collator: 指定如何将一个批次的样本组合成一个 batch tensor。default_data_collator 会自动将列表转换为张量，并对 input_ids 等进行 padding。
b   batch_size=pc.batch_size: 从配置中读取批次大小
    """
    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  collate_fn=default_data_collator,
                                  batch_size=pc.batch_size)
    # 注意这里没有 shuffle=True, 因为验证时不需要打乱顺序
    dev_dataloader = DataLoader(dev_dataset,
                                collate_fn=default_data_collator,
                                batch_size=pc.batch_size)
    return train_dataloader, dev_dataloader


if __name__ == '__main__':
    train_dataloader, dev_dataloader = get_data()
    print(len(train_dataloader))  # 8
    print(len(dev_dataloader)) # 74
    for i, value in enumerate(train_dataloader):
        print("i-->", i)
        """
        value-->
        {
            'input_ids': tensor([[ 101, 6821, 3221,  ...,    0,    0,    0],
                [ 101, 6821, 3221,  ...,    0,    0,    0],
                [ 101, 6821, 3221,  ...,    0,    0,    0],
                ...,
                [ 101, 6821, 3221,  ...,    0,    0,    0],
                [ 101, 6821, 3221,  ...,    0,    0,    0],
                [ 101, 6821, 3221,  ...,    0,    0,    0]]),
            'token_type_ids': tensor([[0, 0, 0,  ..., 0, 0, 0],
                [0, 0, 0,  ..., 0, 0, 0],
                [0, 0, 0,  ..., 0, 0, 0],
                ...,
                [0, 0, 0,  ..., 0, 0, 0],
                [0, 0, 0,  ..., 0, 0, 0],
                [0, 0, 0,  ..., 0, 0, 0]]),
            'attention_mask': tensor([[1, 1, 1,  ..., 0, 0, 0],
                [1, 1, 1,  ..., 0, 0, 0],
                [1, 1, 1,  ..., 0, 0, 0],
                ...,
                [1, 1, 1,  ..., 0, 0, 0],
                [1, 1, 1,  ..., 0, 0, 0],
                [1, 1, 1,  ..., 0, 0, 0]]),
            'mask_positions': tensor([[5, 6],
                [5, 6],
                [5, 6],
                [5, 6],
                [5, 6],
                [5, 6],
                [5, 6],
                [5, 6]]),
            'mask_labels': tensor([[6132, 3302],
                [3717, 3362],
                [6983, 2421],
                [6983, 2421],
                [6132, 3302],
                [2398, 3352],
                [6983, 2421],
                [2398, 3352]])
        }
        """
        print("value-->",value)
        # value['input_ids'].dtype--> torch.int64
        print("value['input_ids'].dtype-->",value['input_ids'].dtype)
        break
