import os
import time
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_scheduler
from utils.metirc_utils import ClassEvaluator
from utils.common_utils import *
from data_handle.data_loader import *
from utils.verbalizer import Verbalizer
from pet_config import *
from tqdm import tqdm
pc = ProjectConfig()


def evaluate_model(model, metric, data_loader, tokenizer, verbalizer):
    """
    在测试集上评估当前模型的训练效果。

    Args:
        model: 当前模型
        metric: 评估指标类(metric)
        data_loader: 测试集的dataloader
        global_step: 当前训练步数
    """
    # 只要是预训练模型,都要进行model.eval()
    model.eval()
    # 调用utils/metric_utils.py中的类 ClassEvaluator 的方法reset(),重置重置积累的数值
    metric.reset()

    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader)): # 对数据集进行操作
            """
            batch--》{'input_ids': tensor([[ 101, 6821, 3221,  ...,    0,    0,    0],
                [ 101, 6821, 3221,  ...,    0,    0,    0],
                [ 101, 6821, 3221,  ...,    0,    0,    0],
                ...,
                [ 101, 6821, 3221,  ...,    0,    0,    0],
                [ 101, 6821, 3221,  ...,    0,    0,    0],
                [ 101, 6821, 3221,  ...,    0,    0,    0]]), 'token_type_ids': 
        tensor([[0, 0, 0,  ..., 0, 0, 0],
                [0, 0, 0,  ..., 0, 0, 0],
                [0, 0, 0,  ..., 0, 0, 0],
                ...,
                [0, 0, 0,  ..., 0, 0, 0],
                [0, 0, 0,  ..., 0, 0, 0],
                [0, 0, 0,  ..., 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1,  ..., 
        0, 0, 0],
                [1, 1, 1,  ..., 0, 0, 0],
                [1, 1, 1,  ..., 0, 0, 0],
                ...,
                [1, 1, 1,  ..., 0, 0, 0],
                [1, 1, 1,  ..., 0, 0, 0],
                [1, 1, 1,  ..., 0, 0, 0]]), 'mask_positions': tensor([[5, 6],
                [5, 6],
                [5, 6],
                [5, 6],
                [5, 6],
                [5, 6],
                [5, 6],
                [5, 6]]), 'mask_labels': tensor([[2398, 3352],
                [3717, 3362],
                [3819, 3861],
                [6983, 2421],
                [6983, 2421],
                [6132, 3302],
                [6983, 2421],
                [2797, 3322]])}
            """
            # print(f'batch--》{batch}')

            logits = model(input_ids=batch['input_ids'].to(pc.device),
                           token_type_ids=batch['token_type_ids'].to(pc.device),
                           attention_mask=batch['attention_mask'].to(pc.device)).logits
            # 验证集模型预测的结果————》torch.Size([8, 256, 21128])
            # print(f'验证集模型预测的结果————》{logits.shape}')

            # batch['mask_labels']为验证集真实的标签, 把它转换成列表
            mask_labels = batch['mask_labels'].numpy().tolist()  # (batch, label_num)
            # mask_labels-0-->[[2398, 3352], [3717, 3362], [3819, 3861], [6983, 2421], [6983, 2421], [6132, 3302], [6983, 2421], [2797, 3322]]
            # print(f"mask_labels-0-->{mask_labels}")

            for i in range(len(mask_labels)):  # 去掉样本主label中的[PAD] token
                while tokenizer.pad_token_id in mask_labels[i]:
                    mask_labels[i].remove(tokenizer.pad_token_id)
            print(f'mask_labels-1-->{mask_labels}')

            # 列表生成式: 把主label id转换成文字
            mask_labels = [''.join(tokenizer.convert_ids_to_tokens(t)) for t in mask_labels]  # id转文字
            # 真实的结果主标签：mask_labels_str-->['水果', '洗浴', '平板', '洗浴', '电器', '水果', '酒店', '洗浴']
            print(f'真实的结果主标签：mask_labels_str-->{mask_labels}')

            predictions = convert_logits_to_ids(logits,batch['mask_positions']).cpu().numpy().tolist()  # (batch, label_num)
            # 模型预测的子标签的结果--》[[5381, 1351], [5381, 4638], [2398, 3352], [4500,
            # 1501], [4500, 4638], [5381, 3340], [5381, 4638], [3633, 4638]]
            # print(f'模型预测的子标签的结果--》{predictions}')

            predictions = verbalizer.batch_find_main_label(predictions)  # 找到子label属于的主label
            """
            找到模型预测的子标签对应的主标签的结果--》[{'label': '电器', 'token_ids': 
                [4510, 1690]}, {'label': '电器', 'token_ids': [4510, 1690]}, {'label': '平板', 
                'token_ids': [2398, 3352]}, {'label': '电器', 'token_ids': [4510, 1690]}, 
                {'label': '电器', 'token_ids': [4510, 1690]}, {'label': '电器', 'token_ids': 
                [4510, 1690]}, {'label': '电器', 'token_ids': [4510, 1690]}, {'label': '电器', 
                'token_ids': [4510, 1690]}]')
            """
            # print(f"找到模型预测的子标签对应的主标签的结果--》{predictions}')")

            predictions = [ele['label'] for ele in predictions]
            # 只获得预测的主标签的结果string--》['电器', '电器', '平板', '电器', '电器', '电器', '电器', '电器']')
            # print(f"只获得预测的主标签的结果string--》{predictions}')")

            # 添加每个批次验证集数据
            metric.add_batch(pred_batch=predictions, gold_batch=mask_labels)
    eval_metric = metric.compute()
    model.train()

    return eval_metric['accuracy'], eval_metric['precision'], \
           eval_metric['recall'], eval_metric['f1'], \
           eval_metric['class_metrics']


def model2train():
    # 加载预训练模型
    model = AutoModelForMaskedLM.from_pretrained(pc.pre_model)
    # print(f'预训练模型带MLM头的--》{model}')
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(pc.pre_model)
    # Verbalizer类，用于将一个Label对应到其子Label的映射
    verbalizer = Verbalizer(verbalizer_file=pc.verbalizer,
                            tokenizer=tokenizer,
                            max_label_len=pc.max_label_len)
    """
    verbalizer--》{'电脑': ['电脑'], '水果': ['水果'], '平板': ['平板'], '衣服': 
        ['衣服'], '酒店': ['酒店'], '洗浴': ['洗浴'], '书籍': ['书籍'], '蒙牛': 
        ['蒙牛'], '手机': ['手机'], '电器': ['电器']}
    """
    # print(f'verbalizer--》{verbalizer.label_dict}')

    no_decay = ["bias", "LayerNorm.weight"] # bert官网要求: 不用权重衰减的数据: bias 偏置, LayerNorm.weight:层归一化的权重

    # class 'generator'> 生成器类型(也是迭代器类型), 结果是一个张量
    # print(type(model.parameters()))

    # 权重衰减, 防止过拟合
    optimizer_grouped_parameters = [ # 一个dict, AdamW底层能够接收该类型进行计算
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": pc.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=pc.learning_rate)
    model.to(pc.device)

    # 获取训练集, 验证集迭代数据
    train_dataloader, dev_dataloader = get_data()

    # 进行学习率的预热处理
    # 根据训练轮数计算最大训练步数，以便于scheduler)(调度器)动态调整lr
    num_update_steps_per_epoch = len(train_dataloader) # 一个批次迭代多少步
    # 指定总的训练步数，它会被学习率调度器用来确定学习率的变化规律，确保学习率在整个训练过程中得以合理地调节
    max_train_steps = pc.epochs * num_update_steps_per_epoch
    warm_steps = int(pc.warmup_ratio * max_train_steps) # 预热阶段的训练步数
    lr_scheduler = get_scheduler( # 学习率预热
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warm_steps,
        num_training_steps=max_train_steps,
    )

    loss_list = [] # 损失
    tic_train = time.time()
    metric = ClassEvaluator() # （多）分类问题下的指标评估
    criterion = torch.nn.CrossEntropyLoss() # 损失函数: 交叉熵损失
    global_step, best_f1 = 0, 0 # global_step: 迭代步数
    print('开始训练：')

    for epoch in range(pc.epochs):
        for batch in tqdm(train_dataloader): # 一个批次一个批次拿数据进行训练
            """
            batch--》{ # 一个批次8条数据
            # input_ids: 每条数据的token id
            'input_ids': tensor([[ 101, 6821, 3221,  ...,    0,    0,    0],
                [ 101, 6821, 3221,  ...,    0,    0,    0],
                [ 101, 6821, 3221,  ...,    0,    0,    0],
                ...,
                [ 101, 6821, 3221,  ...,    0,    0,    0],
                [ 101, 6821, 3221,  ...,    0,    0,    0],
                [ 101, 6821, 3221,  ...,    0,    0,    0]]), 
            # token_type_ids: 每条数据的token type id
            'token_type_ids': tensor([[0, 0, 0,  ..., 0, 0, 0],
                [0, 0, 0,  ..., 0, 0, 0],
                [0, 0, 0,  ..., 0, 0, 0],
                ...,
                [0, 0, 0,  ..., 0, 0, 0],
                [0, 0, 0,  ..., 0, 0, 0],
                [0, 0, 0,  ..., 0, 0, 0]]), 
            # attention_mask: 每条数据的自注意力掩码
            'attention_mask': tensor([[1, 1, 1,  ..., 0, 0, 0],
                [1, 1, 1,  ..., 0, 0, 0],
                [1, 1, 1,  ..., 0, 0, 0],
                ...,
                [1, 1, 1,  ..., 0, 0, 0],
                [1, 1, 1,  ..., 0, 0, 0],
                [1, 1, 1,  ..., 0, 0, 0]]),
            # mask_positions: 掩码的位置
            'mask_positions': tensor([[5, 6],
                [5, 6],
                [5, 6],
                [5, 6],
                [5, 6],
                [5, 6],
                [5, 6],
                [5, 6]]), 
            # mask_labels: 真实主标签token id
            'mask_labels': tensor([[6132, 3302],
                [3819, 3861],
                [5885, 4281],
                [6983, 2421],
                [6983, 2421],
                [2398, 3352],
                [3819, 3861],
                [6983, 2421]])}
            """
            # print(f'batch--》{batch}')

            # 把batch中的数据送给模型
            logits = model(input_ids=batch['input_ids'].to(pc.device),
                           token_type_ids=batch['token_type_ids'].to(pc.device),
                           attention_mask=batch['attention_mask'].to(pc.device)).logits
            # logits->[8, 256, 21128] 表示 批次大小为8, 每个词的维度为256, 词表大型为21128(在vocab.txt中可以知道)
            # print(f'logits->{logits.shape}')
    #         print('*'*80)

            # 真实主标签
            mask_labels = batch['mask_labels'].numpy().tolist() # tensor张量转换成列表
            # mask_labels--》[[6132, 3302], [6983, 2421], [2398, 3352], [4510, 5554], [6983, 2421], [6983, 2421], [6983, 2421], [3717, 3362]]
            # print(f'mask_labels--》{mask_labels}')
            """
            tensor([[5, 6],
                [5, 6],
                [5, 6],
                [5, 6],
                [5, 6],
                [5, 6],
                [5, 6],
                [5, 6]])
            """
            # print("batch['mask_positions']-->", batch['mask_positions'])

            sub_labels = verbalizer.batch_find_sub_labels(mask_labels) # 通过主标签获取子标签
            """
            主标签对应的子标签列表:
            sub_labels-->[{'sub_labels': ['衣服'], 'token_ids': [[6132, 3302]]}, 
                {'sub_labels': ['酒店'], 'token_ids': [[6983, 2421]]}, {'sub_labels': ['平板'],
                'token_ids': [[2398, 3352]]}, {'sub_labels': ['电脑'], 'token_ids': [[4510, 
                5554]]}, {'sub_labels': ['酒店'], 'token_ids': [[6983, 2421]]}, {'sub_labels': 
                ['酒店'], 'token_ids': [[6983, 2421]]}, {'sub_labels': ['酒店'], 'token_ids': 
                [[6983, 2421]]}, {'sub_labels': ['水果'], 'token_ids': [[3717, 3362]]}]
            """
            # print(f'sub_labels-->{sub_labels}')

            # 获取子标签对应的ids, 是一个三维的list
            sub_labels = [ele['token_ids'] for ele in sub_labels]
            # sub_labels-->[[[6132, 3302]], [[6983, 2421]], [[2398, 3352]], [[4510, 5554]], [[6983, 2421]], [[6983, 2421]], [[6983, 2421]], [[3717, 3362]]]
            # print(f'sub_labels-->{sub_labels}')

            # 计算损失
            loss = mlm_loss(logits, # 模型最终预测出来的结果: [8, 256, 21128]
                            batch['mask_positions'].to(pc.device), # 掩码位置
                            sub_labels, # 子标签ids
                            criterion, # 损失函数
                            pc.device,
                            )
            print(f'计算损失值--》{loss}')

            optimizer.zero_grad() # 梯度清零
            loss.backward() # 前向传播
            optimizer.step() # 梯度更新
            lr_scheduler.step() # 学习率更新
            # loss_list.append(float(loss.cpu().detach()))
            loss_list.append(loss)
            # #
            global_step += 1

            # 打印训练日志
            if global_step % pc.logging_steps == 0:
                time_diff = time.time() - tic_train
                loss_avg = sum(loss_list) / len(loss_list) # 平均损失
                print("global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                      % (global_step, epoch, loss_avg, pc.logging_steps / time_diff))
                tic_train = time.time()
            # 模型验证
            if global_step % pc.valid_steps == 0:
                # 可以对比不同global_step的模型差异, 这里可以先注释, 因为每隔global_step % pc.valid_steps都要保存一下, 会占用大量空间
                # cur_save_dir = os.path.join(pc.save_dir, "model_%d" % global_step)
                # print(f'cur_save_dir--》{cur_save_dir}')
                # if not os.path.exists(cur_save_dir):
                #     os.makedirs(cur_save_dir)
                # print(f'os.path.join(cur_save_dir)-->{os.path.join(cur_save_dir)}')
                # model.save_pretrained(os.path.join(cur_save_dir))
                # # model.save_pretrained(cur_save_dir)
                # tokenizer.save_pretrained(os.path.join(cur_save_dir))

                # 得到评估指标
                acc, precision, recall, f1, class_metrics = evaluate_model(model,
                                                                           metric,
                                                                           dev_dataloader,
                                                                           tokenizer,
                                                                           verbalizer)

                print("Evaluation precision: %.5f, recall: %.5f, F1: %.5f" % (precision, recall, f1))
                if f1 > best_f1: # 这里保存最好的模型
                    print(
                        f"best F1 performence has been updated: {best_f1:.5f} --> {f1:.5f}"
                    )
                    print(f'Each Class Metrics are: {class_metrics}') # 打印每个类别指标
                    best_f1 = f1
                    cur_save_dir = os.path.join(pc.save_dir, "model_best")
                    if not os.path.exists(cur_save_dir):
                        os.makedirs(cur_save_dir)
                    model.save_pretrained(os.path.join(cur_save_dir))
                    tokenizer.save_pretrained(os.path.join(cur_save_dir))
                tic_train = time.time()

    print('训练结束')


if __name__ == '__main__':
    model2train()
