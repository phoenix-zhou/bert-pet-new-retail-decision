# 导入必备工具包

import torch
from rich import print


def mlm_loss(logits, mask_positions, sub_mask_labels, cross_entropy_criterion, device):
    """
    计算指定位置的mask token的output与label之间的cross entropy loss。
    自定义的掩码语言模型（MLM）损失计算函数。与标准的 BERT MLM 损失（一个 mask 位置对应一个真实 label）不同，
    这段代码处理的是一对多（1-to-N）的子标签（sub_labels）场景，即一个 mask 位置可能对应多个正确的子标签
    Args:
        logits (torch.tensor): 模型原始输出 -> (batch, seq_len, vocab_size)
        mask_positions (torch.tensor): mask token的位置  -> (batch, mask_label_num)
        sub_mask_labels (list): mask token的sub label(包含每个样本的 mask 位置对应的真实子标签), 由于每个label的sub_label数目不同，所以这里是个变长的list,
                                    e.g. -> [
                                        [[2398, 3352]],
                                        [[2398, 3352], [3819, 3861]]
                                    ]
        cross_entropy_criterion (CrossEntropyLoss): CE Loss计算器
        device (str): cpu还是gpu

    Returns:
        torch.tensor: CE Loss
    """
    batch_size, seq_len, vocab_size = logits.size()

    # 型预测结果logits11-->torch.Size([8, 256, 21128])
    print(f'模型预测结果logits11-->{logits.size()}')
    # mask_positions-->torch.Size([8, 2])
    print(f'mask_positions-->{mask_positions.shape}')
    print("*"*80)

    """
    子标签ids: 
        sub_mask_labels-->[[[6983, 2421]], [[4510, 5554]], [[5885, 4281]], [[6132,3302]], [[2398, 3352]], [[3717, 3362]], [[6983, 2421]], [[3717, 3362]]]
        是一个三维的, 里面的每一个表示主标签对应的子标签列表
    """
    # print(f'sub_mask_labels-->{sub_mask_labels}')
    # print("*" * 80)


    """
    使用 zip 将当前 batch 中的 logits、子标签、mask 位置打包，逐个样本进行循环。
    为什么这样做：
        核心原因：sub_mask_labels 是一个变长列表（每个样本的 mask 数量不同，每个 mask 对应的子标签数量也不同）。PyTorch 的张量操作无法直接处理这种不规则的嵌套结构，因此必须退回到 Python 层面的 for 循环，逐样本处理。
        loss = None 用于初始化累加器，后续在循环中逐步累加每个样本的 loss
    """
    loss = None
    # for single_logits, single_sub_mask_labels, single_mask_positions in zip(logits, sub_mask_labels, mask_positions):
    # 从底层可知: zip 里面传的值是可迭代对象
    for single_value in zip(logits, sub_mask_labels, mask_positions):  # single_value是一个元组
        single_logits = single_value[0]
        # single_logits-->torch.Size([256, 21128])
        # print(f'single_logits-->{single_logits.shape}')

        single_sub_mask_labels = single_value[1] # 是一个列表
        # single_sub_mask_labels-->[[6983, 2421]]
        # print(f'single_sub_mask_labels-->{single_sub_mask_labels}')

        single_mask_positions = single_value[2]
        # single_mask_positions-->tensor([5, 6])
        # print(f'single_mask_positions-->{single_mask_positions}')
        # print("*"*80)

        # todo:single_logits-->形状[512, 21128],
        # todo:single_mask_positions--》形状size[2]-->具体值([5, 6])
        # 利用高级索引（Advanced Indexing），从整个序列的 logits 中，只提取出 [MASK] 所在位置的预测向量
        # 为什么这样做：MLM 任务只计算被 mask 掉的 token 的损失，忽略其他非 mask 位置的 token。
        # 这一步将形状从 (seq_len, vocab_size) 降维到 (mask_label_num, vocab_size)，大幅减少后续计算量
        single_mask_logits = single_logits[single_mask_positions]  # (mask_label_num, vocab_size)
        # single_mask_logits4-->torch.Size([2, 21128])
        # print(f'single_mask_logits4-->{single_mask_logits.shape}')


        """
        为什么这样做：
            这是解决一对多标签对齐的关键。假设当前样本有 2 个 mask 位置，且有 1 组子标签（即 len=1）。repeat(1, 1, 1) 后形状不变，reshape 后为 (2, vocab_size)。
            如果有 2 组子标签，repeat(2, 1, 1) 会将这 2 个 mask 的 logits 复制一份，变成 4 个 mask 的 logits，reshape 后为 (4, vocab_size)。
            目的：确保预测结果（logits）的数量与真实标签（labels）的数量严格一致，以便传入 CrossEntropyLoss
        """
        # single_mask_logits-->水果-》【苹果；香蕉；橘子】
        # repeat重复的倍数: 目的是解决一个主标签对应多个子标签情况, 需要对齐
        single_mask_logits = single_mask_logits.repeat(len(single_sub_mask_labels), 1, 1)  # (sub_label_num, mask_label_num, vocab_size)
        # single_mask_logits5:torch.Size([1, 2, 21128])
        # print(f'single_mask_logits5:{single_mask_logits.shape}')

        # 预测结果: reshape(-1, vocab_size)：将三维张量展平为二维张量
        single_mask_logits = single_mask_logits.reshape(-1, vocab_size)  # (sub_label_num * mask_label_num, vocab_size)
        # 模型预测的结果：single_mask_logits6:torch.Size([2, 21128])
        # print(f'模型预测的结果：single_mask_logits6:{single_mask_logits.shape}')


        """
        将 Python List 转换为 PyTorch 的 LongTensor，并移动到指定设备。
        先 reshape(-1, 1) 再 squeeze()，将其强制展平为一维张量。
        为什么这样做：
            类型要求：PyTorch 的 CrossEntropyLoss 要求 target（标签）必须是 LongTensor（整型），且与 logits 在同一设备上。
            形状要求：CrossEntropyLoss 要求 target 的形状必须是 (N,)，即一维。
                由于前面 sub_mask_labels 可能是 [[6983, 2421]] 这样的二维嵌套列表，
                reshape(-1, 1).squeeze() 是一种将其强制拉平为一维 [6983, 2421] 的写法。
                （注：其实直接 .reshape(-1) 或 .view(-1) 会更简洁且不易出错）
        """
        # 把single_sub_mask_labels 通过 torch.LongTensor转换成张量,才能进行计算, single_sub_mask_labels 的形状为[1, 2]
        single_sub_mask_labels = torch.LongTensor(single_sub_mask_labels).to(device)  # (sub_label_num, mask_label_num)
        # single_sub_mask_labels 通过reshape(-1, 1) 把形状从[1, 2] 转换成[2, 1], 然后通过squeeze()降维成[1 * 2]
        single_sub_mask_labels = single_sub_mask_labels.reshape(-1, 1).squeeze()  # (sub_label_num * mask_label_num)

        # 真实子标签mask值：single_sub_mask_labels7-->torch.Size([2])
        # print(f'真实子标签mask值：single_sub_mask_labels7-->{single_sub_mask_labels.shape}')
        # 真实子标签mask值：single_sub_mask_labels7-->tensor([6132, 3302])
        # print(f'真实子标签mask值：single_sub_mask_labels7-->{single_sub_mask_labels}')

        cur_loss = cross_entropy_criterion(single_mask_logits, single_sub_mask_labels) # 当前的loss = cross_entropy_criterion(预测结果-二维, 真实结果-一维)
        cur_loss = cur_loss / len(single_sub_mask_labels) # 平均损失 = 当前loss / 子标签个数

        if not loss:
            loss = cur_loss
        else:
            loss += cur_loss

    loss = loss / batch_size  # (1,) 批次的平均损失 = loss/batch_size
    return loss


def convert_logits_to_ids(
        logits: torch.tensor,
        mask_positions: torch.tensor):
    """
    代码实现了一个推理（Inference）阶段的核心函数：将模型输出的概率分布（logits）转换为具体的预测词表 ID（Token IDs）

    输入Language Model的词表概率分布（LMModel的logits），将mask_position位置的
    token logits转换为token的id。

    Args:
        logits (torch.tensor): model output -> (batch, seq_len, vocab_size)[8, 512, 21128] 是模型输出的原始预测分布
        mask_positions (torch.tensor): mask token的位置 -> (batch, mask_label_num)[8,2] 记录了每个样本中需要提取预测结果的 mask 位置

    Returns:
        torch.LongTensor: 对应mask position上最大概率的推理token -> (batch, mask_label_num)[8, 2]
    """
    label_length = mask_positions.size()[1]  # 标签长度: 获取(batch, mask_label_num)[8,2]中的mask_label_num => 2
    # label_length--》2
    print(f'label_length--》{label_length}')

    batch_size, seq_len, vocab_size = logits.size()

    mask_positions_after_reshaped = []
    # mask_positions.detach().cpu().numpy().tolist()-->[[5, 6], [5, 6]]
    print(f'mask_positions.detach().cpu().numpy().tolist()-->{mask_positions.detach().cpu().numpy().tolist()}')

    """
    解析：将二维的相对坐标 (batch_idx, seq_pos) 转换为全局一维绝对坐标。
    为什么这样做：
        降维打击：PyTorch 的二维高级索引（如 logits[batch_indices, seq_indices]）在底层实现上比较慢。
                为了追求极致性能，代码选择将 logits 展平成一维序列，然后用一维索引去取值。
        公式 batch * seq_len + pos：这是二维矩阵展平为一维的标准数学公式。
                例如，第 1 个 batch 的第 5 个位置，在全局一维数组中的索引就是 1 * seq_len + 5。
        .detach().cpu().numpy().tolist()：因为 Python 的 for 循环无法直接遍历 GPU 上的 Tensor，
                所以必须先将其脱离计算图（detach）、移到 CPU、转为 NumPy 数组，最后转为 Python 原生 List 才能进行遍历
    """
    for batch, mask_pos in enumerate(mask_positions.detach().cpu().numpy().tolist()): # 把张量转换成列表
        for pos in mask_pos:
            mask_positions_after_reshaped.append(batch * seq_len + pos)
    # mask_positions_after_reshaped -->[5, 6, 25, 26]
    # print(f'mask_positions_after_reshaped-->{mask_positions_after_reshaped}')
    # 原始的logits-->torch.Size([2, 20, 21128])
    # print(f'原始的logits-->{logits.shape}')

    """
        将 logits 从 (batch, seq_len, vocab_size) 展平为 (batch * seq_len, vocab_size)。
        使用上一步计算出的一维索引列表，直接从展平后的 logits 中提取目标位置的向量。
        为什么这样做：
            通过 reshape 将三维张量变为二维，配合一维索引，一次性并行提取了所有 batch 中所有 mask 位置的 logits。
            这种写法完全避免了在 batch 维度上进行 Python 层面的 for 循环，极大提升了推理时的并行计算效率
    """
    logits = logits.reshape(batch_size * seq_len, -1)  # (batch_size * seq_len, vocab_size)
    # 改变原始模型输出的结果形状: torch.Size([40, 21128])
    # print('改变原始模型输出的结果形状', logits.shape)

    mask_logits = logits[mask_positions_after_reshaped]  # (batch * label_num, vocab_size)
    # 选择真实掩码位置预测的数据形状:  torch.Size([4, 21128])
    # print('选择真实掩码位置预测的数据形状',mask_logits.shape)

    """
    在词表维度（dim=-1）上寻找最大值，返回其对应的索引（即 Token ID）。
    为什么这样做：
        在推理阶段（Greedy Search / 贪心解码），我们通常直接选择概率最大的那个词作为预测结果。
        argmax 会将形状从 (batch * label_num, vocab_size) 降维为 (batch * label_num)，每个元素就是一个预测出的 token id
    """
    predict_tokens = mask_logits.argmax(dim=-1)  # (batch * label_num)
    # 求出每个样本真实mask位置预测的tokens tensor([15489,  7815,  1002, 15636])
    print('求出每个样本真实mask位置预测的tokens', predict_tokens)

    """
    将一维的预测结果重新 reshape 回 (batch_size, label_length)，并返回。
    为什么这样做：
        为了保持输入和输出的形状一致性。输入是 (batch, mask_label_num) 的 mask 位置，输出的预测 token 也应该是同样的形状，方便后续业务逻辑（如拼接回原句子、计算准确率等）直接使用
    """
    predict_tokens = predict_tokens.reshape(-1, label_length)  # (batch, label_num)
    """
    predict_tokens--》tensor([[15489,  7815],
                [ 1002, 15636]])
        tensor([[15489,  7815],
                [ 1002, 15636]])
            """
    print(f'predict_tokens--》{predict_tokens}')

    return predict_tokens


if __name__ == '__main__':
    logits = torch.randn(2, 20, 21128)
    mask_positions = torch.LongTensor([
        [5, 6],
        [5, 6],
    ])
    predict_tokens = convert_logits_to_ids(logits, mask_positions)
    print(predict_tokens)
