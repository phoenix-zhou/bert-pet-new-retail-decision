# # # # from rich import print
# # def get_common_sub_str(str1: str, str2: str):
# #     """
# #     寻找最大公共子串：两个字符串中同时出现的最长的子串（连续）
# #     str1:abcd
# #     str2:abadbcdba
# #     """
# #     # 生成0矩阵，为方便后续计算，比字符串长度多了一列
# #     # record[i,j]表示公共子串的最大长度，该公共子串以str1的第i个字符结尾并且以str2的第j个字符结尾
# #     # record[i, j] = record[i-1,j-1] + 1
# #     # 完成所有record的计算后，选择最大的record值，即为两个字符串str1与str2的最长公共子串长度；
# #     # 往前回溯，即可得到最长公共子串
# #     lstr1, lstr2 = len(str1), len(str2)
# #     print(f'lstr1--》{lstr1}')
# #     print(f'lstr2--》{lstr2}')
# #
# #     record = [[0 for i in range(lstr2 + 1)] for _ in range(lstr1 + 1)]
# #     print(f'record-->{record}')
# #     end_index = 0  # 最长匹配对应在str1中的最后一位
# #     max_length = 0  # 最长匹配长度
# #
# #     for i in range(1, lstr1+1):
# #         for j in range(1, lstr2+1):
# #             if str1[i-1] == str2[j-1]: # 必定选择这两个字符作为最长公共子序列的结尾
# #                 record[i][j] = record[i-1][j-1] + 1
# #                 if record[i][j] > max_length:
# #                     max_length = record[i][j]
# #                     end_index = i
# #
# #     return str1[end_index - max_length:end_index], max_length
# #
# # str1 = "abcd"
# # str2 = "abadbcdba"
# # a = get_common_sub_str(str1=str1, str2=str2)
# # # print(a)
# # # # #
# # # # # # dict1 = {"word":'你好'}
# # # # # # for value in dict1.items():
# # # # # #     print(value)
# # # # # # import torch
# # # # # # a = [[[1, 2]], [[2, 3]]] # 2-1-2
# # # # # # a = torch.randn(2,1,2)
# # # # # # b = [[3], [4]] # 2-1
# # # # # # # c = [[[5]], [[6]]] # 2-1
# # # # # # c = torch.randn(2,1,1)
# # # # # # for value in zip(a, b, c):
# # # # # #     print(value)
# # # # #
# # # # # # list1 = [1, 2, 0, 0]
# # # # # # # list1.remove(0)
# # # # # # # print(list1)
# # # # # # max_len = 6
# # # # # # a = list1[:max_len]
# # # # # # if len(a) < max_len :
# # # # # #     b = a + [0] * (max_len - len(a))
# # # # # #     print(b)
# # # #
# # # # def get_common_str(s1, s2):
# # # #     # 1.获取长度
# # # #     s1_len = len(s1)
# # # #     s2_len = len(s2)
# # # #     # 2.初始化二维数组
# # # #     dp = [[0]*(s2_len+1) for _ in range(s1_len+1)]
# # # #     # 3.初始化值
# # # #     max_len = 0
# # # #     end_index = 0
# # # #
# # # #     # 4.进入循环
# # # #     for i in range(1, s1_len+1):
# # # #         for j in range(1, s2_len+1):
# # # #             if s1[i-1] == s2[j-1]:
# # # #                 dp[i][j] = dp[i-1][j-1] + 1
# # # #                 if dp[i][j] > max_len:
# # # #                     max_len = dp[i][j]
# # # #                     end_index = i
# # # #
# # # #     return s1[end_index-max_len: end_index]
# # # #
# # # #
# # # # str1 = "abcdba"
# # # # str2 = "abadbcdba"
# # # # a = get_common_str(str2, str1)
# # # # print(a)
# # # best_f1 = 4.564542
# # # f1 = 5.4599332
# # # print(
# # #     f"best F1 performence has been updated: {best_f1:.5f} --> {f1:.5f}"
# # # )
# # #
# # # print(
# # #     f"best F1 performence has been updated: {best_f1} --> {f1}"
# # # )
# #
# from pprint import pprint
# def get_common_sub_str(str1: str, str2: str):
#     """
#     寻找最大公共子串：两个字符串中同时出现的最长的子串（连续）
#     str1:abcd
#     str2:abadbcdba
#     """
#     # 动态规划：数组存储一个状态，record[i,j]表示公共子串的最大长度；该公共子串以str1的第i个字符结尾并且以str2的第j个字符结尾
#     # 生成0矩阵，为方便后续计算，比字符串长度多了一列
#     # record[i, j] = record[i-1,j-1] + 1 # record[i][j] = record[i-1][j-1] + 1
#     # record[i=3, j=6] = 3
#     # record[i=3-1=2, j=6-1=5] =  2
#     # 完成所有record的计算后，选择最大的record值，即为两个字符串str1与str2的最长公共子串长度；
#     # 往前回溯，即可得到最长公共子串
#     len_str1 = len(str1)
#     len_str2 = len(str2)
#     # 定义一个数组，每一个元素代表最大公共子串的长度
#     record = [[0 for _ in range(len_str2 + 1)] for _ in range(len_str1 + 1)]
#     max_len = 0 # 最大长度
#     end_idx = 0
#     for i in range(1, len_str1+1):
#         for j in range(1, len_str2+1):
#             if str1[i-1] == str2[j-1]:
#                 record[i][j] = record[i-1][j-1] + 1
#                 if max_len < record[i][j]:
#                     max_len = record[i][j]
#                     end_idx = i
#
#     return str1[end_idx-max_len: end_idx], max_len
#
#
# str1= 'abcd'
# str2 = 'abadbcdba'
# print(get_common_sub_str(str1, str2))
# import torch
# a = torch.randn(4, 2, 3)
# print(a.numpy())

import numpy as np

# b = np.array([1,2, 3])
# print(torch.from_numpy(b))
# b = torch.randn(3, 2)
# c = [[[1]], [[2]], [[3]], [[4]]]
# for i in zip(a, b, c):
#     print(i)

print(f"111: {2.5666745645:.4f} --> {4.77576:.4f}"
                    )