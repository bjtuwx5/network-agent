import numpy as np # 导入numpy库
import torch # 导入torch库
import torch.nn as nn # 导入torch.nn库

d_k = 64 # K（=Q）维度
d_v = 64 # V维度

# 定义缩放点积注意力类
class ScaleDotProductAttention(nn.Module): # 继承nn.Module类
    def __init__(self):
        super().__init__()  #调用父类nnModule的构造方法
    
    # 前向传播算法，注意力计算
    def forward(self, Q, K, V, attn_mask):
        #-----维度信息-----
        # Q K V [batch_size, n_heads, len_q/k/v, dim q=k/v] (dim_q=dim_k)
        # attn_mask[batch_size, n_heads, len_q, len_k]

        # 计算注意力分数（原始权重）[batch_size, n_heads, len_q, len_k]，transpose是转置
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        #-----维度信息-----
        # scores[batch_size, n_heads, len_q, len_k]

        #-----------------
        # 使用注意力掩码，将attn_mask中值为1的位置权重替换为极小值
        # -----维度信息-----
        # attn_mask[batch_size, n_heads, len_q, len_k]，形状和scores相同
        # -----------------
        # attn_mask中值为TRUE的位置表示需要屏蔽，将这些位置设置成-1e9（负无穷，-1×10⁹），后续经过softmax，这些值会变为0，不参与计算
        scores.masked_fill(attn_mask, -1e9)  

        # 用softmax对注意力分数进行归一化
        weights = nn.Softmax(dim=-1)(scores)
        # -----维度信息-----
        # weights[batch_size, n_heads, len_q, len_k]，形状和scores相同

        # -----------------
        # 计算上下文向量(也就是注意力的输出)，是上下文信息的紧凑表示
        context = torch.matmul(weights, V)
        # -----维度信息-----
        # context[batch_size, n_heads, len_q, len_k]
        # -----------------

        return context, weights # 返回上下文向量和注意力分数


print ("test")