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

        # 计算注意力分数（原始权重）[batch_size, n_heads, len_q, len_k]，transpose(-1, -2)是将K的-1维和-2维交换
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


# 定义多头注意力类
d_embedding = 512 # Embedding维度
n_heads = 8 # Multi-head Attention 中头的个数
batch_size = 3 # 每一批的数据大小
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(self).__init__()
        self.W_Q = nn.Linear(d_embedding, d_k * n_heads)     # Q的线性变换层
        self.W_K = nn.Linear(d_embedding, d_k * n_heads)     # K的线性变换层
        self.W_V = nn.Linear(d_embedding, d_v * n_heads)     # V的线性变换层
        self.linear = nn.Linear(n_heads * d_v, d_embedding)  # V的线性变换层
        self.layer_normal = nn.LayerNorm(d_embedding)
    
    def forward(self, Q, K, V, attn_mask):
        #-----维度信息-----
        # Q K V [batch_size, len_q/k/v, embedding_dim]
        # -----------------
        residual, batch_size = Q, Q.size(0) # 保留残差连接
        # 将输入进行线性变换和重塑，以便后续处理
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        #-----维度信息-----
        # q_s k_s v_s: [batch_size, n_heads, len_q/k/v, d_q=k/v]
        # -----------------
        
        # 将注意力掩码复制到多头attn_mask：[batch_size, n_heads, len_q, len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        #-----维度信息-----
        # attn_mask [batch_size, n_heads, len_q, len_k]
        # -----------------

        # 使用缩放点积注意力计算上下文和注意力权重，先创建实例ScaleDotProductAttention()，再调用参数
        context, weights = ScaleDotProductAttention()(q_s, k_s, v_s, attn_mask)
        #-----维度信息-----
        # context [batch_size, n_heads, len_q, dim_v]
        # weights [batch_size, n_heads, len_q, len_k]
        # -----------------

        # 通过调整维度将多个头的上下文向量连接在一起
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        #-----维度信息-----
        # context [batch_size, len_q, n_heads * dim_v]
        # -----------------

        # 用一个线性层把连接后的多头自注意力结果转换，原始地嵌入维度
        output = self.linear(context)
        #-----维度信息-----
        # output [batch_size, len_q, embedding_dim]
        # -----------------

        # 与输入（Q）进行残差连接，并进行层归一化后输出
        output = self.layer_normal(output + residual)
        #-----维度信息-----
        # output [batch_size, len_q, embedding_dim]
        # -----------------

        return output, weights # 返回层归一化的输出和注意力权重

    

print ("test")