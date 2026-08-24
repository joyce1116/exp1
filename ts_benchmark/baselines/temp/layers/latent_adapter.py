# 说明本模块为 VisionTS patch token 提供全局修正 latent adapter。
"""Global-correction latent adapter for VisionTS patch tokens."""

# 导入 PyTorch tensor 与参数初始化功能。
import torch
# 导入神经网络层模块。
from torch import nn


# 定义通过单个 latent query 注入全局修正表示的轻量 adapter。
class VariableAwareLatentAdapter(nn.Module):
    # 说明该类以单个 latent query 构成轻量的全局信息瓶颈。
    """Use one latent query as a lightweight global-correction bottleneck."""

    # 创建 projection、双向 cross-attention 和 LayerNorm 等子层。
    def __init__(
        self,
        embed_dim=768,  # 输入 patch token 的 embedding 维度 E。
        num_latents=1,  # 全局修正 latent query 的数量，默认为 1。
        latent_dim=192,  # attention 内部 latent space 的维度 D。
        num_heads=4,  # multi-head cross-attention 的 head 数量。
    ):
        # 初始化 nn.Module 的内部状态。
        super().__init__()

        # 记录 latent query 数量，供外部检查模型结构。
        self.num_latents = num_latents
        # 在降维前对输入 patch embedding 执行 LayerNorm。
        self.patch_norm = nn.LayerNorm(embed_dim)
        # 将 patch embedding 从 E 维 projection 到 D 维 latent space。
        self.patch_down = nn.Linear(embed_dim, latent_dim)

        # 创建可学习且由所有样本共享的全局修正 query，默认 shape 为 [1, 1, D]。
        self.latent_queries = nn.Parameter(
            torch.zeros(1, num_latents, latent_dim)
        )
        # 在 latent query 读取 patch token 前执行 LayerNorm。
        self.latent_query_norm = nn.LayerNorm(latent_dim)
        # 在 patch token 作为 attention memory 前执行 LayerNorm。
        self.patch_memory_norm = nn.LayerNorm(latent_dim)
        # 创建“latent query 读取 patch token”的 multi-head cross-attention。
        self.latent_cross_attention = nn.MultiheadAttention(
            latent_dim, num_heads, batch_first=True
        )

        # 在 patch query 读取 latent representation 前执行 LayerNorm。
        self.patch_query_norm = nn.LayerNorm(latent_dim)
        # 在 latent representation 作为 attention memory 前执行 LayerNorm。
        self.latent_memory_norm = nn.LayerNorm(latent_dim)
        # 创建“patch query 读取 latent representation”的 multi-head cross-attention。
        self.patch_cross_attention = nn.MultiheadAttention(
            latent_dim, num_heads, batch_first=True
        )

        # 将 attention update 从 D 维 latent space projection 回 E 维 embedding。
        self.patch_up = nn.Linear(latent_dim, embed_dim)

        # 用截断正态分布初始化可学习 latent query。
        nn.init.trunc_normal_(self.latent_queries, std=0.02)
        # 将 output projection 参数置零，使 adapter 初始为恒等 residual 分支。
        nn.init.zeros_(self.patch_up.weight)
        nn.init.zeros_(self.patch_up.bias)

    # 聚合所有变量的 patch token，并为每个 patch 注入同一个全局修正表示。
    # shape 约定：B 为 batch size，V 为变量数，P 为可见 patch 数，E 为 embedding 维度。
    # 输入 patch_tokens 与输出 tensor 的 shape 均为 [B, V, P, E]。
    def forward(self, patch_tokens):
        # 从输入 shape [B, V, P, E] 中读取各维度大小。
        batch_size, num_variables, num_patches, embed_dim = patch_tokens.shape
        # 合并 variable 与 patch 两个轴，得到 [B, V*P, E] 并保留作 residual。
        original = patch_tokens.reshape(batch_size, -1, embed_dim)
        # 对 patch token 做 LayerNorm 和降维 projection，得到 [B, V*P, D]。
        patches = self.patch_down(self.patch_norm(original))

        # 为 batch 中每个样本扩展单个 latent query，默认得到 [B, 1, D]。
        latents = self.latent_queries.expand(batch_size, -1, -1)
        # 归一化 patch token，作为第一阶段 cross-attention 的 key 和 value。
        patch_memory = self.patch_memory_norm(patches)
        # 以单个 latent query [B, 1, D] 聚合 patch memory [B, V*P, D]。
        latent_update = self.latent_cross_attention(
            self.latent_query_norm(latents),
            patch_memory,
            patch_memory,
            need_weights=False,
        )[0]
        # 通过 residual connection 写回 attention update，默认 shape 保持 [B, 1, D]。
        latents = latents + latent_update

        # 归一化 latent representation，作为第二阶段 attention 的 key 和 value。
        latent_memory = self.latent_memory_norm(latents)
        # 让每个 patch query [B, V*P, D] 读取同一个全局 latent memory [B, 1, D]。
        patch_update = self.patch_cross_attention(
            self.patch_query_norm(patches),
            latent_memory,
            latent_memory,
            need_weights=False,
        )[0]
        # 将 patch update 从 [B, V*P, D] projection 为 [B, V*P, E]。
        patch_update = self.patch_up(patch_update)

        # 通过 residual connection 将全局修正表示加回原始 patch embedding。
        output = original + patch_update
        # 恢复四维布局，并返回 shape 为 [B, V, P, E] 的 tensor。
        return output.reshape(
            batch_size, num_variables, num_patches, embed_dim
        )
