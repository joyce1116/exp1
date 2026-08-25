# 说明本模块为 VisionTS patch token 提供全局修正 latent adapter。
"""Global-correction latent adapter for VisionTS patch tokens."""

# 导入 PyTorch tensor 与参数初始化功能。
import torch
# 导入神经网络层模块。
from torch import nn
from torch.nn import functional as F


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
        # 仅在测试阶段收集 Global Token 的 softmax attention 统计。
        self.statistics_mode = "train"
        self._test_statistics_batches = []
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

    def _reset_test_statistics(self):
        self._test_statistics_batches = []

    def set_statistics_mode(self, mode):
        if mode == "test" and self.statistics_mode != "test":
            self._reset_test_statistics()
        self.statistics_mode = mode

    @staticmethod
    def _normalized_entropy(distribution):
        support_size = distribution.shape[-1]
        if support_size <= 1:
            return torch.ones(
                distribution.shape[:-1],
                dtype=distribution.dtype, device=distribution.device
            )
        return -(
            distribution * distribution.clamp_min(1e-12).log()
        ).sum(dim=-1) / torch.log(
            distribution.new_tensor(float(support_size))
        )

    def _record_test_statistics(
        self, attention_weights, num_variables, num_patches
    ):
        with torch.no_grad():
            # attention_weights: [B, H, Q, P]，Q 为 Global Token 数。
            weights = attention_weights.detach().float()
            batch_size, num_heads, num_global_tokens, source_count = (
                weights.shape
            )
            if source_count != num_patches:
                raise RuntimeError(
                    "Global Token attention source count does not match "
                    "the shared-patch layout."
                )

            # 每个公共 patch 是 C 个变量同位置 patch 的等权平均，因此将真实的
            # patch attention 按 1/C 展开为各变量的等效贡献，保留原统计格式。
            weights = weights.unsqueeze(-2).expand(
                batch_size, num_heads, num_global_tokens,
                num_variables, num_patches
            ) / num_variables

            # 保留 sample/head 维的等效分布，再严格按原统计定义聚合。
            variable_routing = weights.sum(dim=-1).mean(dim=1)
            patch_routing = weights.sum(dim=-2).mean(dim=1)
            variable_entropy = self._normalized_entropy(variable_routing)
            patch_entropy = self._normalized_entropy(patch_routing)

            temporal_profiles = weights / weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
            temporal_sum = temporal_profiles.sum(dim=(0, 1))

            normalized_profiles = temporal_profiles / temporal_profiles.square(
            ).sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)
            profile_vectors = normalized_profiles.permute(
                2, 3, 0, 1, 4
            ).reshape(num_global_tokens, num_variables, -1)
            cosine_sum = torch.matmul(
                profile_vectors, profile_vectors.transpose(-1, -2)
            )

            return {
                "variable_routing": variable_routing.cpu(),
                "patch_routing": patch_routing.cpu(),
                "variable_entropy": variable_entropy.cpu(),
                "patch_entropy": patch_entropy.cpu(),
                "temporal_sum": temporal_sum.cpu(),
                "cosine_sum": cosine_sum.cpu(),
                "profile_count": batch_size * num_heads,
            }

    def test_statistics(self):
        if not self._test_statistics_batches:
            return None

        batches = self._test_statistics_batches
        temporal_sum = torch.zeros_like(
            batches[0]["temporal_sum"], dtype=torch.float64
        )
        cosine_sum = torch.zeros_like(
            batches[0]["cosine_sum"], dtype=torch.float64
        )
        profile_count = 0
        for batch in batches:
            temporal_sum.add_(batch["temporal_sum"].to(torch.float64))
            cosine_sum.add_(batch["cosine_sum"].to(torch.float64))
            profile_count += batch["profile_count"]

        return {
            "variable_routing": torch.cat([
                batch["variable_routing"] for batch in batches
            ], dim=0),
            "patch_routing": torch.cat([
                batch["patch_routing"] for batch in batches
            ], dim=0),
            "variable_entropy": torch.cat([
                batch["variable_entropy"] for batch in batches
            ], dim=0),
            "patch_entropy": torch.cat([
                batch["patch_entropy"] for batch in batches
            ], dim=0),
            "temporal_profile": temporal_sum / profile_count,
            "temporal_cosine_similarity": cosine_sum / profile_count,
        }

    # 将原始 MAE patch representation 映射为第一阶段 Global Attention 的 memory。
    def project_patch_memory(self, patch_tokens):
        patches = self.patch_down(self.patch_norm(patch_tokens))
        return self.patch_memory_norm(patches)

    # 使用已经跨全部变量求平均的公共 patch，只计算一次 Global Token。
    def global_latent_memory(self, mean_patch_memory, num_variables):
        batch_size, num_patches, _ = mean_patch_memory.shape
        latents = self.latent_queries.expand(batch_size, -1, -1)
        latent_query = self.latent_query_norm(latents)
        latent_update = self.latent_cross_attention(
            latent_query,
            mean_patch_memory,
            mean_patch_memory,
            need_weights=False,
        )[0]

        if self.statistics_mode == "test":
            # 额外的无梯度调用只取 softmax 后的 per-head 权重；
            # 上面用于模型输出的 attention 调用及 kernel 保持不变。
            with torch.no_grad():
                attention_weights = self.latent_cross_attention(
                    latent_query,
                    mean_patch_memory,
                    mean_patch_memory,
                    need_weights=True,
                    average_attn_weights=False,
                )[1]
            statistics_batch = self._record_test_statistics(
                attention_weights, num_variables, num_patches
            )
            self._test_statistics_batches.append(statistics_batch)

        return self.latent_memory_norm(latents + latent_update)

    # 单个 Global Token 作为唯一 K/V 时 softmax 恒为 1，只需执行 V 与 output projection。
    def shared_correction(self, latent_memory):
        if latent_memory.shape[1] != 1:
            raise RuntimeError(
                "Shared correction requires exactly one Global Token."
            )
        attention = self.patch_cross_attention
        if attention.training and attention.dropout != 0:
            raise RuntimeError(
                "Single-source attention shortcut requires zero dropout."
            )
        if attention.bias_k is not None or attention.bias_v is not None:
            raise RuntimeError(
                "Single-source attention shortcut does not support bias_kv."
            )
        if attention.add_zero_attn:
            raise RuntimeError(
                "Single-source attention shortcut does not support zero attention."
            )

        embed_dim = attention.embed_dim
        if attention._qkv_same_embed_dim:
            value_weight = attention.in_proj_weight[2 * embed_dim:]
        else:
            value_weight = attention.v_proj_weight
        value_bias = (
            None if attention.in_proj_bias is None
            else attention.in_proj_bias[2 * embed_dim:]
        )
        correction = F.linear(latent_memory, value_weight, value_bias)
        correction = attention.out_proj(correction)
        correction = self.patch_up(correction)
        # 仅保留 [B,1,1,E]，后续依靠 broadcasting 注入当前 variable chunk。
        return correction.unsqueeze(2)

    def correction_from_mean_memory(self, mean_patch_memory, num_variables):
        latent_memory = self.global_latent_memory(
            mean_patch_memory, num_variables
        )
        return self.shared_correction(latent_memory)

    @staticmethod
    def apply_correction(patch_tokens, correction):
        return patch_tokens + correction

    # 聚合所有变量的 patch token，并为每个 patch 注入同一个全局修正表示。
    # shape 约定：B 为 batch size，V 为变量数，P 为可见 patch 数，E 为 embedding 维度。
    # 输入 patch_tokens 与输出 tensor 的 shape 均为 [B, V, P, E]。
    def forward(self, patch_tokens):
        # 从输入 shape [B, V, P, E] 中读取各维度大小。
        batch_size, num_variables, num_patches, embed_dim = patch_tokens.shape
        # 归一化后恢复变量维，并对同一 patch 位置的所有变量直接求平均。
        # 公共 patch memory 仅供 Global Token 第一阶段 attention 的 K/V 使用。
        patch_memory = self.project_patch_memory(patch_tokens).mean(dim=1)
        correction = self.correction_from_mean_memory(
            patch_memory, num_variables
        )
        # 不显式复制 correction；由加法自动 broadcast 到全部变量和 patch。
        return self.apply_correction(patch_tokens, correction)
