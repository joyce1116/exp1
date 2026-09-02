"""Channel-token latent adapter for VisionTS patch tokens."""

# 导入 PyTorch tensor 与参数初始化功能。
import torch
# import math  # Disabled attention-entropy diagnostics.
# 导入神经网络层模块。
from torch import nn
from torch.nn import functional as F


class VariableAwareLatentAdapter(nn.Module):
    """Use channel tokens to exchange information between variables."""

    # 创建 projection、双向 cross-attention 和 LayerNorm 等子层。
    def __init__(
        self,
        embed_dim=768,  # 输入 patch token 的 embedding 维度 E。
        num_latents=1,  # 每个变量使用的 channel token 数量。
        latent_dim=192,  # attention 内部 latent space 的维度 D。
        num_heads=4,  # multi-head cross-attention 的 head 数量。
        channel_depth=1,
    ):
        # 初始化 nn.Module 的内部状态。
        super().__init__()

        if num_latents != 1:
            raise ValueError("Channel-token adapter requires num_latents=1.")
        if not isinstance(channel_depth, int) or isinstance(channel_depth, bool) \
                or channel_depth < 1:
            raise ValueError("channel_depth must be a positive integer.")
        self.num_latents = num_latents
        self.channel_depth = channel_depth
        # Existing diagnostic-statistics state is intentionally disabled.
        # self.collect_statistics = False
        # self.latest_statistics = None
        # 在降维前对输入 patch embedding 执行 LayerNorm。
        self.patch_norm = nn.LayerNorm(embed_dim)
        # 将 patch embedding 从 E 维 projection 到 D 维 latent space。
        self.patch_down = nn.Linear(embed_dim, latent_dim)

        # 创建由所有变量共享初值的可学习 channel token。
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
        self.channel_query_norm = nn.LayerNorm(latent_dim)
        self.channel_attention = nn.MultiheadAttention(
            latent_dim, num_heads, batch_first=True
        )
        self.channel_memory_norm = nn.LayerNorm(latent_dim)

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

    # 将原始 MAE patch representation 映射为 channel-token memory。
    def project_patch_memory(self, patch_tokens):
        patches = self.patch_down(self.patch_norm(patch_tokens))
        return self.patch_memory_norm(patches)

    def initial_channel_tokens(self, batch_size, num_variables):
        return self.latent_queries.expand(batch_size, num_variables, -1)

    def update_variable_tokens(self, patch_memory, channel_tokens):
        batch_size, num_variables, num_patches, latent_dim = (
            patch_memory.shape
        )
        sequence = torch.cat((
            channel_tokens.unsqueeze(2), patch_memory
        ), dim=2).reshape(
            batch_size * num_variables, num_patches + 1, latent_dim
        )
        normalized = self.latent_query_norm(sequence)
        update = self.latent_cross_attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        sequence = self.latent_memory_norm(sequence + update).reshape(
            batch_size, num_variables, num_patches + 1, latent_dim
        )
        return sequence[:, :, 1:, :], sequence[:, :, 0, :]

    def mix_channel_tokens(self, channel_tokens):
        normalized = self.channel_query_norm(channel_tokens)
        update = self.channel_attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        return self.channel_memory_norm(channel_tokens + update)

    # Existing channel diagnostic metrics are intentionally disabled and
    # retained as comments.
    # @staticmethod
    # def relative_rms(output, input):
    #     with torch.no_grad():
    #         output = output.detach().float()
    #         input = input.detach().float()
    #         output_rms = output.square().mean().sqrt()
    #         input_rms = input.square().mean().sqrt()
    #         return output.sub(input).square().mean().sqrt() / (
    #             torch.maximum(input_rms, output_rms) + 1e-12
    #         )
    #
    # @staticmethod
    # def pair_cosine(tokens):
    #     with torch.no_grad():
    #         normalized = F.normalize(
    #             tokens.detach().float(), dim=-1, eps=1e-12
    #         )
    #         num_tokens = normalized.shape[-2]
    #         if num_tokens == 1:
    #             return normalized.new_tensor(1.0)
    #         summed = normalized.sum(dim=-2)
    #         diagonal = normalized.square().sum(dim=-1).sum(dim=-1)
    #         return (
    #             (summed.square().sum(dim=-1) - diagonal)
    #             / (num_tokens * (num_tokens - 1))
    #         ).mean()
    #
    # @classmethod
    # def channel_pair_cosine(cls, channel_tokens):
    #     return cls.pair_cosine(channel_tokens)
    #
    # @classmethod
    # def patch_pair_cosines(cls, patch_tokens):
    #     return (
    #         cls.pair_cosine(patch_tokens),
    #         cls.pair_cosine(patch_tokens.transpose(1, 2)),
    #     )
    #
    # @staticmethod
    # def pair_cosine_from_sum(token_sum, num_tokens):
    #     with torch.no_grad():
    #         if num_tokens == 1:
    #             return token_sum.new_tensor(1.0)
    #         return (
    #             (token_sum.square().sum(dim=-1) - num_tokens)
    #             / (num_tokens * (num_tokens - 1))
    #         ).mean()
    #
    # def round_statistics(
    #     self, channel_input, channel_local, channel_mixed,
    #     patch_input=None, patch_output=None, patch_ratio=None,
    #     patch_input_cosines=None, patch_output_cosines=None
    # ):
    #     if patch_ratio is None:
    #         patch_ratio = self.relative_rms(patch_output, patch_input)
    #     if patch_input_cosines is None:
    #         patch_input_cosines = self.patch_pair_cosines(patch_input)
    #     if patch_output_cosines is None:
    #         patch_output_cosines = self.patch_pair_cosines(patch_output)
    #     return torch.stack((
    #         self.relative_rms(channel_local, channel_input),
    #         patch_ratio,
    #         self.relative_rms(channel_mixed, channel_local),
    #         self.channel_pair_cosine(channel_local),
    #         self.channel_pair_cosine(channel_mixed),
    #         patch_input_cosines[0],
    #         patch_output_cosines[0],
    #         patch_input_cosines[1],
    #         patch_output_cosines[1],
    #     ))
    #
    # def complete_statistics(
    #     self, round_statistics, correction, reference=None,
    #     reference_rms=None
    # ):
    #     with torch.no_grad():
    #         correction_rms = correction.detach().float().square().mean().sqrt()
    #         if reference_rms is None:
    #             reference_rms = (
    #                 reference.detach().float().square().mean().sqrt()
    #             )
    #         result = torch.full(
    #             (self.channel_depth, 11), float("nan"),
    #             device=correction.device, dtype=torch.float32
    #         )
    #         result[:, :9] = torch.stack(round_statistics)
    #         result[-1, 9] = correction_rms
    #         result[-1, 10] = correction_rms / (reference_rms + 1e-12)
    #         return result

    # 每个 channel token 作为对应变量的唯一 K/V，只需执行 V 与 output projection。
    def shared_correction(self, latent_memory):
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
        return correction.unsqueeze(2)

    def correction_from_patch_memory(self, patch_memory):
        batch_size, num_variables = patch_memory.shape[:2]
        channel_tokens = self.initial_channel_tokens(
            batch_size, num_variables
        )
        # statistics = []  # Disabled channel diagnostics.
        for _ in range(self.channel_depth):
            patch_input = patch_memory
            channel_input = channel_tokens
            patch_memory, channel_tokens = self.update_variable_tokens(
                patch_memory, channel_tokens
            )
            channel_local = channel_tokens
            channel_tokens = self.mix_channel_tokens(channel_local)
            # if self.collect_statistics:
            #     statistics.append(self.round_statistics(
            #         channel_input, channel_local, channel_tokens,
            #         patch_input=patch_input, patch_output=patch_memory
            #     ))
        correction = self.shared_correction(channel_tokens)
        return correction, None

    @staticmethod
    def apply_correction(patch_tokens, correction):
        return patch_tokens + correction

    # 通过 channel tokens 交互，并为每个变量的 patch 注入对应修正。
    # shape 约定：B 为 batch size，V 为变量数，P 为可见 patch 数，E 为 embedding 维度。
    # 输入 patch_tokens 与输出 tensor 的 shape 均为 [B, V, P, E]。
    def forward(self, patch_tokens, return_correction=False):
        # 从输入 shape [B, V, P, E] 中读取各维度大小。
        batch_size, num_variables, num_patches, embed_dim = patch_tokens.shape
        # self.latest_statistics = None
        patch_memory = self.project_patch_memory(patch_tokens)
        correction, statistics = self.correction_from_patch_memory(patch_memory)
        # if self.collect_statistics:
        #     self.latest_statistics = self.complete_statistics(
        #         statistics, correction, reference=patch_tokens
        #     )
        output = self.apply_correction(patch_tokens, correction)
        if return_correction:
            return output, correction
        return output


class RelativePositionAttention(nn.Module):
    def __init__(self, dim, num_heads, num_offsets):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("Attention dimension must be divisible by heads.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.relative_bias = nn.Parameter(torch.zeros(num_heads, num_offsets))

    def forward(self, x, bias_index, collect_statistics=False):
        leading_shape = x.shape[:-2]
        length, dim = x.shape[-2:]
        x = x.reshape(-1, length, dim)
        q = self.q_proj(x).reshape(
            x.shape[0], length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(x).reshape(
            x.shape[0], length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(x).reshape(
            x.shape[0], length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        logits = logits + self.relative_bias[:, bias_index].unsqueeze(0)
        weights = logits.softmax(dim=-1)
        output = torch.matmul(weights, v).transpose(1, 2).reshape(
            x.shape[0], length, dim
        )
        output = self.out_proj(output).reshape(*leading_shape, length, dim)
        return output, None

        # Existing attention entropy/offset-mass diagnostics are intentionally
        # disabled and retained as comments.
        # if not collect_statistics:
        #     return output, None
        # with torch.no_grad():
        #     probabilities = weights.detach().float()
        #     if length > 1:
        #         entropy = -(
        #             probabilities * probabilities.clamp_min(1e-12).log()
        #         ).sum(dim=-1).mean() / math.log(length)
        #     else:
        #         entropy = probabilities.new_tensor(0.0)
        #     masses = torch.stack([
        #         probabilities[..., bias_index == offset].sum()
        #         for offset in range(self.relative_bias.shape[1])
        #     ])
        #     masses = masses / probabilities.sum().clamp_min(1e-12)
        # return output, (entropy, masses)


class TemporalPeriodicAdapter(nn.Module):
    def __init__(
        self, embed_dim=768, bottleneck_dim=64, num_heads=4,
        num_rows=14, num_columns=1, center=True,
        interaction_mode="axis_attention",
    ):
        super().__init__()
        if interaction_mode not in {
            "axis_attention", "global_attention", "axis_mean"
        }:
            raise ValueError(f"Unknown TP interaction mode: {interaction_mode}")
        self.num_rows = num_rows
        self.num_columns = num_columns
        self.center = center
        self.interaction_mode = interaction_mode
        self.center_norm = nn.LayerNorm(embed_dim)
        self.down = nn.Linear(embed_dim, bottleneck_dim)
        if interaction_mode != "axis_mean":
            self.temporal_attention = RelativePositionAttention(
                bottleneck_dim, num_heads, 2 * num_columns - 1
            )
            self.periodic_attention = RelativePositionAttention(
                bottleneck_dim, num_heads, num_rows
            )
        else:
            self.temporal_attention = None
            self.periodic_attention = None
        self.output_norm = nn.LayerNorm(bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, embed_dim)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        columns = torch.arange(num_columns)
        rows = torch.arange(num_rows)
        if interaction_mode == "global_attention":
            columns = columns.repeat(num_rows)
            rows = rows.repeat_interleave(num_columns)
        self.register_buffer(
            "temporal_bias_index",
            columns[:, None] - columns[None, :] + num_columns - 1,
        )
        self.register_buffer(
            "periodic_bias_index",
            (rows[:, None] - rows[None, :]) % num_rows,
        )

    # Existing TP diagnostic helpers are intentionally disabled and retained
    # as comments.
    # @staticmethod
    # def rms(x):
    #     return x.detach().float().square().mean().sqrt()
    #
    # @staticmethod
    # def pair_cosine(x):
    #     x = F.normalize(x.detach().float(), dim=-1, eps=1e-12)
    #     count = x.shape[-2]
    #     if count == 1:
    #         return x.new_tensor(1.0)
    #     summed = x.sum(dim=-2)
    #     diagonal = x.square().sum(dim=-1).sum(dim=-1)
    #     return (
    #         (summed.square().sum(dim=-1) - diagonal)
    #         / (count * (count - 1))
    #     ).mean()

    def forward(self, patch_tokens, residual_gate, collect_statistics=False):
        batch_size, num_variables, num_patches, embed_dim = patch_tokens.shape
        if num_patches != self.num_rows * self.num_columns:
            raise ValueError("Unexpected visible patch geometry.")
        grid = patch_tokens.reshape(
            batch_size, num_variables, self.num_rows,
            self.num_columns, embed_dim
        )
        centered = (
            grid - grid.mean(dim=(2, 3), keepdim=True)
            if self.center else grid
        )
        hidden = self.down(self.center_norm(centered))
        if self.interaction_mode == "axis_attention":
            temporal = hidden.reshape(
                batch_size, num_variables, self.num_rows,
                self.num_columns, hidden.shape[-1]
            )
            temporal_update, temporal_statistics = self.temporal_attention(
                temporal, self.temporal_bias_index, collect_statistics
            )
            temporal = temporal + temporal_update
            periodic = temporal.permute(0, 1, 3, 2, 4)
            periodic_update, periodic_statistics = self.periodic_attention(
                periodic, self.periodic_bias_index, collect_statistics
            )
            periodic = periodic + periodic_update
            hidden = periodic.permute(0, 1, 3, 2, 4)
        elif self.interaction_mode == "global_attention":
            flattened = hidden.reshape(
                batch_size, num_variables,
                self.num_rows * self.num_columns, hidden.shape[-1]
            )
            temporal_update, temporal_statistics = self.temporal_attention(
                flattened, self.temporal_bias_index, collect_statistics
            )
            flattened = flattened + temporal_update
            periodic_update, periodic_statistics = self.periodic_attention(
                flattened, self.periodic_bias_index, collect_statistics
            )
            hidden = (flattened + periodic_update).reshape_as(hidden)
        else:
            # Parameter-free Temporal-axis then Periodic-axis mean pooling.
            hidden = hidden + hidden.mean(dim=3, keepdim=True)
            hidden = hidden + hidden.mean(dim=2, keepdim=True)
            temporal_statistics = periodic_statistics = None
        raw_correction = self.up(self.output_norm(hidden))
        applied_correction = residual_gate * raw_correction
        output = grid + applied_correction
        return output.reshape_as(patch_tokens), None

        # Existing TP diagnostics are intentionally disabled and retained as
        # comments.
        # if not collect_statistics:
        #     return output.reshape_as(patch_tokens), None
        # with torch.no_grad():
        #     reference_rms = self.rms(grid)
        #     raw_patch_cosine = self.pair_cosine(
        #         grid.reshape(
        #             batch_size, num_variables, num_patches, embed_dim
        #         )
        #     )
        #     statistics = {
        #         "raw_patch_cosine": raw_patch_cosine,
        #         "centered_residual_ratio": self.rms(centered)
        #         / (reference_rms + 1e-12),
        #         "centered_patch_cosine": self.pair_cosine(
        #             centered.reshape(
        #                 batch_size, num_variables, num_patches, embed_dim
        #             )
        #         ),
        #         "temporal_entropy": temporal_statistics[0],
        #         "temporal_mass": temporal_statistics[1],
        #         "periodic_entropy": periodic_statistics[0],
        #         "periodic_mass": periodic_statistics[1],
        #         "raw_correction_ratio": self.rms(raw_correction)
        #         / (reference_rms + 1e-12),
        #         "beta": residual_gate.detach().float(),
        #         "applied_correction_ratio": self.rms(applied_correction)
        #         / (reference_rms + 1e-12),
        #         "update_patch_cosine": F.cosine_similarity(
        #             grid.detach().float(),
        #             applied_correction.detach().float(),
        #             dim=-1, eps=1e-12
        #         ).mean(),
        #         "before_adapter_cosine": raw_patch_cosine,
        #         "after_adapter_cosine": self.pair_cosine(
        #             output.reshape(
        #                 batch_size, num_variables, num_patches, embed_dim
        #             )
        #         ),
        #     }
        # return output.reshape_as(patch_tokens), statistics
