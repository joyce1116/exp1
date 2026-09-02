# 导入 PyTorch 以创建 tensor 并加载 model weight。
import torch

# 导入路径工具以定位 pretrained checkpoint。
from pathlib import Path

# 导入 MAE 模型构建函数。
from . import models_mae
# 导入 channel-token adapter。
from ..layers.latent_adapter import (
    TemporalPeriodicAdapter,
    VariableAwareLatentAdapter,
)
# 导入 einops 以执行 tensor 维度变换。
import einops
# 导入 PyTorch 函数接口以填充 tensor。
import torch.nn.functional as F
# 导入神经网络基类。
from torch import nn
from torch.utils.checkpoint import checkpoint
# 导入 PIL 的 interpolation 方式常量。
from PIL import Image
# 导入本项目的通用工具函数。
from ..utils import util


# 将 MAE 架构名映射到构建函数和默认 checkpoint 文件名。
MAE_ARCH = {
    # 配置 base 规模的 MAE。
    "mae_base": [models_mae.mae_vit_base_patch16, "mae_visualize_vit_base.pth"],
    # 配置 large 规模的 MAE。
    "mae_large": [models_mae.mae_vit_large_patch16, "mae_visualize_vit_large.pth"],
    # 配置 huge 规模的 MAE。
    "mae_huge": [models_mae.mae_vit_huge_patch14, "mae_visualize_vit_huge.pth"]
}

ABLATION_MODES = {
    "full",
    "wo_channel",
    "wo_tp",
    "wo_centering",
    "shared_tp",
    "wo_tp_factorization",
    "tp_mean_pooling",
}


# 定义使用视觉 MAE backbone 与跨变量 adapter 进行时间序列预测的主模型。
class VisionTS(nn.Module):

    # 分块大小固定为 64；是否启用由外层配置决定，不把大小暴露为超参数。
    variable_chunk_size = 16

    # 初始化视觉 backbone、pretrained checkpoint 和 channel-token adapter。
    def __init__(self, arch='mae_base', ckpt_path=None, load_ckpt=True,
                 num_latents=1, latent_dim=192, adapter_num_heads=4,
                 channel_depth=1, ablation_mode="full",
                 tp_bottleneck_dim=64):
        # 调用 nn.Module 的初始化逻辑。
        super(VisionTS, self).__init__()

        # 拒绝未在架构映射表中注册的模型名。
        if arch not in MAE_ARCH:
            # 说明非法架构名及可用选项。
            raise ValueError(f"Unknown arch: {arch}. Should be in {list(MAE_ARCH.keys())}")
        if not isinstance(channel_depth, int) or isinstance(channel_depth, bool) \
                or channel_depth < 1:
            raise ValueError("channel_depth must be a positive integer.")
        if ablation_mode not in ABLATION_MODES:
            raise ValueError(
                f"Unknown ablation_mode: {ablation_mode}. "
                f"Should be in {sorted(ABLATION_MODES)}"
            )
        self.ablation_mode = ablation_mode
        self.tp_bottleneck_dim = tp_bottleneck_dim
        self.use_channel = ablation_mode != "wo_channel"
        self.use_tp = ablation_mode != "wo_tp"
        self.use_centering = ablation_mode != "wo_centering"
        self.share_tp = ablation_mode == "shared_tp"
        if ablation_mode == "wo_tp_factorization":
            self.tp_interaction_mode = "global_attention"
        elif ablation_mode == "tp_mean_pooling":
            self.tp_interaction_mode = "axis_mean"
        else:
            self.tp_interaction_mode = "axis_attention"

        # 调用对应构建函数创建 MAE 视觉 backbone。
        self.vision_model = MAE_ARCH[arch][0]()

        # 根据开关决定是否加载 pretrained checkpoint。
        if load_ckpt:
            # 未显式指定路径时组合出项目内的默认 checkpoint 路径。
            if ckpt_path is None:
                ckpt_path = (
                    # 从当前文件回溯到项目根目录。
                    Path(__file__).resolve().parents[4]
                    # 进入 pretrained checkpoint 根目录。
                    / "pretrained_weights"
                    # 进入 MAE checkpoint 子目录。
                    / "mae"
                    # 追加当前架构对应的 checkpoint 文件名。
                    / MAE_ARCH[arch][1]
                )
            # 将 checkpoint 加载到 CPU，避免强制占用 GPU。
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            # 严格将 checkpoint 中的 state_dict 写入 MAE backbone。
            self.vision_model.load_state_dict(checkpoint['model'], strict=True)

        # 图片步骤 1：冻结完整 MAE，包括 Encoder、Decoder、Norm、Attention、MLP 和所有 projection。
        for param in self.vision_model.parameters():
            # 关闭每个 backbone 参数的梯度计算。
            param.requires_grad = False

        # 创建 channel-token adapter。
        if self.use_channel:
            self.variable_adapter = VariableAwareLatentAdapter(
                # 使 adapter 输入维度与 MAE positional embedding 维度一致。
                embed_dim=self.vision_model.pos_embed.shape[-1],
                # 设置每个变量的 channel token 数量。
                num_latents=num_latents,
                # 设置每个 latent 的特征维度。
                latent_dim=latent_dim,
                # 设置 adapter 的 attention head 数。
                num_heads=adapter_num_heads,
                channel_depth=channel_depth,
            )
        else:
            self.variable_adapter = None
        if self.use_channel and self.use_tp:
            self.fusion_logit = nn.Parameter(torch.zeros(()))
        else:
            self.fusion_logit = None
    # 根据序列长度和周期更新图像化、对齐与 mask 配置。
    def update_config(self, context_len, pred_len, periodicity=1, norm_const=0.4, align_const=0.4, interpolation='bilinear'):
        # 读取 MAE 期望的方形图像边长。
        self.image_size = self.vision_model.patch_embed.img_size[0]
        # 读取 MAE 的 patch 边长。
        self.patch_size = self.vision_model.patch_embed.patch_size[0]
        # 计算单个维度上的 patch 数量。
        self.num_patch = self.image_size // self.patch_size

        # 保存输入回看窗口长度。
        self.context_len = context_len
        # 保存预测窗口长度。
        self.pred_len = pred_len
        # 保存将一维序列折叠为二维图像的周期。
        self.periodicity = periodicity

        # 默认输入左侧无需填充。
        self.pad_left = 0
        # 默认输出右侧无需填充。
        self.pad_right = 0
        # 检查回看窗口能否按周期完整分组。
        if self.context_len % self.periodicity != 0:
            # 计算输入左侧补齐到整周期所需的点数。
            self.pad_left = self.periodicity - self.context_len % self.periodicity

        # 检查预测窗口能否按周期完整分组。
        if self.pred_len % self.periodicity != 0:
            # 计算输出右侧补齐到整周期所需的点数。
            self.pad_right = self.periodicity - self.pred_len % self.periodicity

        # 计算补齐后的历史区间在整张序列图中的比例。
        input_ratio = (self.pad_left + self.context_len) / (self.pad_left + self.context_len + self.pad_right + self.pred_len)
        # 结合对齐系数，将历史区间比例换算为输入 patch 数。
        self.num_patch_input = int(input_ratio * self.num_patch * align_const)
        # 保证历史区间至少占用一个 patch。
        if self.num_patch_input == 0:
            # 将过小的输入 patch 数下限设为一。
            self.num_patch_input = 1
        # 将剩余的 patch 划为待重建的预测区间。
        self.num_patch_output = self.num_patch - self.num_patch_input
        # 用实际 patch 数重新计算对齐后的输入比例。
        adjust_input_ratio = self.num_patch_input / self.num_patch

        # 将字符串 interpolation 方式转换为 PIL 常量。
        interpolation = {
            # 选用 bilinear interpolation。
            "bilinear": Image.BILINEAR,
            # 选用 nearest-neighbor interpolation。
            "nearest": Image.NEAREST,
            # 选用 bicubic interpolation。
            "bicubic": Image.BICUBIC,
        }[interpolation]

        # 将历史序列图缩放到 MAE 高度和历史 patch 宽度。
        self.input_resize = util.safe_resize((self.image_size, int(self.image_size * adjust_input_ratio)), interpolation=interpolation)
        # 计算从图像宽度恢复到时间步数时的水平缩放比。
        self.scale_x = ((self.pad_left + self.context_len) // self.periodicity) / (int(self.image_size * adjust_input_ratio))
        # 创建将重建图像恢复为周期分段尺寸的缩放器。
        self.output_resize = util.safe_resize((self.periodicity, int(round(self.image_size * self.scale_x))), interpolation=interpolation)
        # 保存 normalization 幅度调节常数。
        self.norm_const = norm_const

        # 创建默认全部遮挡的二维 patch mask。
        mask = torch.ones((self.num_patch, self.num_patch)).to(self.vision_model.cls_token.device)
        # 将左侧历史 patch 标记为保留区域。
        mask[:, :self.num_patch_input] = torch.zeros((self.num_patch, self.num_patch_input))
        # 将 mask 展平后注册为会跟随模型迁移设备的 buffer。
        self.register_buffer("mask", mask.float().reshape((1, -1)))
        # 记录 mask 中待重建 patch 所占的比例。
        self.mask_ratio = torch.mean(mask).item()
        if self.use_tp and (
            self.num_patch != 14 or len(self.vision_model.blocks) != 12
        ):
            raise ValueError("Temporal-periodic branch requires mae_base geometry.")
        if self.use_tp and self.share_tp:
            self.tp_shared = TemporalPeriodicAdapter(
                embed_dim=self.vision_model.pos_embed.shape[-1],
                bottleneck_dim=self.tp_bottleneck_dim,
                num_heads=4,
                num_rows=14,
                num_columns=self.num_patch_input,
                center=self.use_centering,
                interaction_mode=self.tp_interaction_mode,
            )
            self.temporal_periodic_adapters = None
        elif self.use_tp:
            self.temporal_periodic_adapters = nn.ModuleList([
                TemporalPeriodicAdapter(
                    embed_dim=self.vision_model.pos_embed.shape[-1],
                    bottleneck_dim=self.tp_bottleneck_dim,
                    num_heads=4,
                    num_rows=14,
                    num_columns=self.num_patch_input,
                    center=self.use_centering,
                    interaction_mode=self.tp_interaction_mode,
                )
                for _ in range(3)
            ])
            self.tp_shared = None
        else:
            self.temporal_periodic_adapters = None
            self.tp_shared = None
        if self.use_tp:
            self.tp_residual_logits = nn.Parameter(torch.full((12,), -2.944))
        else:
            self.tp_residual_logits = None

    def _normalize_series(self, x, fp64=False):
        means = x.mean(1, keepdim=True).detach()
        x_enc = x - means
        stdev = torch.sqrt(
            torch.var(
                x_enc.to(torch.float64) if fp64 else x_enc,
                dim=1, keepdim=True, unbiased=False
            ) + 1e-5
        )
        stdev /= self.norm_const
        x_enc /= stdev
        return einops.rearrange(x_enc, 'b s n -> b n s'), means, stdev

    def _render_variable_images(self, x_enc):
        batch_size, num_variables, _ = x_enc.shape
        x_pad = F.pad(x_enc, (self.pad_left, 0), mode='replicate')
        x_2d = einops.rearrange(
            x_pad, 'b n (p f) -> (b n) 1 f p', f=self.periodicity
        )
        x_resize = self.input_resize(x_2d)
        masked = torch.zeros(
            (
                batch_size * num_variables, 1, self.image_size,
                self.num_patch_output * self.patch_size
            ),
            device=x_2d.device,
            dtype=x_2d.dtype,
        )
        image_input = torch.cat((x_resize, masked), dim=-1)
        return einops.repeat(
            image_input, 'b 1 h w -> b c h w', c=3
        )

    def _shared_mask_noise(self):
        ids_shuffle = torch.argsort(self.mask, dim=1)
        ranks = torch.arange(
            self.mask.shape[1], device=self.mask.device, dtype=self.mask.dtype
        ).unsqueeze(0)
        return torch.empty_like(self.mask).scatter_(1, ids_shuffle, ranks)

    def _visible_grid_permutations(self, noise):
        visible_positions = torch.argsort(noise, dim=1)[
            :, :self.num_patch * self.num_patch_input
        ]
        to_grid = torch.argsort(visible_positions, dim=1)[0]
        return to_grid, torch.argsort(to_grid)

    def _visible_patch_chunk(self, x_enc, noise):
        batch_size, num_variables, _ = x_enc.shape
        image_input = self._render_variable_images(x_enc)
        visible, _, _ = self.vision_model.prepare_visible_patches(
            image_input,
            self.mask_ratio,
            noise.expand(batch_size * num_variables, -1),
        )
        return visible.reshape(
            batch_size, num_variables, visible.shape[1], visible.shape[2]
        )

    def _channel_token_chunk(self, x_enc, noise, *channel_history):
        visible = self._visible_patch_chunk(x_enc, noise)
        patch_memory = self.variable_adapter.project_patch_memory(visible)
        for channel_tokens in channel_history:
            # patch_input = patch_memory  # Disabled channel diagnostics.
            patch_memory, channel_tokens = (
                self.variable_adapter.update_variable_tokens(
                    patch_memory, channel_tokens
                )
            )
        # Existing channel diagnostic calculations are intentionally disabled.
        # if self.variable_adapter.collect_statistics:
        #     with torch.no_grad():
        #         normalized_patch_input = F.normalize(
        #             patch_input.detach().float(), dim=-1, eps=1e-12
        #         )
        #         normalized_patch_output = F.normalize(
        #             patch_memory.detach().float(), dim=-1, eps=1e-12
        #         )
        #         num_patches = normalized_patch_output.shape[2]
        #         patch_input_sum = normalized_patch_input.sum(dim=2)
        #         patch_output_sum = normalized_patch_output.sum(dim=2)
        #         return (
        #             channel_tokens,
        #             patch_memory.detach().float().sub(
        #                 patch_input.detach().float()
        #             ).square().sum(),
        #             patch_input.detach().float().square().sum(),
        #             patch_memory.detach().float().square().sum(),
        #             visible.detach().float().square().sum(),
        #             visible.new_tensor(visible.numel(), dtype=torch.float32),
        #             (
        #                 patch_input_sum.square().sum(dim=-1) - num_patches
        #             ).sum(),
        #             (
        #                 patch_output_sum.square().sum(dim=-1) - num_patches
        #             ).sum(),
        #             visible.new_tensor(
        #                 normalized_patch_output.shape[0]
        #                 * normalized_patch_output.shape[1]
        #                 * num_patches * (num_patches - 1),
        #                 dtype=torch.float32
        #             ),
        #             normalized_patch_input.sum(dim=1),
        #             normalized_patch_output.sum(dim=1),
        #         )
        return channel_tokens

    def prepare_variable_chunk_context(self, x, fp64=False):
        x_enc, means, stdev = self._normalize_series(x, fp64=fp64)
        num_variables = x_enc.shape[1]
        noise = self._shared_mask_noise()
        if not self.use_channel:
            return {
                "x_enc": x_enc,
                "means": means,
                "stdev": stdev,
                "noise": noise,
                "grid_permutations": self._visible_grid_permutations(noise),
                "correction": None,
                "num_variables": num_variables,
            }
        channel_history = [
            self.variable_adapter.initial_channel_tokens(
                x_enc.shape[0], num_variables
            )
        ]
        # statistics = []  # Disabled channel diagnostics.
        for _ in range(self.variable_adapter.channel_depth):
            token_chunks = []
            # patch_delta_square = 0
            # patch_input_square = 0
            # patch_output_square = 0
            # reference_square = 0
            # reference_count = 0
            # within_patch_input_cosine_sum = 0
            # within_patch_output_cosine_sum = 0
            # within_patch_cosine_count = 0
            # between_patch_input_token_sum = None
            # between_patch_output_token_sum = None
            for start in range(0, num_variables, self.variable_chunk_size):
                end = min(start + self.variable_chunk_size, num_variables)
                variable_chunk = x_enc[:, start:end, :]
                history_chunk = [
                    channel_tokens[:, start:end, :]
                    for channel_tokens in channel_history
                ]
                if self.training and torch.is_grad_enabled():
                    token_chunk = checkpoint(
                        self._channel_token_chunk,
                        variable_chunk,
                        noise,
                        *history_chunk,
                        use_reentrant=False,
                    )
                else:
                    token_chunk = self._channel_token_chunk(
                        variable_chunk, noise, *history_chunk
                    )
                # Existing per-chunk channel diagnostics are disabled.
                # if self.variable_adapter.collect_statistics:
                #     (
                #         token_chunk, patch_delta, patch_input, patch_output,
                #         visible_square, visible_count, within_patch_input_sum,
                #         within_patch_output_sum, within_patch_count,
                #         between_patch_input_sum, between_patch_output_sum
                #     ) = token_chunk
                #     patch_delta_square = patch_delta_square + patch_delta
                #     patch_input_square = patch_input_square + patch_input
                #     patch_output_square = patch_output_square + patch_output
                #     reference_square = reference_square + visible_square
                #     reference_count = reference_count + visible_count
                #     within_patch_input_cosine_sum = (
                #         within_patch_input_cosine_sum
                #         + within_patch_input_sum
                #     )
                #     within_patch_output_cosine_sum = (
                #         within_patch_output_cosine_sum
                #         + within_patch_output_sum
                #     )
                #     within_patch_cosine_count = (
                #         within_patch_cosine_count + within_patch_count
                #     )
                #     between_patch_input_token_sum = (
                #         between_patch_input_sum
                #         if between_patch_input_token_sum is None
                #         else between_patch_input_token_sum
                #         + between_patch_input_sum
                #     )
                #     between_patch_output_token_sum = (
                #         between_patch_output_sum
                #         if between_patch_output_token_sum is None
                #         else between_patch_output_token_sum
                #         + between_patch_output_sum
                #     )
                token_chunks.append(token_chunk)
            channel_local = torch.cat(token_chunks, dim=1)
            channel_tokens = self.variable_adapter.mix_channel_tokens(
                channel_local
            )
            # Existing per-round channel diagnostics are disabled.
            # if self.variable_adapter.collect_statistics:
            #     statistics.append(
            #         self.variable_adapter.round_statistics(
            #             channel_history[-1], channel_local, channel_tokens,
            #             patch_ratio=(
            #                 patch_delta_square.sqrt()
            #                 / (
            #                     torch.maximum(
            #                         patch_input_square.sqrt(),
            #                         patch_output_square.sqrt()
            #                     ) + 1e-12
            #                 )
            #             ),
            #             patch_input_cosines=(
            #                 torch.where(
            #                     within_patch_cosine_count > 0,
            #                     within_patch_input_cosine_sum
            #                     / within_patch_cosine_count.clamp_min(1),
            #                     within_patch_input_cosine_sum.new_tensor(1.0)
            #                 ),
            #                 self.variable_adapter.pair_cosine_from_sum(
            #                     between_patch_input_token_sum, num_variables
            #                 ),
            #             ),
            #             patch_output_cosines=(
            #                 torch.where(
            #                     within_patch_cosine_count > 0,
            #                     within_patch_output_cosine_sum
            #                     / within_patch_cosine_count.clamp_min(1),
            #                     within_patch_output_cosine_sum.new_tensor(1.0)
            #                 ),
            #                 self.variable_adapter.pair_cosine_from_sum(
            #                     between_patch_output_token_sum, num_variables
            #                 ),
            #             )
            #         )
            #     )
            channel_history.append(channel_tokens)
        correction = self.variable_adapter.shared_correction(channel_tokens)
        # Existing completed channel diagnostics are disabled.
        # if self.variable_adapter.collect_statistics:
        #     self.variable_adapter.latest_statistics = (
        #         self.variable_adapter.complete_statistics(
        #             statistics, correction,
        #             reference_rms=(
        #                 reference_square / reference_count
        #             ).sqrt()
        #         )
        #     )
        return {
            "x_enc": x_enc,
            "means": means,
            "stdev": stdev,
            "noise": noise,
            "grid_permutations": self._visible_grid_permutations(noise),
            "correction": correction,
            "num_variables": num_variables,
        }

    def _sequence_with_cls(self, patch_tokens):
        batch_size, num_variables, num_patches, embed_dim = patch_tokens.shape
        patches = patch_tokens.reshape(
            batch_size * num_variables, num_patches, embed_dim
        )
        cls_token = self.vision_model.cls_token + self.vision_model.pos_embed[:, :1]
        return torch.cat((cls_token.expand(patches.shape[0], -1, -1), patches), dim=1)

    def _encode_branches(
        self, visible, grid_permutations, correction=None,
        collect_diagnostics=False
    ):
        channel_latent = None
        if self.use_channel:
            if correction is None:
                channel_patches, correction = self.variable_adapter(
                    visible, return_correction=True
                )
            else:
                channel_patches = self.variable_adapter.apply_correction(
                    visible, correction
                )
            channel_latent = self._sequence_with_cls(channel_patches)
            for block in self.vision_model.blocks:
                channel_latent = block(channel_latent)
            channel_latent = self.vision_model.norm(channel_latent)

        tp_latent = None
        tp_statistics = []
        if self.use_tp:
            tp_latent = self._sequence_with_cls(visible)
            to_grid, from_grid = grid_permutations
            for layer, block in enumerate(self.vision_model.blocks):
                tp_latent = block(tp_latent)
                patches = tp_latent[:, 1:].reshape_as(visible)
                grid_patches = patches.index_select(2, to_grid)
                adapter = (
                    self.tp_shared if self.share_tp
                    else self.temporal_periodic_adapters[layer // 4]
                )
                grid_patches, statistics = adapter(
                    grid_patches,
                    torch.sigmoid(self.tp_residual_logits[layer]),
                    False,
                )
                # Existing TP diagnostics are intentionally disabled.
                # if collect_diagnostics:
                #     tp_statistics.append(statistics)
                patches = grid_patches.index_select(2, from_grid)
                tp_latent = torch.cat((
                    tp_latent[:, :1],
                    patches.reshape(
                        tp_latent.shape[0], patches.shape[2], patches.shape[3]
                    ),
                ), dim=1)
            tp_latent = self.vision_model.norm(tp_latent)
        return channel_latent, tp_latent, correction, tp_statistics

    def _decode_normalized(
        self, latent, ids_restore, batch_size, num_variables,
        return_image=False
    ):
        prediction = self.vision_model.forward_decoder(latent, ids_restore)
        image = self.vision_model.unpatchify(prediction)
        segments = self.output_resize(image.mean(1, keepdim=True))
        flattened = einops.rearrange(
            segments, '(b n) 1 f p -> b (p f) n',
            b=batch_size, f=self.periodicity
        )
        normalized = flattened[
            :,
            self.pad_left + self.context_len:
            self.pad_left + self.context_len + self.pred_len,
            :,
        ]
        if return_image:
            return normalized, image
        return normalized

    # Existing channel/TP diagnostic formatters are intentionally disabled
    # and retained as comments.
    # def _channel_diagnostics(self, correction):
    #     statistics = self.variable_adapter.latest_statistics
    #     result = {}
    #     for round_index in range(self.variable_adapter.channel_depth):
    #         result[f"channel_q_pre_inter_cos_r{round_index + 1}"] = (
    #             statistics[round_index, 3]
    #         )
    #         result[f"channel_q_post_inter_cos_r{round_index + 1}"] = (
    #             statistics[round_index, 4]
    #         )
    #     result["channel_correction_ratio"] = statistics[-1, 10]
    #     result["channel_correction_crossvar_cosine"] = (
    #         self.variable_adapter.pair_cosine(correction.squeeze(2))
    #     )
    #     return result
    #
    # def _format_tp_diagnostics(self, tp_statistics):
    #     result = {}
    #     for layer, statistics in enumerate(tp_statistics, 1):
    #         prefix = f"tp_block_{layer}"
    #         for name in (
    #             "raw_patch_cosine", "centered_residual_ratio",
    #             "centered_patch_cosine", "temporal_entropy",
    #             "periodic_entropy", "raw_correction_ratio", "beta",
    #             "applied_correction_ratio", "update_patch_cosine",
    #             "before_adapter_cosine", "after_adapter_cosine",
    #         ):
    #             result[f"{prefix}_{name}"] = statistics[name]
    #         for index, value in enumerate(statistics["temporal_mass"]):
    #             offset = index - self.num_patch_input + 1
    #             label = "0" if offset == 0 else f"{offset:+d}"
    #             result[f"{prefix}_temporal_mass_offset_{label}"] = value
    #         for offset, value in enumerate(statistics["periodic_mass"]):
    #             result[f"{prefix}_periodic_mass_offset_{offset}"] = value
    #     return result

    def _forward_visible(
        self, visible, ids_restore, grid_permutations, means, stdev,
        correction=None,
        collect_diagnostics=False, return_image=False, return_branches=False
    ):
        batch_size, num_variables = visible.shape[:2]
        channel_latent, tp_latent, correction, tp_statistics = (
            self._encode_branches(
                visible, grid_permutations, correction=correction,
                collect_diagnostics=collect_diagnostics
            )
        )
        channel_decoded = None
        if self.use_channel:
            channel_decoded = self._decode_normalized(
                channel_latent, ids_restore, batch_size, num_variables,
                return_image=return_image
            )
        tp_decoded = None
        if self.use_tp:
            tp_decoded = self._decode_normalized(
                tp_latent, ids_restore, batch_size, num_variables,
                return_image=return_image
            )
        if return_image:
            if self.use_channel:
                channel_normalized, channel_image = channel_decoded
            if self.use_tp:
                tp_normalized, tp_image = tp_decoded
        else:
            if self.use_channel:
                channel_normalized = channel_decoded
            if self.use_tp:
                tp_normalized = tp_decoded
        if self.use_channel and self.use_tp:
            gate = torch.sigmoid(self.fusion_logit)
            final_normalized = (
                gate * channel_normalized + (1 - gate) * tp_normalized
            )
            if return_image:
                final_image = gate * channel_image + (1 - gate) * tp_image
        elif self.use_channel:
            final_normalized = channel_normalized
            if return_image:
                final_image = channel_image
        else:
            final_normalized = tp_normalized
            if return_image:
                final_image = tp_image
        result = {
            "output": final_normalized * stdev + means,
        }
        if return_branches:
            if self.use_channel and self.use_tp:
                result.update({
                    "channel": channel_normalized * stdev + means,
                    "tp": tp_normalized * stdev + means,
                    "channel_normalized": channel_normalized,
                    "tp_normalized": tp_normalized,
                })
            elif self.use_channel:
                result.update({
                    "channel": channel_normalized * stdev + means,
                    "channel_normalized": channel_normalized,
                })
            else:
                result.update({
                    "tp": tp_normalized * stdev + means,
                    "tp_normalized": tp_normalized,
                })
        if return_image:
            result["image"] = final_image
        # Existing diagnostic metrics are intentionally disabled.
        # if collect_diagnostics:
        #     result["diagnostics"] = {
        #         **self._channel_diagnostics(correction),
        #         **self._format_tp_diagnostics(tp_statistics),
        #     }
        return result

    def forward_variable_chunk(
        self, context, start, end, correction=None, export_image=False,
        return_branches=False, return_diagnostics=False
    ):
        x_enc = context["x_enc"][:, start:end, :]
        batch_size, num_variables, _ = x_enc.shape
        image_input = self._render_variable_images(x_enc)
        visible, mask, ids_restore = self.vision_model.prepare_visible_patches(
            image_input,
            self.mask_ratio,
            context["noise"].expand(batch_size * num_variables, -1),
        )
        visible = visible.reshape(
            batch_size, num_variables, visible.shape[1], visible.shape[2]
        )
        if self.use_channel:
            if correction is None:
                correction = context["correction"]
            correction = correction[:, start:end]
        else:
            correction = None
        result = self._forward_visible(
            visible, ids_restore, context["grid_permutations"],
            context["means"][:, :, start:end],
            context["stdev"][:, :, start:end],
            correction=correction,
            collect_diagnostics=False,
            return_image=export_image,
            return_branches=return_branches,
        )
        if export_image:
            mask_image = self.vision_model.unpatchify(
                mask.detach().unsqueeze(-1).repeat(
                    1, 1, self.patch_size ** 2 * 3
                )
            )
            reconstructed = (
                image_input * (1 - mask_image) + result["image"] * mask_image
            )
            green_bg = -torch.ones_like(image_input) * 2
            masked_input = image_input * (1 - mask_image) + green_bg * mask_image
            result["input_image"] = einops.rearrange(
                masked_input, '(b n) c h w -> b n h w c', b=batch_size
            )
            result["reconstructed_image"] = einops.rearrange(
                reconstructed, '(b n) c h w -> b n h w c', b=batch_size
            )
        # or return_diagnostics  # Disabled diagnostic return path.
        if return_branches:
            return result
        if export_image:
            return (
                result["output"], result["input_image"],
                result["reconstructed_image"]
            )
        return result["output"]

    def _forward_variable_chunks(
        self, x, export_image=False, fp64=False, return_branches=False,
        return_diagnostics=False
    ):
        context = self.prepare_variable_chunk_context(x, fp64=fp64)
        results = []
        # widths = []  # Disabled diagnostic aggregation.
        for start in range(
            0, context["num_variables"], self.variable_chunk_size
        ):
            end = min(
                start + self.variable_chunk_size,
                context["num_variables"],
            )
            results.append(self.forward_variable_chunk(
                context, start, end, export_image=export_image,
                return_branches=return_branches,
                return_diagnostics=False,
            ))
            # widths.append(end - start)
        # or return_diagnostics  # Disabled diagnostic return path.
        if not return_branches:
            if export_image:
                return (
                    torch.cat([item[0] for item in results], dim=-1),
                    torch.cat([item[1] for item in results], dim=1),
                    torch.cat([item[2] for item in results], dim=1),
                )
            return torch.cat(results, dim=-1)
        merged_names = (
            "output", "channel", "tp", "channel_normalized",
            "tp_normalized"
        )
        merged = {
            name: torch.cat([item[name] for item in results], dim=-1)
            for name in merged_names if name in results[0]
        }
        if export_image:
            merged["input_image"] = torch.cat(
                [item["input_image"] for item in results], dim=1
            )
            merged["reconstructed_image"] = torch.cat(
                [item["reconstructed_image"] for item in results], dim=1
            )
        # Existing diagnostic aggregation is intentionally disabled.
        # if return_diagnostics:
        #     diagnostics = self._channel_diagnostics(context["correction"])
        #     tp_keys = [
        #         key for key in results[0]["diagnostics"]
        #         if key.startswith("tp_block_")
        #     ]
        #     total_width = sum(widths)
        #     for key in tp_keys:
        #         diagnostics[key] = sum(
        #             item["diagnostics"][key] * width
        #             for item, width in zip(results, widths)
        #         ) / total_width
        #     merged["diagnostics"] = diagnostics
        return merged

    def _attach_statistics(self, result, return_statistics):
        # Existing channel statistics are intentionally disabled.
        # if not return_statistics:
        #     return result
        # return result, self.variable_adapter.latest_statistics.unsqueeze(0)
        return result

    def forward(
        self, x, export_image=False, fp64=False, use_variable_chunk=False,
        return_statistics=False, return_branches=False,
        return_diagnostics=False
    ):
        # Existing diagnostic collection is intentionally disabled.
        # self.variable_adapter.collect_statistics = (
        #     return_statistics or return_diagnostics
        # )
        # self.variable_adapter.latest_statistics = None
        if use_variable_chunk:
            result = self._forward_variable_chunks(
                x, export_image=export_image, fp64=fp64,
                return_branches=return_branches,
                return_diagnostics=False,
            )
            return self._attach_statistics(result, return_statistics)
        x_enc, means, stdev = self._normalize_series(x, fp64=fp64)
        batch_size, num_variables = x_enc.shape[:2]
        image_input = self._render_variable_images(x_enc)
        noise = self._shared_mask_noise()
        visible, mask, ids_restore = self.vision_model.prepare_visible_patches(
            image_input,
            self.mask_ratio,
            noise.expand(batch_size * num_variables, -1),
        )
        visible = visible.reshape(
            batch_size, num_variables, visible.shape[1], visible.shape[2]
        )
        result = self._forward_visible(
            visible, ids_restore, self._visible_grid_permutations(noise),
            means, stdev,
            collect_diagnostics=False,
            return_image=export_image,
            return_branches=return_branches,
        )
        if export_image:
            mask_image = self.vision_model.unpatchify(
                mask.detach().unsqueeze(-1).repeat(
                    1, 1, self.patch_size ** 2 * 3
                )
            )
            reconstructed = (
                image_input * (1 - mask_image) + result["image"] * mask_image
            )
            green_bg = -torch.ones_like(image_input) * 2
            masked_input = image_input * (1 - mask_image) + green_bg * mask_image
            result["input_image"] = einops.rearrange(
                masked_input, '(b n) c h w -> b n h w c', b=batch_size
            )
            result["reconstructed_image"] = einops.rearrange(
                reconstructed, '(b n) c h w -> b n h w c', b=batch_size
            )
        # or return_diagnostics  # Disabled diagnostic return path.
        if return_branches:
            return self._attach_statistics(result, return_statistics)
        if export_image:
            output = (
                result["output"], result["input_image"],
                result["reconstructed_image"]
            )
        else:
            output = result["output"]
        return self._attach_statistics(output, return_statistics)
