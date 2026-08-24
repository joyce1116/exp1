# 导入 PyTorch 以创建 tensor 并加载 model weight。
import torch
# 导入数学函数，用于按当前可访问 source 数量归一化 routing entropy。
import math

# 导入路径工具以定位 pretrained checkpoint。
from pathlib import Path

# 导入 MAE 模型构建函数。
from . import models_mae
# 导入 Global Token adapter，用于提取并广播真实 patch 的公共成分。
from ..layers.latent_adapter import VariableAwareLatentAdapter
# 导入 einops 以执行 tensor 维度变换。
import einops
# 导入 PyTorch 函数接口以填充 tensor。
import torch.nn.functional as F
# 导入神经网络基类。
from torch import nn
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


# 图片步骤 6～11：为 Block 6～12 的每个 Attention/MLP target 执行逐 token 深度路由。
class CrossDepthResidualRouter(nn.Module):
    # 16 个 raw residual source 按 B5-A、B5-M、…、B12-A、B12-M 排列。
    source_names = tuple(
        f"B{block}-{branch}"
        for block in range(5, 13)
        for branch in ("A", "M")
    )
    # Block 5 只提供 source，因此 14 个 routing target 从 B6-A 开始。
    target_names = source_names[2:]

    # 初始化步骤 8 的 14 条 pseudo-query，以及训练/测试所需的在线统计累计量。
    def __init__(self, embed_dim):
        # 初始化 Router 的 nn.Module 状态。
        super().__init__()
        # 图片步骤 8：每个 target 各有一条 query；同一 target 内所有 patch 共用它，14 条均从 0 初始化。
        self.queries = nn.Parameter(torch.zeros(14, embed_dim))
        # 显式区分训练、验证和测试，验证 forward 不写入任何 routing 累计量。
        self.statistics_mode = "train"
        # 按 target 累计训练期 routing 相对 uniform 的绝对偏离总和。
        self.register_buffer(
            "_learning_deviation_sum", torch.zeros(14, dtype=torch.float64),
            persistent=False
        )
        # 按 target 累计训练期 normalized entropy 总和。
        self.register_buffer(
            "_learning_entropy_sum", torch.zeros(14, dtype=torch.float64),
            persistent=False
        )
        # 按 target 累计同一样本不同真实 patch 的 routing variation。
        self.register_buffer(
            "_learning_variation_sum", torch.zeros(14, dtype=torch.float64),
            persistent=False
        )
        # 按 target 累计 correction norm 相对当前 frozen branch 输出 norm 的比例。
        self.register_buffer(
            "_learning_correction_ratio_sum",
            torch.zeros(14, dtype=torch.float64), persistent=False
        )
        # 保存各 target 已统计的真实 patch token 数，供逐 token 指标求平均。
        self.register_buffer(
            "_learning_token_count", torch.zeros(14, dtype=torch.float64),
            persistent=False
        )
        # 保存各 target 已统计的原始样本数，供 token routing variation 求平均。
        self.register_buffer(
            "_learning_sample_count", torch.zeros(14, dtype=torch.float64),
            persistent=False
        )
        # 测试期只在线累计 14×16 target-source routing weight 总和，不保存逐样本数据。
        self.register_buffer(
            "_test_routing_sum", torch.zeros(14, 16, dtype=torch.float64),
            persistent=False
        )
        # 为 14 个 target 分别记录参与测试汇总的真实 patch token 数。
        self.register_buffer(
            "_test_token_count", torch.zeros(14, dtype=torch.float64),
            persistent=False
        )

    # 一个训练 epoch 写出统计后清零六个学习累计量，下一 epoch 独立重新统计。
    def _reset_learning_statistics(self):
        self._learning_deviation_sum.zero_()
        self._learning_entropy_sum.zero_()
        self._learning_variation_sum.zero_()
        # correction、token 数和 sample 数与前三项在同一 epoch 边界一起重置。
        self._learning_correction_ratio_sum.zero_()
        self._learning_token_count.zero_()
        self._learning_sample_count.zero_()

    # 清零测试矩阵累计量；具体调用由非 test→test 的模式转换控制。
    def _reset_test_statistics(self):
        self._test_routing_sum.zero_()
        self._test_token_count.zero_()

    # 只有从非 test 模式首次进入 test 时才重置，保证所有测试 batch 能共同求平均。
    def set_statistics_mode(self, mode):
        if mode == "test" and self.statistics_mode != "test":
            self._reset_test_statistics()
        self.statistics_mode = mode

    # 图片步骤 7：只对每个真实 patch 自己的 D 维 source 做无参数 RMSNorm，得到同形状 Key。
    @staticmethod
    def make_key(source):
        # 不跨 patch 求均值/std，不构造 sample-level signature；raw Value 不做此归一化。
        return F.rms_norm(source, (source.shape[-1],), eps=1e-6)

    # 图片步骤 8～11：当前 target 只比较同一 patch 位置的历史 Key，并返回输入侧 correction。
    def route(self, target_index, sources, keys):
        # 当前 target 取自己的 pseudo-query；同一 target 的所有真实 patch 共用这一条 query。
        query = self.queries[target_index].to(keys[0].dtype)
        # 图片步骤 9：逐 source 做 query·Key，结果为 [B,N,M]，不除以 sqrt(D)。
        scores = torch.stack([
            torch.matmul(key, query)
            for key in keys
        ], dim=-1)
        # 只沿 source/depth 维做 softmax，不进行任何 patch-to-patch attention。
        routing_weights = torch.softmax(scores, dim=-1)
        # 用同形状零 logits 构造当前 M 个 source 的严格 uniform routing。softmax=[1/M,1/M, ... ,1/M]
        uniform_weights = torch.softmax(torch.zeros_like(scores), dim=-1)
        # 图片步骤 11：learned-uniform 系数在 query 为 0 时严格为 0，不乘固定或可学习 scale。
        coefficients = (routing_weights - uniform_weights).to(sources[0].dtype)
        # 图片步骤 10：先创建一个全零 correction：correction.shape = [B, N, 768].用于累计所有历史 source 对当前 target 的 depth correction。
        correction = torch.zeros_like(sources[0])
        # coefficients[:, :, source_index] 原本形状是：[B, N].增加 None 后变成：[B, N, 1].
        # source.shape = [B, N, 768]. 乘法广播
        for source_index, source in enumerate(sources):
            correction = correction + (
                coefficients[:, :, source_index, None] * source
            )
        # 当前 frozen branch 尚未执行，因此同时返回 weights，待其输出产生后再记录统计。
        # 返回 [B,N,D] correction；encoder 仅把它加到当前 sublayer 的输入侧。
        # N = V * P
        return correction, routing_weights

    # 在线累计图片要求的训练学习指标或测试 deviation 矩阵，不保存逐样本、逐 patch 明细。
    def record_statistics(
        self, target_index, routing_weights, correction, target
    ):
        # validation 模式直接跳过，防止验证 routing 混入训练 epoch 或测试统计。
        if self.statistics_mode not in {"train", "test"}:
            return
        # 诊断统计不保留计算图，也不参与 forecasting loss 的反向传播。
        with torch.no_grad():
            # routing_weights 为 [B,N,M]；B×N 是当前 target 的有效真实 token 数。
            weights = routing_weights.detach().float()
            token_count = weights.shape[0] * weights.shape[1]
            # 测试期只累加每个 target-source 的权重和及有效 token 数。
            if self.statistics_mode == "test":
                source_count = weights.shape[-1]
                # 仅写入当前 target 可访问的 source 前缀，未来 source 保持未统计状态。
                self._test_routing_sum[
                    target_index, :source_count
                ].add_(weights.sum(dim=(0, 1)).to(torch.float64))
                # 累加有效真实 patch token 数后结束本次 test 统计分支。
                self._test_token_count[target_index].add_(token_count)
                return
            # 训练期的 deviation 使用 mean(|w-1/M|)，避免 signed deviation 沿 source 相消。
            source_count = weights.shape[-1]
            uniform = 1.0 / source_count
            # normalized entropy 除以 log(M)，uniform routing 对应 1。
            entropy = -(
                weights * weights.clamp_min(1e-12).log()
            ).sum(dim=-1) / math.log(source_count)
            # 对同一样本沿真实 token 维求 population std，再对 source 求平均。
            variation = weights.std(dim=1, unbiased=False).mean(dim=-1)
            # 逐 token 比较实际 correction 与当前 frozen Attention/MLP raw 输出的 L2 norm。
            correction_ratio = correction.detach().float().norm(dim=-1) / (
                target.detach().float().norm(dim=-1) + 1e-12
            )
            # 累计 deviation；epoch 结束后再除以 token_count×source_count。
            self._learning_deviation_sum[target_index].add_(
                (weights - uniform).abs().sum().to(torch.float64)
            )
            # 累计 normalized entropy，epoch 结束后按真实 token 数求平均。
            self._learning_entropy_sum[target_index].add_(
                entropy.sum().to(torch.float64)
            )
            # 累计 token routing variation，epoch 结束后按样本数求平均。
            self._learning_variation_sum[target_index].add_(
                variation.sum().to(torch.float64)
            )
            # 累计 correction ratio，epoch 结束后按真实 token 数求平均。
            self._learning_correction_ratio_sum[target_index].add_(
                correction_ratio.sum().to(torch.float64)
            )
            # 同步更新当前 target 的 token 与样本计数。
            self._learning_token_count[target_index].add_(token_count)
            self._learning_sample_count[target_index].add_(weights.shape[0])

    # 将一个 epoch 的累计量整理为 14 个 target 各一行；epoch/val_loss 由外层补入。
    def pop_learning_records(self):
        # 任一 target 尚无训练 token 时不生成不完整的 epoch 记录。
        if not torch.all(self._learning_token_count > 0):
            return []
        # query_norm 用于检查每条零初始化 pseudo-query 是否已经学离 0。
        query_norms = self.queries.detach().float().norm(dim=-1)
        # 准备收集当前 epoch 的 14 条 target 记录。
        rows = []
        # B6-A 只访问 B5-A/B5-M 两个旧 source，此后每个 target 增加一个可访问 source。
        for target_index, target_name in enumerate(self.target_names):
            source_count = target_index + 2
            token_count = self._learning_token_count[target_index]
            sample_count = self._learning_sample_count[target_index]
            # 固定写入当前 target 名称及其 pseudo-query 的 L2 norm。
            rows.append({
                "target_branch": target_name,
                "query_norm": query_norms[target_index].item(),
                # 对该 target 的全部训练 token 和全部可访问 source 求平均绝对偏离。
                "routing_deviation": (
                    self._learning_deviation_sum[target_index]
                    / (token_count * source_count)
                ).item(),
                # normalized entropy 按真实 token 数归一化，uniform routing 对应 1。
                "normalized_entropy": (
                    self._learning_entropy_sum[target_index] / token_count
                ).item(),
                # token routing variation 先逐样本统计，再按原始样本数归一化。
                "token_routing_variation": (
                    self._learning_variation_sum[target_index] / sample_count
                ).item(),
                # correction ratio 按真实 token 数归一化，比较输入侧 correction 与当前 raw branch 输出。
                "correction_ratio": (
                    self._learning_correction_ratio_sum[target_index]
                    / token_count
                ).item(),
            })
        # 当前 epoch 的 14 行已完成，清零后由下一轮训练重新累计。
        self._reset_learning_statistics()
        return rows

    # 将测试期在线累计量转换为热力图所需的 14×16 mean deviation-from-uniform 矩阵。
    def test_deviation_matrix(self):
        # 任一 target 未看到测试 token 时返回空结果，避免输出不完整矩阵。
        if not torch.all(self._test_token_count > 0):
            return None
        # 未来不可访问的 source 预填 NaN，保存 CSV 和绘图时均作为 mask。
        matrix = torch.full(
            (14, 16), float("nan"), dtype=torch.float64,
            device=self._test_routing_sum.device
        )
        # 每个有效位置先对全部测试样本/真实 patch 求 mean weight，再减对应 1/M。
        for target_index in range(14):
            source_count = target_index + 2
            # 只填写当前 target 可访问的 source 前缀；其余位置继续保持 NaN mask。
            matrix[target_index, :source_count] = (
                self._test_routing_sum[target_index, :source_count]
                / self._test_token_count[target_index]
                - 1.0 / source_count
            )
        # 0/正/负分别表示无深度偏好、额外选择及相对抑制；不保留逐 token 数据。
        return matrix.cpu()


# 定义使用视觉 MAE backbone 与跨变量 adapter 进行时间序列预测的主模型。
class VisionTS(nn.Module):

    # 初始化视觉 backbone、pretrained checkpoint 和 Global Token adapter。
    def __init__(self, arch='mae_base', ckpt_path=None, load_ckpt=True,
                 num_latents=1, latent_dim=192, adapter_num_heads=4):
        # 调用 nn.Module 的初始化逻辑。
        super(VisionTS, self).__init__()

        # 拒绝未在架构映射表中注册的模型名。
        if arch not in MAE_ARCH:
            # 说明非法架构名及可用选项。
            raise ValueError(f"Unknown arch: {arch}. Should be in {list(MAE_ARCH.keys())}")

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

        # 图片步骤 1：Global Token 只提取并向真实 patch 广播公共成分，原功能保持不变。
        self.variable_adapter = VariableAwareLatentAdapter(
            # 使 adapter 输入维度与 MAE positional embedding 维度一致。
            embed_dim=self.vision_model.pos_embed.shape[-1],
            # 设置可学习 Global Token 数量，当前默认值为 1。
            num_latents=num_latents,
            # 设置每个 latent 的特征维度。
            latent_dim=latent_dim,
            # 设置 adapter 的 attention head 数。
            num_heads=adapter_num_heads,
        )
        # 图片步骤 8：Router 位于 frozen MAE 外部；Router 内唯一新增的可学习参数是 14 条 query。
        self.residual_router = CrossDepthResidualRouter(
            self.vision_model.pos_embed.shape[-1]
        )

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

    # 将时间序列渲染为图像，通过 MAE 重建未来区间并恢复为预测序列。
    def forward(self, x, export_image=False, fp64=False):
        # 输入 x 的 shape 为 [B, L_ctx, N]：B 是 batch size，L_ctx 是回看长度，N 是变量数。
        # fp64=True 可避免比特币等数据集上的数学溢出。
        # 默认返回 y，shape 为 [B, L_pred, N]，L_pred 是预测长度。
        # export_image=True 时还返回两个 [B, N, H, W, C] 图像，H/W/C 为高/宽/channel 数。
        # 中间 shape 中 F 是 periodicity，P_ctx 是历史周期段数，P_all 是重建 image 的周期段数。
        # L_pad 是左侧填充后的输入长度，L_all=F×P_all 是重建 image 展开后的 sequence 长度。

        # 1. 计算并切断每个样本、每个变量的时间维均值。
        means = x.mean(1, keepdim=True).detach()  # shape：[B, 1, N]。
        # 从输入中减去均值以进行中心化。
        x_enc = x - means
        # 计算每个变量的标准差，fp64 模式使用双精度以提高数值稳定性。
        stdev = torch.sqrt(
            torch.var(x_enc.to(torch.float64) if fp64 else x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)  # shape：[B, 1, N]。
        # 用配置常数调整标准差，从而控制归一化后的幅度。
        stdev /= self.norm_const
        # 用调整后的标准差缩放中心化序列。
        x_enc /= stdev
        # 将变量维移到时间维之前，使各变量在渲染和 patch embedding 时保持独立。
        x_enc = einops.rearrange(x_enc, 'b s n -> b n s')  # shape：[B, N, L_ctx]。

        # 2. 复制边界值填充序列左侧，使时间长度能按周期分组。
        x_pad = F.pad(x_enc, (self.pad_left, 0), mode='replicate')  # shape：[B, N, L_pad]，L_pad 为填充后长度。
        # 将每个变量按周期折叠成单 channel 二维图像。
        x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=self.periodicity)  # shape：[B×N, 1, F, P_ctx]。

        # 3. 将历史图像缩放到预留的输入 patch 区域。
        x_resize = self.input_resize(x_2d)  # shape：[B×N, 1, H, W_in]。
        # 创建与预测 patch 等宽的全零待重建区域。
        masked = torch.zeros((x_2d.shape[0], 1, self.image_size, self.num_patch_output * self.patch_size), device=x_2d.device, dtype=x_2d.dtype)  # shape：[B×N, 1, H, W_out]。
        # 在水平方向拼接历史图像与待重建区域。
        x_concat_with_masked = torch.cat([
            # 放入已知的历史图像。
            x_resize,
            # 放入代表未来的空白图像。
            masked
        ], dim=-1)  # shape：[B×N, 1, H, W]，W = W_in + W_out。
        # 将单 channel 序列图复制成 MAE 期望的 3-channel 图像。
        image_input = einops.repeat(x_concat_with_masked, 'b 1 h w -> b c h w', c=3)  # shape：[B×N, 3, H, W]。

        # 4. 调用 frozen MAE；Global Token 仅提取并向真实 patch 广播公共成分。
        # 返回 y [B×N, L, p²×3] 和 mask [B×N, L]；L 为 patch 数，p 为 patch 边长。
        _, y, mask = self.vision_model(
            # 传入渲染后的 3-channel 图像。
            image_input,
            # 指定需要重建的 patch 比例。
            mask_ratio=self.mask_ratio,
            # 将固定 mask 复制到当前的所有变量样本。
            noise=einops.repeat(self.mask, '1 l -> n l', n=image_input.shape[0]),
            # 传入原有 Global Token adapter；它不承担后续的 source-depth 选择。
            variable_adapter=self.variable_adapter,
            # 告知 adapter 原始 batch size。
            batch_size=x_enc.shape[0],
            # 告知 adapter 每个样本的变量数。
            num_variables=x_enc.shape[1],
            # 图片步骤 2～11：传入逐 token Router；Global Token 与 frozen MAE 其余路径保持不变。
            residual_router=self.residual_router,
        )
        # 将 MAE 输出的 patch 序列还原为完整图像。
        image_reconstructed = self.vision_model.unpatchify(y)  # shape：[B×N, 3, H, W]。

        # 5. 对重建图像的颜色 channel 取平均，恢复为灰度序列图。
        y_grey = torch.mean(image_reconstructed, 1, keepdim=True)  # shape：[B×N, 1, H, W]。
        # 将灰度图缩放回周期分段对应的尺寸。
        y_segmentations = self.output_resize(y_grey)  # shape：[B×N, 1, F, P_all]。
        # 将每个变量的二维周期分段展平回一维时间序列。
        y_flatten = einops.rearrange(
            # 传入已缩放的重建分段。
            y_segmentations,
            # 合并分段位置与周期内位置，并恢复变量维。
            '(b n) 1 f p -> b (p f) n',
            # 提供原始 batch size 以拆分合并维。
            b=x_enc.shape[0], f=self.periodicity
        )  # shape：[B, L_all, N]，L_all = F×P_all。
        # 跳过填充和历史部分，截取指定长度的未来预测窗口。
        y = y_flatten[:, self.pad_left + self.context_len: self.pad_left + self.context_len + self.pred_len, :]  # shape：[B, L_pred, N]。

        # 6. 使用每个变量的标准差恢复原始数值尺度。
        y = y * (stdev.repeat(1, self.pred_len, 1))
        # 加回每个变量的均值，完成 denormalization。
        y = y + (means.repeat(1, self.pred_len, 1))

        # 根据开关决定是否额外导出可视化图像。
        if export_image:
            # 切断 mask 与计算图的连接。
            mask = mask.detach()
            # 将 patch 级 mask 扩展到每个 patch 的所有 pixel 和三个 channel。
            mask = mask.unsqueeze(-1).repeat(1, 1, self.vision_model.patch_embed.patch_size[0]**2 *3)  # shape：[B×N, L, p²×3]。
            # 将 patch mask 还原为 image，1 表示移除，0 表示保留。
            mask = self.vision_model.unpatchify(mask)  # shape：[B×N, 3, H, W]。
            # 如需改为 channel 后置布局，可使用下面的维度变换。
            # mask = torch.einsum('nchw->nhwc', mask)
            # 保留已知输入区域，并用重建结果覆盖 mask 区域。
            image_reconstructed = image_input * (1 - mask) + image_reconstructed * mask
            # 创建数值为 -2 的背景图，用于标记待重建区域。
            green_bg = -torch.ones_like(image_reconstructed) * 2
            # 保留已知输入区域，并用背景值替换 mask 区域。
            image_input = image_input * (1 - mask) + green_bg * mask
            # 将输入图像拆回 batch 和变量维，同时将 channel 移到末维。
            image_input = einops.rearrange(image_input, '(b n) c h w -> b n h w c', b=x_enc.shape[0])  # shape：[B, N, H, W, 3]。
            # 以相同布局整理重建图像，便于可视化。
            image_reconstructed = einops.rearrange(image_reconstructed, '(b n) c h w -> b n h w c', b=x_enc.shape[0])  # shape：[B, N, H, W, 3]。
            # 同时返回预测值、mask 后输入图和重建图。
            return y, image_input, image_reconstructed
        # 不导出图像时仅返回预测序列。
        return y

    # 设置路由统计阶段；test 仅在线累计最终 14×16 deviation 矩阵。
    def set_routing_statistics_mode(self, mode):
        self.residual_router.set_statistics_mode(mode)

    # 数据完整时弹出并清空一个训练 epoch 的 14 行 routing learning 指标。
    def pop_routing_learning_records(self):
        return self.residual_router.pop_learning_records()

    # 返回测试集所有样本、所有真实 patch 聚合后的 mean deviation 矩阵。
    def routing_deviation_matrix(self):
        return self.residual_router.test_deviation_matrix()

    # 返回热力图横轴 source 和纵轴 target 的固定顺序。
    def routing_metadata(self):
        return {
            "source_names": self.residual_router.source_names,
            "target_names": self.residual_router.target_names,
        }
