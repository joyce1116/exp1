# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

# 导入 partial，用于预先固定 normalization layer 的参数。
from functools import partial

# 导入 PyTorch、神经网络模块和 NumPy 数值计算库。
import torch
import torch.nn as nn
import numpy as np

# 导入 ViT 的 patch embedding 层和 Transformer block。
from timm.models.vision_transformer import PatchEmbed, Block

# 导入二维正弦余弦 positional embedding 生成函数。
from ..utils.pos_embed import get_2d_sincos_pos_embed


# 定义以 Vision Transformer 为 backbone 的 masked autoencoder。
class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    # 初始化 MAE 的 encoder、decoder、patch 和 quantile prediction 配置。
    # img_size、patch_size、in_chans 分别指定图像边长、patch 边长和输入 channel 数。
    # embed_dim、depth、num_heads 分别指定 encoder embedding、block 和 attention head 数量。
    # decoder_embed_dim、decoder_depth、decoder_num_heads 指定 decoder 的对应结构参数。
    # mlp_ratio 与 norm_layer 指定 MLP 扩张比和 normalization layer 类型。
    # norm_pix_loss 控制 pixel normalization，quantile 与 quantile_head_num 控制 quantile heads。
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 quantile=False, quantile_head_num=9):
        # 初始化 PyTorch 模块的基类状态。
        super().__init__()

        # 保存 pixel normalization loss 开关、quantile 模式和 quantile head 数量。
        self.norm_pix_loss = norm_pix_loss
        self.quantile = quantile
        self.quantile_head_num = quantile_head_num

        # --------------------------------------------------------------------------
        # 构建 MAE encoder 的 patch embedding 层，并记录 patch 总数。
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # 创建可训练的 cls token，shape 为 (1, 1, embed_dim)。
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 创建固定的正弦余弦 positional embedding，并为 cls token 预留一个位置。
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        # 按指定深度堆叠 encoder Block；未传 drop_path/init_values，DropPath 与 LayerScale 均关闭。
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        # 对 encoder 的最终输出执行 LayerNorm。
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # 将 encoder feature 映射到 decoder embedding 维度。
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # 创建用于填充被 mask 位置的可训练 mask token。
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # 创建 decoder 使用的固定正弦余弦 positional embedding。
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)

        # decoder Block 同样使用默认 drop_path=0、init_values=None，不启用 DropPath 或 LayerScale。
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        # 对 decoder 的最终 feature 执行 LayerNorm。
        self.decoder_norm = norm_layer(decoder_embed_dim)

        # 用主 prediction head 把每个 token 还原为一个 patch 的 pixel。
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)
        # 启用 quantile prediction 时，为其余 quantile 创建独立 prediction head。
        if self.quantile:
            # 主 prediction head 负责 median，其余 head 负责其他 quantile 输出。
            self.decoder_pred_quantile_list = nn.ModuleList()
            # 按配置数量逐个添加额外的线性预测 head。
            for i in range(self.quantile_head_num-1):
                self.decoder_pred_quantile_list.append(nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True))

        # --------------------------------------------------------------------------

        # 完成所有参数的统一初始化。
        self.initialize_weights()


    # 初始化 positional embedding、patch projection 和网络中的可训练参数。
    def initialize_weights(self):
        # 生成 encoder 的固定正弦余弦 positional embedding，并复制到参数 tensor 中。
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # 生成 decoder 的固定正弦余弦 positional embedding，并复制到参数 tensor 中。
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # 展平卷积核的输入维度，按线性层方式进行 Xavier 均匀初始化。
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # 使用标准差为 0.02 的正态分布初始化 cls token 和 mask token。
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # 递归调用自定义规则初始化所有 Linear 和 LayerNorm 层。
        self.apply(self._init_weights)

    # 根据模块类型应用对应的参数初始化规则。
    def _init_weights(self, m):
        # 对线性层权重使用 Xavier 均匀初始化。
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            # 在线性层存在偏置时将偏置清零。
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # 将 LayerNorm 的偏置置零、缩放权重置一。
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # 将 batch 图像重排为由展平 patch 组成的 sequence。
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 * 3)
        """
        # 输入 imgs shape 为 [N, 3, H, W]，输出 x shape 为 [N, L, p**2*3]。
        # N 是 batch size，H/W 是图像高/宽，p 是 patch 边长，L=(H/p)*(W/p)。
        # 读取方形 patch 的边长。
        p = self.patch_embed.patch_size[0]
        # 要求输入图像为正方形，且边长能被 patch 边长整除。
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        # 计算图像在高、宽方向上的 patch 数量。
        h = w = imgs.shape[2] // p
        # 将图像拆分为高宽两个 patch 网格维和两个 patch 内部维度。
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        # 调整维度顺序，使 patch 网格位置位于 channel 和 patch 内 pixel 之前。
        x = torch.einsum('nchpwq->nhwpqc', x)
        # 将每个 patch 展平，并合并二维网格为 sequence 维。
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        # 返回 patch sequence。
        return x

    # 将展平的 patch sequence 重新拼接成 batch 图像。
    def unpatchify(self, x, n_channels=3):
        """
        x: (N, L, patch_size**2 * n_channels)
        imgs: (N, 3, H, W)
        """
        # 输入 x shape 为 [N, L, p**2*C]，输出 imgs shape 为 [N, C, h*p, h*p]。
        # N 是 batch size，C=n_channels，p 是 patch 边长，h=sqrt(L) 是方形 patch 网格边长。
        # 读取方形 patch 的边长。
        p = self.patch_embed.patch_size[0]
        # 根据 sequence 长度推算正方形 patch 网格的边长。
        h = w = int(x.shape[1]**.5)
        # 确认 sequence 长度确实能构成方形网格。
        assert h * w == x.shape[1]

        # 恢复网格、patch 内部高宽和 channel 维。
        x = x.reshape(shape=(x.shape[0], h, w, p, p, n_channels))
        # 将 channel 维移到网格及 patch 内部空间维之前。
        x = torch.einsum('nhwpqc->nchpwq', x)
        # 合并网格与 patch 内部维度，得到完整图像。
        imgs = x.reshape(shape=(x.shape[0], n_channels, h * p, h * p))
        # 返回重建后的图像 tensor。
        return imgs

    # 按样本随机打乱 patch，并保留指定比例的 patch。
    def random_masking(self, x, mask_ratio, noise=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        noise: [N, L]
        """
        # N 是 batch size，L 是 patch 数，D 是 embedding 维度。
        # 返回 x_masked [N, L_keep, D]、mask [N, L] 和 ids_restore [N, L]。
        # 分别取得 batch size、sequence 长度和 embedding 维度。
        N, L, D = x.shape
        # 根据 mask 比例计算要保留的 patch 数量。
        len_keep = int(round(L * (1 - mask_ratio)))

        # 未传入固定 noise 时，为每个样本和位置生成独立随机数。
        if noise is None:
            noise = torch.rand(N, L, device=x.device)

        # 按 noise 升序得到 shuffle 索引，较小值对应保留位置。
        ids_shuffle = torch.argsort(noise, dim=1)
        # 再次排序得到可恢复原始顺序的逆 permutation 索引。
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # 取打乱后序列的前 len_keep 个位置作为保留索引。
        ids_keep = ids_shuffle[:, :len_keep]
        # 沿 sequence 维收集所有被保留的 patch feature。
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # 在 shuffle 后的顺序中创建二值 mask，其中 0 表示保留、1 表示移除。
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # 使用逆 permutation 把二值 mask 恢复到原始 patch 顺序。
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # 返回保留 feature、原顺序 mask 以及顺序恢复索引。
        return x_masked, mask, ids_restore

    # 图片步骤 5～6：完整执行 LayerNorm1→Attention，在原 residual add 前取得 raw Attention 输出。
    @staticmethod
    def _attention_output(block, x):
        # block.attn 会完整走完 QKV、softmax、多头合并和 output projection，不在内部截断。
        update = block.attn(block.norm1(x))
        # 当前构造的 init_values=None，因此 timm 的 ls1 是 Identity，不产生 LayerScale 缩放。
        if getattr(block, "ls1", None) is not None:
            update = block.ls1(update)
        # 返回完整 Attention transformation 的最终输出，尚未与当前 hidden state 做 residual add。
        return update

    # 当前 drop_path=0；这里只保留拆分前 Block 的分支调用顺序，实际不做随机丢弃。
    @staticmethod
    def _attention_drop_path(block, update):
        # 优先读取 drop_path1，旧版才回退到共享 drop_path；当前实际模块为 Identity。
        drop_path = getattr(block, "drop_path1", getattr(block, "drop_path", None))
        # None 或 Identity 都保持 update 不变，因此 residual 分支没有启用 DropPath。
        return update if drop_path is None else drop_path(update)

    # 图片步骤 5～6：完整执行 LayerNorm2→MLP，在原 residual add 前取得 raw MLP 输出。
    @staticmethod
    def _mlp_output(block, x):
        # block.mlp 会完整走完两层 FFN、激活和内部 projection，不在 MLP 内部截断。
        update = block.mlp(block.norm2(x))
        # 当前构造的 init_values=None，因此 timm 的 ls2 是 Identity，不产生 LayerScale 缩放。
        if getattr(block, "ls2", None) is not None:
            update = block.ls2(update)
        # 返回完整 MLP transformation 的最终输出，尚未与当前 hidden state 做 residual add。
        return update

    # 当前 drop_path=0；这里只保留拆分前 Block 的分支调用顺序，实际不做随机丢弃。
    @staticmethod
    def _mlp_drop_path(block, update):
        # 优先读取 drop_path2，旧版才回退到共享 drop_path；当前实际模块为 Identity。
        drop_path = getattr(block, "drop_path2", getattr(block, "drop_path", None))
        # None 或 Identity 都保持 update 不变，因此 residual 分支没有启用 DropPath。
        return update if drop_path is None else drop_path(update)

    # 图片步骤 7：去掉 CLS，把 [B*V,P,D] 恢复并展平为真实 token 布局 [B,V*P,D]。
    @staticmethod
    def _real_patch_output(output, batch_size, num_variables):
        # 这里只重排真实 patch，不跨 patch 求均值、std 或 sample-level signature。
        return output[:, 1:, :].reshape(
            batch_size, num_variables, output.shape[1] - 1, output.shape[2]
        ).flatten(1, 2)

    # 把 Router 输出的 [B,V*P,D] correction 还原为 MAE 主干使用的 [B*V,P,D]。
    @staticmethod
    def _restore_patch_output(output, batch_size, num_variables):
        return output.reshape(
            batch_size * num_variables, -1, output.shape[-1]
        )

    def prepare_visible_patches(self, x, mask_ratio, noise=None):
        # 将输入图像映射为 patch embedding sequence。
        x = self.patch_embed(x)
        # 为 patch 加入 positional embedding，不使用 cls token 对应的位置。
        x = x + self.pos_embed[:, 1:, :]
        # 按统一 mask permutation 取得所有真实可见 patch。
        return self.random_masking(x, mask_ratio, noise)

    # 对 patch 执行 encoder，并返回 latent、mask 和顺序恢复索引。
    def forward_encoder(self, x, mask_ratio, noise=None, variable_adapter=None,
                        batch_size=None, num_variables=None,
                        residual_router=None, variable_correction=None):
        # 输入 x shape 为 [N, C, H, W]，可选 noise shape 为 [N, L]。
        # 返回 latent [N, 1+L_keep, D]、mask [N, L] 和 ids_restore [N, L]。
        # N 是图像 batch size，C/H/W 是 channel/高/宽，L 与 L_keep 是总 patch 数与保留 patch 数。
        # D 是 encoder embedding 维度；启用 variable adapter 时 N=B*V，B/V 是原 batch size/变量数。
        # residual_router 为 None 时执行原始 MAE；非空时按图片步骤 2～11 路由 Block 6～12。
        x, mask, ids_restore = self.prepare_visible_patches(
            x, mask_ratio, noise
        )

        # 图片步骤 1：Global Token 在 MAE Transformer 前提取并广播真实 patch 的公共成分。
        if variable_adapter is not None:
            # 使用必需的 batch_size=B 和 num_variables=V，将 [N,L_keep,D] reshape 为 [B,V,L_keep,D]。
            x = x.reshape(batch_size, num_variables, x.shape[1], x.shape[2])
            # 分块模式复用完整变量共同得到的 correction；普通模式保持原 adapter forward。
            if variable_correction is None:
                x = variable_adapter(x)
            else:
                x = variable_adapter.apply_correction(
                    x, variable_correction
                )
            # 重新合并 batch 与 variable 轴，恢复为 [B*V,L_keep,D]。
            x = x.reshape(batch_size * num_variables, x.shape[2], x.shape[3])

        # 为 cls token 加入其专属 positional embedding。
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        # 将 cls token 扩展到当前 batch 中的每个样本。[B×V,1,768]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # 把 cls token 拼接到 patch sequence 开头。[B×V,1+P,768]
        x = torch.cat((cls_tokens, x), dim=1)

        # 依次通过所有 encoder Transformer block。
        if residual_router is None:
            # 未提供路由器时保持原始 MAE encoder 路径，直接调用每个完整 Block。
            for blk in self.blocks:
                x = blk(x)
        else:
            # 图片步骤 3：Block 1～4 完整正常前向，不参与 Router，也不保存 residual source。
            for block_index in range(4):
                x = self.blocks[block_index](x)

            # 图片步骤 5：Block 5 是 source-only Block，不执行 depth routing 或 correction。
            block = self.blocks[4]
            # 它只产生供 Block 6 以后使用的 D5-A/D5-M，不读取任何 residual source。
            source_values = []
            source_keys = []
            # 以 x0 完整执行 LayerNorm1→Attention，在 x1=x0+D5-A 前保存 raw D5-A。
            attention_output = self._attention_output(block, x)
            # 去掉 CLS 后，D5-A 的真实 patch source 形状为 [B,V*P,D]。
            source = self._real_patch_output(
                attention_output, batch_size, num_variables
            )
            # 保存 raw D5-A 作为 Value，并只为 Key 做无参数逐 token RMSNorm。
            source_values.append(source)
            source_keys.append(residual_router.make_key(source))
            # DropPath 当前关闭；按原 frozen MAE 正常执行 x1=x0+D5-A。
            x = x + self._attention_drop_path(block, attention_output)
            # 再以 x1 完整执行 LayerNorm2→MLP，并在 x2=x1+D5-M 前保存 raw D5-M。
            mlp_output = self._mlp_output(block, x)
            # 去掉 CLS 后，D5-M 同样以 [B,V*P,D] 保存，且不覆盖 D5-A。
            source = self._real_patch_output(
                mlp_output, batch_size, num_variables
            )
            # 保存 raw D5-M Value 及其逐 token RMSNorm Key，供 Block 6～12 使用。
            source_values.append(source)
            source_keys.append(residual_router.make_key(source))
            # DropPath 当前关闭；按原 frozen MAE 正常执行 x2=x1+D5-M，得到 H5。
            x = x + self._mlp_drop_path(block, mlp_output)

            # 图片步骤 6/8：14 个 target 依次为 B6-A、B6-M、…、B12-A、B12-M。
            target_index = 0
            # Block 6～12 均重复“旧 source 路由→修正 sublayer 输入→保存 raw 输出→原 residual add”。
            for block_index in range(5, 12):
                # Python 下标 5～11 对应人类编号 Block 6～12，先处理当前 Block-Attention。
                block = self.blocks[block_index]
                # 图片步骤 8～11：Bk-A 先仅用此前 source 计算逐 token 输入侧 correction。
                depth_correction, routing_weights = residual_router.route(
                    target_index, source_values, source_keys
                )
                # 将 [B,V*P,D] correction 恢复为 [B*V,P,D]，但不改写原 residual stream。
                input_correction = self._restore_patch_output(
                    depth_correction, batch_size, num_variables
                )
                # CLS correction 严格为 0，仅真实 patch 读取历史 depth information。
                input_correction = torch.cat((
                    torch.zeros_like(x[:, :1, :]), input_correction
                ), dim=1)
                # 图片步骤 11：correction 只进入 LayerNorm1→Frozen Attention 的输入。
                attention_output = self._attention_output(
                    block, x + input_correction
                )
                # 完整 Attention 输出在 residual add 前保存为 raw source，供后续 target 使用。
                source = self._real_patch_output(
                    attention_output, batch_size, num_variables
                )
                source_key = residual_router.make_key(source)
                # 当前 raw source 产生后再记录原统计；它未参与自己的 routing weights。
                residual_router.record_statistics(
                    target_index, routing_weights, depth_correction, source
                )
                # 原 identity residual 保留：x1=原 x+Attention(LayerNorm1(x+correction))。
                x = x + self._attention_drop_path(block, attention_output)
                # 当前 raw Bk-A 从此进入 source pool，仅供后续 MLP/Block 使用。
                source_values.append(source)
                source_keys.append(source_key)
                # raw Attention source 不被覆盖；target_index 推进到同一 Block 的 MLP query。
                target_index += 1

                # Bk-M 可读取刚保存的 Bk-A，但仍不读取尚未产生的自身 MLP 输出。
                depth_correction, routing_weights = residual_router.route(
                    target_index, source_values, source_keys
                )
                # 将 patch correction 恢复到 MAE 布局，不把它写入当前 residual hidden state。
                input_correction = self._restore_patch_output(
                    depth_correction, batch_size, num_variables
                )
                # CLS 不参与 depth routing，因此在输入侧 correction 前补一个全零位置。
                input_correction = torch.cat((
                    torch.zeros_like(x[:, :1, :]), input_correction
                ), dim=1)
                # 图片步骤 11：correction 只进入 LayerNorm2→Frozen MLP 的输入。
                mlp_output = self._mlp_output(
                    block, x + input_correction
                )
                # 完整 MLP 输出在 residual add 前保存为 raw source，供后续 target 使用。
                source = self._real_patch_output(
                    mlp_output, batch_size, num_variables
                )
                source_key = residual_router.make_key(source)
                # MLP raw source 产生后再完成统计，correction_ratio 定义保持不变。
                residual_router.record_statistics(
                    target_index, routing_weights, depth_correction, source
                )
                # 原 identity residual 保留：x2=原 x+MLP(LayerNorm2(x+correction))。
                x = x + self._mlp_drop_path(block, mlp_output)
                # 当前 raw Bk-M 从此进入 source pool，仅供下一 Attention/MLP 使用。
                source_values.append(source)
                source_keys.append(source_key)
                # raw MLP source 保持不变；target_index 推进到下一 Block 的 Attention query。
                target_index += 1

            # 图片仅定义到 Block 12；large/huge 的 Block 13 以后恢复原始完整 frozen Block。
            for block_index in range(12, len(self.blocks)):
                x = self.blocks[block_index](x)
        # 对 encoder 输出执行最终 LayerNorm。
        x = self.norm(x)

        # 返回 latent 及 decoder 所需的 mask 信息。
        return x, mask, ids_restore

    # 恢复被 mask 的 sequence 位置，并将 encoder feature 解码为 patch 预测。
    def forward_decoder(self, x, ids_restore):
        # 输入 x 为 [N, 1+L_keep, D_enc]，N 是 batch size，L_keep 是保留 patch 数。
        # ids_restore 为 [N, L]，D_enc 是 encoder embedding 维度，L 是完整 patch 数。
        # point 模式返回 tensor [N,L,p**2*C]；p/C 分别是 patch 边长和 channel 数。
        # quantile 模式返回 (x_mid, x_quantile_list)：x_mid shape 为 [N,L,p**2*C]。
        # x_quantile_list 长度为 quantile_head_num-1，其中每个 tensor 的 shape 同 x_mid。
        # 将 encoder token 映射到 decoder embedding 维度。
        x = self.decoder_embed(x)

        # 按被 mask 的 patch 数量复制 mask token。
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        # 去掉 cls token 后，将保留 token 与 mask token 拼接成完整长度。
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        # 根据逆 permutation 索引把所有 patch token 恢复到原始空间顺序。
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        # 将 cls token 重新拼接到恢复后的 sequence 开头。
        x = torch.cat([x[:, :1, :], x_], dim=1)

        # 为完整 decoder sequence 加入固定 positional embedding。
        x = x + self.decoder_pos_embed

        # 依次通过所有 decoder Transformer block。
        for blk in self.decoder_blocks:
            x = blk(x)
        # 对 decoder 输出执行最终 LayerNorm。
        x = self.decoder_norm(x)

        # 未启用 quantile 模式时，只通过主 prediction head 生成 point prediction。
        if not self.quantile:
            x = self.decoder_pred(x)
            # 移除不对应 patch 的 cls token 预测。
            x = x[:, 1:, :]
            # 返回所有 patch 的 pixel prediction。
            return x
        else:
            # 使用主 prediction head 计算 50% quantile，并移除 cls token 位置。
            x_mid = self.decoder_pred(x)[:, 1:, :]

            # 创建列表以收集其余 quantile prediction。
            x_quantile_list = []
            # 依次调用各个额外预测 head，并移除 cls token 位置。
            for i in range(self.quantile_head_num-1):
                x_quantile = self.decoder_pred_quantile_list[i](x)[:, 1:, :]
                x_quantile_list.append(x_quantile)

            # 分别返回 median prediction 和其他 quantile prediction 列表。
            return x_mid, x_quantile_list

    # 计算被 mask patch 上的 pixel mean squared error loss。
    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        # N 是 batch size，H/W 是图像高/宽，L 是 patch 数，p 是 patch 边长；返回 scalar loss。
        # 将原始图像转换成与预测相同布局的 patch sequence。
        target = self.patchify(imgs)
        # 开启 pixel normalization 时，对每个 patch 独立做 normalization。
        if self.norm_pix_loss:
            # 计算每个 patch 内 pixel 的均值和方差。
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            # 使用带微小稳定项的标准差对目标 pixel 做 normalization。
            target = (target - mean) / (var + 1.e-6)**.5

        # 计算 prediction pixel 与 target pixel 之间的逐元素平方 error。
        loss = (pred - target) ** 2
        # 对每个 patch 内部的 pixel error 取平均。
        loss = loss.mean(dim=-1)

        # 仅汇总被 mask patch 的 loss，并按其数量求平均。
        loss = (loss * mask).sum() / mask.sum()
        # 返回 scalar reconstruction loss。
        return loss

    # 执行完整 forward，依次经过 encoder、decoder 并返回预测和 mask。
    def forward(self, imgs, mask_ratio=0.75, noise=None, variable_adapter=None,
                batch_size=None, num_variables=None, residual_router=None,
                variable_correction=None):
        # 输入 imgs shape 为 [N,C,H,W]，可选 noise shape 为 [N,L]；N/C/H/W 是 batch/channel/高/宽。
        # 启用 variable adapter 时 N=B*V，B/V 分别是原 batch size 和变量数。
        # point 模式返回 (None, pred, mask)，pred [N,L,p**2*C]，mask [N,L]。
        # quantile 模式的 pred 为 (x_mid, x_quantile_list)，x_mid shape 为 [N,L,p**2*C]。
        # x_quantile_list 内每个 tensor 的 shape 同 x_mid，列表长度为 quantile_head_num-1。
        # L 是 patch 数，p 是 patch 边长，C 是 channel 数。
        # residual_router 是外层创建的逐 token 深度 Router；为 None 时不改变原 MAE。
        # 通过 encoder 处理输入图像，并获取 mask 及恢复原序所需索引。
        latent, mask, ids_restore = self.forward_encoder(
            imgs, mask_ratio, noise, variable_adapter, batch_size, num_variables,
            # 仅把 Router 透传给 encoder；Decoder、decoder_norm 和 decoder_pred 保持原 frozen 路径。
            residual_router, variable_correction
        )
        # 通过 decoder 将 latent 转换为每个 patch 的 pixel 或 quantile prediction。
        pred = self.forward_decoder(latent, ids_restore)
        # 当前接口不在内部计算 loss，以空值占位并返回预测和 mask。
        return None, pred, mask


# 构建采用 16-pixel patch 的基础规模 MAE 模型。
def mae_vit_base_patch16_dec512d8b(**kwargs):
    # 选择标准 LayerNorm 作为 normalization layer 类型。
    norm = nn.LayerNorm
    # 使用基础规模 encoder 和 512 维、8 层 decoder 配置实例化模型。
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(norm, eps=1e-6), **kwargs)
    # 返回构建完成的基础规模模型。
    return model


# 构建采用 16-pixel patch 的大规模 MAE 模型。
def mae_vit_large_patch16_dec512d8b(**kwargs):
    # 使用大规模 encoder 和 512 维、8 层 decoder 配置实例化模型。
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    # 返回构建完成的大规模模型。
    return model


# 构建采用 14-pixel patch 的超大规模 MAE 模型。
def mae_vit_huge_patch14_dec512d8b(**kwargs):
    # 使用超大规模 encoder 和 512 维、8 层 decoder 配置实例化模型。
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    # 返回构建完成的超大规模模型。
    return model


# 为三种推荐架构提供更简短的公开别名。
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b
