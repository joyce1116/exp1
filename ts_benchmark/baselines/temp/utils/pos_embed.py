# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

# 导入 NumPy，用于生成位置网格并计算正弦余弦 positional embedding。
import numpy as np

# 导入 PyTorch，用于对 checkpoint 中的 positional embedding 执行 interpolation。
import torch

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    # grid_size 是方形 patch 网格边长，embed_dim 是每个位置的 embedding 维度。
    # 返回 [grid_size**2, embed_dim]；包含 cls token 时第一个维度额外加一。
    # 分别生成高度方向和宽度方向的浮点坐标。
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    # 以宽度坐标在前的顺序生成二维坐标网格。
    grid = np.meshgrid(grid_w, grid_h)
    # 将宽、高两个坐标网格堆叠到新的首维。
    grid = np.stack(grid, axis=0)

    # 补充单例维度，使坐标网格符合下游函数的输入布局。
    grid = grid.reshape([2, 1, grid_size, grid_size])
    # 基于二维坐标网格计算每个位置的正弦余弦 positional embedding。
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    # 需要 cls token 时，在 positional embedding 开头拼接一个全零向量。
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    # 返回完整的二维 positional embedding 矩阵。
    return pos_embed


# 分别编码二维网格的高、宽坐标并合并结果。
def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    # 输入 grid shape 为 [2, 1, H, W]，2 表示两个坐标轴；输出为 [H*W, D]。
    # D 即 embed_dim，要求 embedding 维度能被高、宽两个方向平均分配。
    assert embed_dim % 2 == 0

    # 使用一半 embedding 维度编码第一个坐标网格。
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    # 使用另一半 embedding 维度编码第二个坐标网格。
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    # 沿特征维拼接两个方向的编码，形成完整二维编码。
    emb = np.concatenate([emb_h, emb_w], axis=1)
    # 返回 shape 为 (H*W, D) 的 positional embedding。
    return emb


# 为任意一维位置坐标生成正弦余弦 positional embedding。
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    # M 是待编码位置数，D 即 embed_dim；返回 shape 为 (M, D) 的 embedding。
    # 要求输出维度可平均分给正弦和余弦两部分。
    assert embed_dim % 2 == 0
    # 为一半特征维度生成从零开始的频率索引。
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    # 将频率索引归一化到相对于半维度的比例。
    omega /= embed_dim / 2.
    # 按 Transformer positional embedding 公式计算各维度的频率。
    omega = 1. / 10000**omega

    # 将任意形状的位置网格展平为长度为 M 的向量。
    pos = pos.reshape(-1)
    # 计算位置向量与频率向量的外积，得到每个位置的相位。
    out = np.einsum('m,d->md', pos, omega)

    # 分别计算相位的正弦值和余弦值。
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    # 沿特征维拼接正弦与余弦结果，形成完整编码。
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    # 返回 shape 为 (M, D) 的 positional embedding。
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    # 输入 checkpoint positional embedding 为 [1, T_old, D]，T_old 是原 token 数，D 是 embedding 维度。
    # 函数原地更新为 [1, T_new, D]，T_new 是当前模型所需 token 数，不返回新对象。
    # 仅当 checkpoint 包含 positional embedding 时才执行尺寸适配。
    if 'pos_embed' in checkpoint_model:
        # 读取 checkpoint 的 positional embedding tensor。
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        # 取得每个 positional embedding 向量的 feature 维度。
        embedding_size = pos_embed_checkpoint.shape[-1]
        # 读取当前模型需要的 patch 数量。
        num_patches = model.patch_embed.num_patches
        # 计算 cls token 等不属于 patch 的 extra token 数量。
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # 根据 checkpoint 的 patch 数量推算原方形网格边长。
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # 根据当前模型的 patch 数量推算目标方形网格边长。
        new_size = int(num_patches ** 0.5)
        # 仅当新旧网格尺寸不同时执行 positional embedding interpolation。
        if orig_size != new_size:
            # 输出 positional embedding 从原尺寸调整到目标尺寸的提示信息。
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            # 单独保留 cls token 和 distillation token 等 extra token 的 embedding。
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # 截取真正对应 patch 网格的 positional embedding。
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            # 恢复二维网格并把 feature 维移到 channel 位置，以适配 image interpolation 接口。
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            # 使用 bicubic interpolation 将 positional embedding 网格缩放到目标尺寸。
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            # 将特征维移回末尾，并把二维网格重新展平为序列。
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            # 把保持不变的 extra token 与 interpolation 后的 patch token embedding 重新拼接。
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            # 用适配后的结果更新 checkpoint 中的 positional embedding。
            checkpoint_model['pos_embed'] = new_pos_embed
