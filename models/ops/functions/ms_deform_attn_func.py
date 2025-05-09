# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

import MultiScaleDeformableAttention as MSDA


class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


# def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
#     # for debug and test only,
#     # need to use cuda version instead
#     N_, S_, M_, D_ = value.shape
#     _, Lq_, M_, L_, P_, _ = sampling_locations.shape
#     value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
#     sampling_grids = 2 * sampling_locations - 1
#     sampling_value_list = []
#     for lid_, (H_, W_) in enumerate(value_spatial_shapes):
#         # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
#         value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
#         # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
#         sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
#         # N_*M_, D_, Lq_, P_
#         sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
#                                           mode='bilinear', padding_mode='zeros', align_corners=False)
#         sampling_value_list.append(sampling_value_l_)
#     # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
#     attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
#     output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
#     return output.transpose(1, 2).contiguous()

def slice_value(value, value_spatial_shapes):
    # 定义输入张量 value
    # value = torch.randn(2, 10723, 8, 32)

    # 定义 value_spatial_shapes 张量
    # value_spatial_shapes = torch.tensor([[76, 106], [38, 53], [19, 27], [10, 14]], device='cuda:0')

    # 计算每个切片的大小（在第1维上的元素数量）
    split_sizes = [H_ * W_ for H_, W_ in value_spatial_shapes]

    # 初始化一个列表来存储拆分后的张量
    value_list = []

    # 初始化起始索引
    start_idx = 0

    # 使用 for 循环来拆分张量
    for size in split_sizes:
        # 计算结束索引
        end_idx = start_idx + size
        
        # 根据起始和结束索引拆分张量
        # print(f"Split {start_idx}:{end_idx}")
        split_value = value[:, start_idx:end_idx, :, :]
        
        # 将拆分后的张量添加到列表中
        value_list.append(split_value)
        
        # 更新起始索引
        start_idx = end_idx

    # 输出拆分后的张量列表
    # for i, split in enumerate(value_list):
    #     print(f"Split {i+1}: shape={split.shape}")
    return value_list


def ms_deform_attn_core_pytorch_kk(sampling_locations, attention_weights, value_list_f1, value_list_f2, value_list_f3, value_list_f4):
    # for debug and test only,
    # need to use cuda version instead
    _, D_, _, _ = value_list_f1.shape
    N_, Lq_, M_, L_, P_, _ = sampling_locations.shape

    # print("value shape:", value.shape) # [2, 10723, 8, 32]
    print("sampling_locations shape:", sampling_locations.shape) # [2, 10723, 8, 4, 4, 2] onnx第一输入
    print("attention_weights shape:", attention_weights.shape) # [2, 10723, 8, 4, 4] onnx第二输入

    # split 的过程
    # value_list[0] shape: torch.Size([2, 8056, 8, 32])
    # value_list[1] shape: torch.Size([2, 2014, 8, 32])
    # value_list[2] shape: torch.Size([2, 513, 8, 32])
    # value_list[3] shape: torch.Size([2, 140, 8, 32])
    # 分别进行reshape之后的4个结果作为 onnx的剩余4个输入
    
    # value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    # value_list = slice_value(value, value_spatial_shapes)
    # import pdb;pdb.set_trace()
    value_list = [value_list_f1, value_list_f2, value_list_f3, value_list_f4]
    
    # print("value_list length:", len(value_list))
    # for i, v in enumerate(value_list):
    #     # import pdb;pdb.set_trace()
    #     print(f"value_list[{i}] shape:", v.shape)

    sampling_grids = 2 * sampling_locations - 1
    # print("sampling_grids shape:", sampling_grids.shape) # [2, 10723, 8, 4, 4, 2]

    sampling_value_list = []
    # for lid_, (H_, W_) in enumerate(value_spatial_shapes):
    for lid_ in range(len(value_list)):
        # import pdb;pdb.set_trace()
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        # value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        value_l_ = value_list[lid_]
        # print(f"value_l_[{lid_}] shape:", value_l_.shape)
        # [16, 32, 76, 106]
        # [16, 32, 38, 53]
        # [16, 32, 19, 27]
        # [16, 32, 10, 14]

        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # print(f"sampling_grid_l_[{lid_}] shape:", sampling_grid_l_.shape)
        # [16, 10723, 4, 2]
        # [16, 10723, 4, 2]
        # [16, 10723, 4, 2]
        # [16, 10723, 4, 2]

        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, mode='bilinear', padding_mode='zeros', align_corners=False)
        # print(f"sampling_value_l_[{lid_}] shape:", sampling_value_l_.shape)
        # [16, 32, 10723, 4]
        # [16, 32, 10723, 4]
        # [16, 32, 10723, 4]
        # [16, 32, 10723, 4]

        sampling_value_list.append(sampling_value_l_)

    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_) # # [2, 356, 8, 4, 4] -> (16, 1, 356, 16)
    # print("attention_weights shape:", attention_weights.shape)
    # [16, 1, 10723, 16]

    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    # import pdb;pdb.set_trace()
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    print("output shape:", output.shape)
    # [2, 256, 10723]

    return output.transpose(1, 2).contiguous()


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape

    print("value shape:", value.shape) # [2, 10723, 8, 32]
    print("sampling_locations shape:", sampling_locations.shape) # [2, 10723, 8, 4, 4, 2] onnx第一输入
    print("attention_weights shape:", attention_weights.shape) # [2, 10723, 8, 4, 4] onnx第二输入

    # split 的过程
    # value_list[0] shape: torch.Size([2, 8056, 8, 32])
    # value_list[1] shape: torch.Size([2, 2014, 8, 32])
    # value_list[2] shape: torch.Size([2, 513, 8, 32])
    # value_list[3] shape: torch.Size([2, 140, 8, 32])
    # 分别进行reshape之后的4个结果作为 onnx的剩余4个输入
    
    # value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    value_list = slice_value(value, value_spatial_shapes)
    # import pdb;pdb.set_trace()
    
    # print("value_list length:", len(value_list))
    for i, v in enumerate(value_list):
        # import pdb;pdb.set_trace()
        print(f"value_list[{i}] shape:", v.shape)

    sampling_grids = 2 * sampling_locations - 1
    # print("sampling_grids shape:", sampling_grids.shape) # [2, 10723, 8, 4, 4, 2]

    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        # print(f"value_l_[{lid_}] shape:", value_l_.shape)
        # [16, 32, 76, 106]
        # [16, 32, 38, 53]
        # [16, 32, 19, 27]
        # [16, 32, 10, 14]

        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # print(f"sampling_grid_l_[{lid_}] shape:", sampling_grid_l_.shape)
        # [16, 10723, 4, 2]
        # [16, 10723, 4, 2]
        # [16, 10723, 4, 2]
        # [16, 10723, 4, 2]

        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, mode='bilinear', padding_mode='zeros', align_corners=False)
        # print(f"sampling_value_l_[{lid_}] shape:", sampling_value_l_.shape)
        # [16, 32, 10723, 4]
        # [16, 32, 10723, 4]
        # [16, 32, 10723, 4]
        # [16, 32, 10723, 4]

        sampling_value_list.append(sampling_value_l_)

    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_) # [2, 356, 8, 4, 4] -> (16, 1, 356, 16)
    # print("attention_weights shape:", attention_weights.shape)
    # [16, 1, 10723, 16]
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    # import pdb;pdb.set_trace()
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    print("output shape:", output.shape)
    # [2, 256, 10723]

    return output.transpose(1, 2).contiguous()