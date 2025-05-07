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

import time
import torch
import torch.nn as nn
from torch.autograd import gradcheck

from functions.ms_deform_attn_func import MSDeformAttnFunction, ms_deform_attn_core_pytorch, slice_value, ms_deform_attn_core_pytorch_kk

N, M, D = 2, 8, 32
shapes = torch.as_tensor([(48, 120), (24, 60), (12, 30), (6, 15)], dtype=torch.long).cuda()
Lq, L, P = 356, 4, 4
level_start_index = torch.cat((shapes.new_zeros((1, )), shapes.prod(1).cumsum(0)[:-1]))
S = sum([(H*W).item() for H, W in shapes])
torch.manual_seed(3)


@torch.no_grad()
def check_forward_equal_with_pytorch_double():
    value = torch.rand(N, S, M, D).cuda() * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    im2col_step = 64

    output_pytorch = ms_deform_attn_core_pytorch(value.double(), shapes, sampling_locations.double(), attention_weights.double()).detach().cpu()
    output_cuda = MSDeformAttnFunction.apply(value.double(), shapes, level_start_index, sampling_locations.double(), attention_weights.double(), im2col_step).detach().cpu()
    fwdok = torch.allclose(output_cuda, output_pytorch)
    max_abs_err = (output_cuda - output_pytorch).abs().max()
    max_rel_err = ((output_cuda - output_pytorch).abs() / output_pytorch.abs()).max()

    print(f'* {fwdok} check_forward_equal_with_pytorch_double: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}')


@torch.no_grad()
def check_forward_equal_with_pytorch_float():
    value = torch.rand(N, S, M, D).cuda() * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    im2col_step = 2
    output_pytorch = ms_deform_attn_core_pytorch(value, shapes, sampling_locations, attention_weights).detach().cpu()
    output_cuda = MSDeformAttnFunction.apply(value, shapes, level_start_index, sampling_locations, attention_weights, im2col_step).detach().cpu()
    fwdok = torch.allclose(output_cuda, output_pytorch, rtol=1e-2, atol=1e-3)
    max_abs_err = (output_cuda - output_pytorch).abs().max()
    max_rel_err = ((output_cuda - output_pytorch).abs() / output_pytorch.abs()).max()

    print(f'* {fwdok} check_forward_equal_with_pytorch_float: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}')


def check_gradient_numerical(channels=4, grad_value=True, grad_sampling_loc=True, grad_attn_weight=True):

    value = torch.rand(N, S, M, channels).cuda() * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    im2col_step = 2
    func = MSDeformAttnFunction.apply

    value.requires_grad = grad_value
    sampling_locations.requires_grad = grad_sampling_loc
    attention_weights.requires_grad = grad_attn_weight

    gradok = gradcheck(func, (value.double(), shapes, level_start_index, sampling_locations.double(), attention_weights.double(), im2col_step))

    print(f'* {gradok} check_gradient_numerical(D={channels})')

import pandas as pd
import numpy as np
import onnxruntime

class MSDeformAttnCorePyTorch(nn.Module):
    def __init__(self):
        super(MSDeformAttnCorePyTorch, self).__init__()

    def forward(self, value, shapes, sampling_locations, attention_weights):
        return ms_deform_attn_core_pytorch(value, shapes, sampling_locations, attention_weights)

class MSDeformAttnCorePyTorch_Split(nn.Module):
    def __init__(self):
        super(MSDeformAttnCorePyTorch_Split, self).__init__()

    def forward(self, sampling_locations, attention_weights, value_list_f1, value_list_f2, value_list_f3, value_list_f4):
        return ms_deform_attn_core_pytorch_kk(sampling_locations, attention_weights, value_list_f1, value_list_f2, value_list_f3, value_list_f4)


@torch.no_grad()
def check_forward_equal_with_pytorch_double_onnx():
    value = torch.rand(N, S, M, D).cuda() * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    im2col_step = 64
    # 打印输入张量的形状
    print("value shape:", value.shape)
    print("shapes:", shapes)  
    print("sampling_locations shape:", sampling_locations.shape)
    print("attention_weights shape:", attention_weights.shape)

    # 第一 PyTorch 实现
    output_pytorch = ms_deform_attn_core_pytorch(value.double(), shapes, sampling_locations.double(), attention_weights.double()).detach().cpu()
    
    # 第二 CUDA 实现
    output_cuda = MSDeformAttnFunction.apply(value.double(), shapes, level_start_index, sampling_locations.double(), attention_weights.double(), im2col_step).detach().cpu()

    # 第三 导出 ONNX 模型
    model = MSDeformAttnCorePyTorch()
    model.eval()
    torch.onnx.export(model, (value.double(), shapes, sampling_locations.double(), attention_weights.double()), 'ms_deform_attn_core_pytorch.onnx', opset_version=20)
    # 加载 ONNX 模型并运行
    sess = onnxruntime.InferenceSession('ms_deform_attn_core_pytorch.onnx')
    inputs = {
        'value': value.double().cpu().numpy(),
        'value_spatial_shapes': np.array(shapes.cpu(), dtype=np.int64),
        'sampling_locations': sampling_locations.double().cpu().numpy(),
        'attention_weights': attention_weights.double().cpu().numpy()
    }
    ort_output = sess.run(None, inputs)[0]

    # 第四 拆分输入实现
    # value_list = slice_value(value.double(), shapes)
    value_list = value.double().split([H_ * W_ for H_, W_ in shapes], dim=1)
    value_list_f1 = value_list[0]
    value_list_f2 = value_list[1]
    value_list_f3 = value_list[2]
    value_list_f4 = value_list[3]
    output_pytorch_v1 = ms_deform_attn_core_pytorch_kk(sampling_locations.double(), attention_weights.double(), value_list_f1, value_list_f2, value_list_f3, value_list_f4, shapes).detach().cpu()

    # 第五 拆分输入再进行onnx导出 实现
    model = MSDeformAttnCorePyTorch_Split()
    model.eval()
    torch.onnx.export(model, (sampling_locations.double(), attention_weights.double(), value_list_f1, value_list_f2, value_list_f3, value_list_f4, shapes), 'ms_deform_attn_core_pytorch_kk.onnx', opset_version=20)
    # 加载 ONNX 模型并运行
    sess = onnxruntime.InferenceSession('ms_deform_attn_core_pytorch_kk.onnx')
    inputs = {
        'sampling_locations': sampling_locations.double().cpu().numpy(),
        'attention_weights': attention_weights.double().cpu().numpy(),
        'value_list_f1': value_list_f1.cpu().numpy(),
        'value_list_f2': value_list_f2.cpu().numpy(),
        'value_list_f3': value_list_f3.cpu().numpy(),
        'value_list_f4': value_list_f4.cpu().numpy(),
        'value_spatial_shapes': np.array(shapes.cpu(), dtype=np.int64),
    }
    ort_output_v1 = sess.run(None, inputs)[0]
    import pdb;pdb.set_trace()


    # 比较结果
    fwdok_pytorch_cuda = torch.allclose(output_cuda, output_pytorch)
    max_abs_err_pytorch_cuda = (output_cuda - output_pytorch).abs().max()
    max_rel_err_pytorch_cuda = ((output_cuda - output_pytorch).abs() / output_pytorch.abs()).max()

    fwdok_pytorch_onnx = np.allclose(output_pytorch.numpy(), ort_output)
    max_abs_err_pytorch_onnx = np.abs(output_pytorch.numpy() - ort_output).max()
    max_rel_err_pytorch_onnx = (np.abs(output_pytorch.numpy() - ort_output) / np.abs(output_pytorch.numpy())).max()

    fwdok_cuda_onnx = np.allclose(output_cuda.numpy(), ort_output)
    max_abs_err_cuda_onnx = np.abs(output_cuda.numpy() - ort_output).max()
    max_rel_err_cuda_onnx = (np.abs(output_cuda.numpy() - ort_output) / np.abs(output_cuda.numpy())).max()

    print(f'* PyTorch vs CUDA: {fwdok_pytorch_cuda} max_abs_err {max_abs_err_pytorch_cuda:.2e} max_rel_err {max_rel_err_pytorch_cuda:.2e}')
    print(f'* PyTorch vs ONNX: {fwdok_pytorch_onnx} max_abs_err {max_abs_err_pytorch_onnx:.2e} max_rel_err {max_rel_err_pytorch_onnx:.2e}')
    print(f'* CUDA vs ONNX: {fwdok_cuda_onnx} max_abs_err {max_abs_err_cuda_onnx:.2e} max_rel_err {max_rel_err_cuda_onnx:.2e}')


@torch.no_grad()
def check_forward_equal_with_pytorch_double_onnx_remove_split():
    value = torch.rand(N, S, M, D).cuda() * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    im2col_step = 64
    # 打印输入张量的形状
    print("value shape:", value.shape)
    print("shapes:", shapes)  
    print("sampling_locations shape:", sampling_locations.shape)
    print("attention_weights shape:", attention_weights.shape)

    # 第一 PyTorch 实现
    output_pytorch = ms_deform_attn_core_pytorch(value.double(), shapes, sampling_locations.double(), attention_weights.double()).detach().cpu()
    
    # 第二 CUDA 实现
    output_cuda = MSDeformAttnFunction.apply(value.double(), shapes, level_start_index, sampling_locations.double(), attention_weights.double(), im2col_step).detach().cpu()

    # 第三 导出 ONNX 模型
    model = MSDeformAttnCorePyTorch()
    model.eval()
    torch.onnx.export(model, (value.double(), shapes, sampling_locations.double(), attention_weights.double()), 'ms_deform_attn_core_pytorch.onnx', opset_version=20)
    # 加载 ONNX 模型并运行
    sess = onnxruntime.InferenceSession('ms_deform_attn_core_pytorch.onnx')
    inputs = {
        'value': value.double().cpu().numpy(),
        'value_spatial_shapes': np.array(shapes.cpu(), dtype=np.int64),
        'sampling_locations': sampling_locations.double().cpu().numpy(),
        'attention_weights': attention_weights.double().cpu().numpy()
    }
    ort_output = sess.run(None, inputs)[0]

    # 第四 拆分输入实现
    N_, S_, M_, D_ = value.shape
    value_list = value.double().split([H_ * W_ for H_, W_ in shapes], dim=1)
    # import pdb;pdb.set_trace()
    # value_list_f1 = value_list[0].flatten(2).reshape() # [2,5760,256] -> [2,48,120,256]

    value_list_f1 = value_list[0].flatten(2).transpose(1, 2).reshape(N_*M_, D_, shapes[0][0], shapes[0][1])
    value_list_f2 = value_list[1].flatten(2).transpose(1, 2).reshape(N_*M_, D_, shapes[1][0], shapes[1][1])
    value_list_f3 = value_list[2].flatten(2).transpose(1, 2).reshape(N_*M_, D_, shapes[2][0], shapes[2][1])
    value_list_f4 = value_list[3].flatten(2).transpose(1, 2).reshape(N_*M_, D_, shapes[3][0], shapes[3][1])
    output_pytorch_v1 = ms_deform_attn_core_pytorch_kk(sampling_locations.double(), attention_weights.double(), value_list_f1, value_list_f2, value_list_f3, value_list_f4).detach().cpu()

    # 第五 拆分输入再进行onnx导出 实现
    model = MSDeformAttnCorePyTorch_Split()
    model.eval()
    torch.onnx.export(model, (sampling_locations.double(), attention_weights.double(), value_list_f1, value_list_f2, value_list_f3, value_list_f4), 'ms_deform_attn_core_pytorch_kk.onnx', opset_version=20)
    # 加载 ONNX 模型并运行
    sess = onnxruntime.InferenceSession('ms_deform_attn_core_pytorch_kk.onnx')
    inputs = {
        'sampling_locations': sampling_locations.double().cpu().numpy(),
        'attention_weights': attention_weights.double().cpu().numpy(),
        'value_list_f1': value_list_f1.cpu().numpy(),
        'value_list_f2': value_list_f2.cpu().numpy(),
        'value_list_f3': value_list_f3.cpu().numpy(),
        'value_list_f4': value_list_f4.cpu().numpy(),
    }
    ort_output_v1 = sess.run(None, inputs)[0]

    # 比较结果
    outputs = [output_pytorch, output_cuda, torch.from_numpy(ort_output), output_pytorch_v1, torch.from_numpy(ort_output_v1)]
    names = ['PyTorch', 'CUDA', 'ONNX', 'PyTorch Split', 'ONNX Split']
    results = compare_outputs(outputs, names)

    # 创建表格
    df = pd.DataFrame(results, columns=['Output 1', 'Output 2', 'Equal', 'Max Abs Error', 'Max Rel Error'])
    print(df)


def compare_outputs(outputs, names):
    """
    比较多个输出的差异
    :param outputs: 输出列表
    :param names: 输出名称列表
    """
    num_outputs = len(outputs)
    results = []
    for i in range(num_outputs):
        for j in range(i + 1, num_outputs):
            output_i = outputs[i]
            output_j = outputs[j]
            name_i = names[i]
            name_j = names[j]

            # 计算绝对误差和相对误差
            abs_err = (output_i - output_j).abs()
            rel_err = abs_err / output_i.abs()

            # 计算最大绝对误差和最大相对误差
            max_abs_err = abs_err.max()
            max_rel_err = rel_err.max()

            # 判断是否相等
            is_equal = torch.allclose(output_i, output_j)

            print(f'* {name_i} vs {name_j}: {is_equal} max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}')
            results.append([name_i, name_j, is_equal, max_abs_err, max_rel_err])
    return results


if __name__ == '__main__':
    # check_forward_equal_with_pytorch_double()
    # check_forward_equal_with_pytorch_float()

    # for channels in [30, 32, 64, 71, 1025, 2048, 3096]:
    #     check_gradient_numerical(channels, True, True, True)

    # check_forward_equal_with_pytorch_double_onnx()
    # GPUS_PER_NODE=1 ./tools/run_dist_launch.sh 1 ./configs/r50_deformable_detr.sh

    check_forward_equal_with_pytorch_double_onnx_remove_split()



