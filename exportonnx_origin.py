import argparse
import random
from pathlib import Path
import os
from typing import List
import torch
from torch import Tensor
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
import torch.onnx

import datasets
from datasets import build_dataset
import util.misc as utils
from engine import evaluate
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set Point Query Transformer', add_help=False)

    # model parameters
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned', 'fourier'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=512, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    
    # loss parameters
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_point', default=0.05, type=float,
                        help="SmoothL1 point coefficient in the matching cost")
    parser.add_argument('--ce_loss_coef', default=1.0, type=float)
    parser.add_argument('--point_loss_coef', default=5.0, type=float)
    parser.add_argument('--eos_coef', default=0.5, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default="SHA")
    parser.add_argument('--data_path', default="./data/ShanghaiTech/PartA", type=str)

    # misc parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--vis_dir', default="")
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--onnx_output', default='model.onnx', type=str,
                        help='Path to save the exported ONNX model')
    return parser


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    if tensor_list[0].ndim == 3:
        max_size = [max(s) for s in zip(*[img.shape for img in tensor_list])]
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return utils.NestedTensor(tensor, mask)


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tensors, mask):
        samples = utils.NestedTensor(tensors, mask)
        outputs = self.model(samples, test=True)
        # Only return the dynamic tensor outputs for ONNX
        return (outputs['pred_logits'], outputs['pred_points'], outputs['pred_offsets'], outputs['split_map_raw'])


def main(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, criterion = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('params:', n_parameters/1e6)

    # load pretrained model
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])        
    
    # export to ONNX
    model.eval()
    # Create a dummy input with adjusted size
    dummy_images = [torch.randn(3, 512, 1024, dtype=torch.float32, device=device)]  # Adjusted size
    dummy_nested_tensor = nested_tensor_from_tensor_list(dummy_images)
    dummy_tensors = dummy_nested_tensor.tensors  # Shape: [1, 3, 512, 1024]
    dummy_mask = dummy_nested_tensor.mask  # Shape: [1, 512, 1024]

    # Wrap the model to accept tensors and mask separately
    wrapped_model = ModelWrapper(model_without_ddp)
    
    # Debug: Test the model output
    with torch.no_grad():
        outputs = wrapped_model(dummy_tensors, dummy_mask)
        print("Model outputs:", [out.shape if isinstance(out, torch.Tensor) else out for out in outputs])

    # Export the model to ONNX
    torch.onnx.export(
        wrapped_model,
        (dummy_tensors, dummy_mask),
        args.onnx_output,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input_tensors', 'input_mask'],
        output_names=['pred_logits', 'pred_points', 'pred_offsets', 'split_map_raw'],
        dynamic_axes={
            'input_tensors': {0: 'batch_size', 2: 'height', 3: 'width'},
            'input_mask': {0: 'batch_size', 1: 'height', 2: 'width'},
            'pred_logits': {0: 'batch_size'},
            'pred_points': {0: 'batch_size'},
            'pred_offsets': {0: 'batch_size'},
            'split_map_raw': {0: 'batch_size'}
        }
    )
    print(f"Model exported to {args.onnx_output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PET evaluation script with ONNX export', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
