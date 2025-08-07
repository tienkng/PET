import argparse
import os
import time
import torch
import torchvision.transforms as standard_transforms
import numpy as np
from PIL import Image
import cv2
import util.misc as utils
from models import build_model
import matplotlib.pyplot as plt


def get_args_parser():
    parser = argparse.ArgumentParser("Set Point Query Transformer", add_help=False)
    # Model parameters
    parser.add_argument("--backbone", default="vgg16_bn", type=str, help="Name of the convolutional backbone to use")
    parser.add_argument("--position_embedding", default="sine", type=str, choices=("sine", "learned", "fourier"), help="Type of positional embedding")
    parser.add_argument("--dec_layers", default=2, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument("--dim_feedforward", default=512, type=int, help="Intermediate size of the feedforward layers")
    parser.add_argument("--hidden_dim", default=256, type=int, help="Size of the embeddings")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout applied in the transformer")
    parser.add_argument("--nheads", default=8, type=int, help="Number of attention heads")
    # Loss parameters
    parser.add_argument("--set_cost_class", default=1, type=float, help="Class coefficient in the matching cost")
    parser.add_argument("--set_cost_point", default=0.05, type=float, help="SmoothL1 point coefficient")
    parser.add_argument("--ce_loss_coef", default=1.0, type=float)
    parser.add_argument("--point_loss_coef", default=5.0, type=float)
    parser.add_argument("--eos_coef", default=0.5, type=float, help="Relative classification weight of the no-object class")
    # Dataset parameters
    parser.add_argument("--dataset_file", default="SHA")
    parser.add_argument("--data_path", default="./data/ShanghaiTech/PartA", type=str)
    # Misc parameters
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--img_folder", default="", help="folder containing images for inference")
    parser.add_argument("--vis_dir", default="", help="directory to save visualized images")
    parser.add_argument("--num_workers", default=2, type=int)
    # Distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    return parser


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def visualization(samples, pred, vis_dir, img_path):
    pil_to_tensor = standard_transforms.ToTensor()
    restore_transform = standard_transforms.Compose([
        DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        standard_transforms.ToPILImage(),
    ])

    images = samples.tensors
    masks = samples.mask
    for idx in range(images.shape[0]):
        sample = restore_transform(images[idx])
        sample = pil_to_tensor(sample.convert("RGB")).numpy() * 255
        sample_vis = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()

        size = 6
        for p in pred[idx]:
            sample_vis = cv2.circle(sample_vis, (int(p[1]), int(p[0])), size, (0, 255, 0), -1)

        if vis_dir:
            imgH, imgW = masks.shape[-2:]
            valid_area = torch.where(~masks[idx])
            valid_h, valid_w = valid_area[0][-1], valid_area[1][-1]
            sample_vis = sample_vis[:valid_h + 1, :valid_w + 1]
            name = os.path.splitext(os.path.basename(img_path))[0]
            img_save_path = os.path.join(vis_dir, f"{name}_pred{len(pred[idx])}.jpg")
            cv2.imwrite(img_save_path, sample_vis)
            print(f"Image saved to {img_save_path}")


@torch.no_grad()
def evaluate_folder(model, img_folder, device, vis_dir=None):
    model.eval()
    img_count = 0
    total_time = 0
    point_counts = []

    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    img_names = sorted([
        f for f in os.listdir(img_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"))
    ])

    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    for img_name in img_names:
        img_path = os.path.join(img_folder, img_name)
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not load image at {img_path}")
                continue
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img = transform(img)
            img = img.to(device)
            samples = utils.nested_tensor_from_tensor_list([img]).to(device)

            img_h, img_w = samples.tensors.shape[-2:]

            start_time = time.time()
            outputs = model(samples, test=True)
            end_time = time.time()
            inference_time = end_time - start_time
            print(f"Inference {img_name}: {inference_time:.4f} seconds")

            total_time += inference_time
            img_count += 1

            raw_scores = torch.nn.functional.softmax(outputs["pred_logits"], -1)
            outputs_scores = raw_scores[:, :, 1][0]
            outputs_points = outputs["pred_points"][0]
            point_counts.append(len(outputs_scores))
            

            if vis_dir:
                points = [[point[0] * img_h, point[1] * img_w] for point in outputs_points]
                visualization(samples, [points], vis_dir, img_path)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    Avarage_time = total_time/img_count
    print(f"Avarage time of total images: {Avarage_time:.4f} seconds")

    return None


def main(args):
    if not os.path.exists(args.img_folder):
        raise FileNotFoundError(f"Image folder {args.img_folder} does not exist")
    if not os.path.exists(args.resume):
        raise FileNotFoundError(f"Checkpoint file {args.resume} does not exist")

    device = torch.device(args.device)
    print(f"Using device: {device}")

    model, _ = build_model(args)
    model.to(device)
    checkpoint = torch.load(args.resume, map_location=device)
    model.load_state_dict(checkpoint["model"])

    vis_dir = args.vis_dir if args.vis_dir else None
    evaluate_folder(model, args.img_folder, device, vis_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PET evaluation script", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
