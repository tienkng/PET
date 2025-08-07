import argparse
import os
import time
import numpy as np
from PIL import Image
import cv2
import onnxruntime as ort
import torch
import torchvision.transforms as standard_transforms
import logging
def get_args_parser():
    """Parse command-line arguments for ONNX inference script."""
    parser = argparse.ArgumentParser("ONNX inference for Point Query Transformer", add_help=False)
    parser.add_argument("--onnx_model", default="model.onnx", type=str, help="Path to ONNX model")
    parser.add_argument("--img_folder", default="", type=str, help="Folder containing images for inference")
    parser.add_argument("--vis_dir", default="", type=str, help="Directory to save visualized images")
    parser.add_argument("--device", default="cpu", type=str, help="Device to use for preprocessing (cpu or cuda)")
    parser.add_argument("--input_height", default=512, type=int, help="Height of resized input images")
    parser.add_argument("--input_width", default=1024, type=int, help="Width of resized input images")
    parser.add_argument("--num_workers", default=2, type=int, help="Number of workers for data loading")
    parser.add_argument('--providers', help='ONNX runtime providers (comma-separated)', 
                       default='CUDAExecutionProvider,CPUExecutionProvider', type=str)
    return parser

class DeNormalize:
    """Denormalize a tensor image with given mean and std."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def nested_tensor_from_tensor_list(tensor_list, device):
    """Convert a list of tensors to a NestedTensor with padding and mask."""
    if tensor_list[0].ndim == 3:
        max_size = [max(s) for s in zip(*[img.shape for img in tensor_list])]
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("Only 3D tensors are supported")
    return tensor, mask

def visualization(image_orig, pred_points, orig_size, vis_dir, img_path):
    """Visualize predictions by drawing points on the original image and save the result."""
    pil_to_tensor = standard_transforms.ToTensor()
    restore_transform = standard_transforms.Compose([
        DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        standard_transforms.ToPILImage(),
    ])

    # Denormalize and convert to numpy
    sample = restore_transform(image_orig)
    sample = pil_to_tensor(sample.convert("RGB")).numpy() * 255
    sample_vis = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()

    # Resize back to original size
    orig_h, orig_w = orig_size
    sample_vis = cv2.resize(sample_vis, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # Draw predictions (green) on original size
    size = 6
    for p in pred_points:
        x = int(p[1] * orig_w)  # Scale x-coordinate
        y = int(p[0] * orig_h)  # Scale y-coordinate
        sample_vis = cv2.circle(sample_vis, (x, y), size, (0, 255, 0), -1)

    # Save image
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)
        name = os.path.splitext(os.path.basename(img_path))[0]
        img_save_path = os.path.join(vis_dir, f"{name}.jpg")
        cv2.imwrite(img_save_path, sample_vis)
        print(f"Image saved to {img_save_path}")

def evaluate_folder(onnx_model_path, img_folder, device, vis_dir=None, input_size=(512, 1024)):

    """Evaluate the ONNX model on all images in a folder and print model inference time."""
    if str(device) == "cuda" and torch.cuda.is_available():
        providers = ["CUDAExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_model_path, providers=providers) # CPUExecutionProvider and CUDAExecutionProvider

    # Get all images in the folder
    img_names = [f for f in os.listdir(img_folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"))]
    if not img_names:
        raise FileNotFoundError(f"No valid images found in {img_folder}")

    transform = standard_transforms.Compose([
        standard_transforms.Resize(input_size),
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for img_name in img_names:
        img_path = os.path.join(img_folder, img_name)
        try:
            # Load original image to get size
            img_orig = cv2.imread(img_path)
            if img_orig is None:
                print(f"Could not load image at {img_path}")
                continue
            orig_h, orig_w = img_orig.shape[:2]
            img = Image.fromarray(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
            img = transform(img)
            img = img.to(device)
            tensors, mask = nested_tensor_from_tensor_list([img], device)
            
            # Run inference with ONNX and measure only model inference time
            start_time = time.time()
            outputs = session.run(None, {
                'input_tensors': tensors.cpu().numpy(),
                'input_mask': mask.cpu().numpy()
            })
            end_time = time.time()
            inference_time = end_time - start_time

            print(f"Model inference time for {img_name}: {inference_time:.4f} seconds")

            # Process predictions for visualization
            pred_logits = torch.from_numpy(outputs[0]).to(device)  # Shape: [1, num_queries, 2]
            pred_points = torch.from_numpy(outputs[1]).to(device)  # Shape: [1, num_queries, 2]
            outputs_scores = torch.nn.functional.softmax(pred_logits, -1)[:, :, 1][0]  # Positive class scores
            outputs_points = pred_points[0]  # Shape: [num_queries, 2]

            if vis_dir:
                # Scale points back to original size (handled in visualization)
                points = [[point[0], point[1]] for point in outputs_points]  # Keep normalized coordinates
                visualization(img, points, (orig_h, orig_w), vis_dir, img_path)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

def main(args):
    """Main function to run ONNX inference."""
    if not os.path.exists(args.img_folder):
        raise FileNotFoundError(f"Image folder {args.img_folder} does not exist")
    if not os.path.exists(args.onnx_model):
        raise FileNotFoundError(f"ONNX model file {args.onnx_model} does not exist")

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vis_dir = args.vis_dir if args.vis_dir else None
    input_size = (args.input_height, args.input_width)
    evaluate_folder(
        args.onnx_model, args.img_folder, device, vis_dir, input_size
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser("ONNX inference script", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)