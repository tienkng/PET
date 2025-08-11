import os
import numpy as np
from PIL import Image
import cv2
import onnxruntime as ort
import torch
import torchvision.transforms as standard_transforms

NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def softmax(x: np.ndarray, axis: int = -1):
    """Equivalent to torch.nn.functional.softmax()."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

class PETModel:
    def __init__(self, model_path="model.onnx", device="cuda", input_size=(512, 1024)):
        """Initialize the PET model for ONNX inference."""
        self.device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        providers = ["CUDAExecutionProvider"] if self.device.type == "cuda" else ["CPUExecutionProvider"]
        self.model = ort.InferenceSession(model_path, providers=providers)
        
        self.inp_names = [x.name for x in self.model.get_inputs()]
        self.out_names = [x.name for x in self.model.get_outputs()]
        
    def preprocess(self, image: np.ndarray) -> tuple:
        """Preprocess image for ONNX inference.

        Args:
            image (np.ndarray): Input image in shape (H, W, C) - OpenCV format.

        Returns:
            tuple: (input_tensor, input_mask, normalized_image, original_size)
        """
        orig_h, orig_w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = standard_transforms.Compose([
            standard_transforms.ToPILImage(),
            standard_transforms.Resize(self.input_size),
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ])
        norm_image = transform(image_rgb).to(self.device)

        tensor, mask = self.nested_tensor_from_tensor_list([norm_image], self.device)
        return tensor.cpu().numpy(), mask.cpu().numpy(), norm_image, (orig_h, orig_w)

    @staticmethod
    def nested_tensor_from_tensor_list(tensor_list, device):
        """Convert a list of tensors to a NestedTensor with padding and mask."""
        if tensor_list[0].ndim == 3:
            max_size = [max(s) for s in zip(*[img.shape for img in tensor_list])]
            batch_shape = [len(tensor_list)] + max_size
            b, _, h, w = batch_shape
            dtype = tensor_list[0].dtype
            tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
            mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
            for img, pad_img, m in zip(tensor_list, tensor, mask):
                pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
                m[: img.shape[1], : img.shape[2]] = False
        else:
            raise ValueError("Only 3D tensors are supported")
        return tensor, mask

    def postprocess(self, predict, threshold=0.5):
        """Postprocess predictions to get points and count.

        Args:
            predict (list): [pred_logits, pred_points] from ONNX model.
            threshold (float): Confidence threshold for filtering points.

        Returns:
            tuple: (points, point_count)
        """
        pred_logits = torch.from_numpy(predict[0]).to(self.device)  
        pred_points = torch.from_numpy(predict[1]).to(self.device)  
        
        scores = torch.nn.functional.softmax(pred_logits, dim=-1)[:, :, 1][0]  
        mask = scores > threshold
        filtered_points = pred_points[0][mask].cpu().numpy()  
        point_count = len(filtered_points)
        
        return filtered_points.tolist(), point_count

    def run(self, image: np.ndarray, threshold: float = 0.5):
        """Run inference on a single image.

        Args:
            image (np.ndarray): Input image in shape (H, W, C).
            threshold (float): Confidence threshold for filtering points.

        Returns:
            tuple: (points, point_count, normalized_image, original_size)
        """
        inp_tensor, inp_mask, norm_image, orig_size = self.preprocess(image)
        predict = self.model.run(self.out_names, {
            self.inp_names[0]: inp_tensor,
            self.inp_names[1]: inp_mask
        })
        points, point_count = self.postprocess(predict, threshold)
        return points, point_count, norm_image, orig_size

    def visualize(self, norm_image: torch.Tensor, points: list, orig_size: tuple, save_path: str = None):
        """Visualize predictions by drawing points on the image.

        Args:
            norm_image (torch.Tensor): Normalized image tensor [3, H, W].
            points (list): List of [y, x] points (normalized).
            orig_size (tuple): (orig_h, orig_w) of original image.
            save_path (str, optional): Path to save visualized image.
        """
        orig_h, orig_w = orig_size
        restore_transform = standard_transforms.Compose([
            standard_transforms.Normalize(mean=[-m/s for m, s in zip(NORM_MEAN, NORM_STD)], std=[1/s for s in NORM_STD]),
            standard_transforms.ToPILImage(),
        ])
        
        img = restore_transform(norm_image.cpu())
        img = standard_transforms.ToTensor()(img.convert("RGB")).numpy() * 255
        img_vis = img.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()
        
        img_vis = cv2.resize(img_vis, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        
        for p in points:
            x = int(p[1] * orig_w)  
            y = int(p[0] * orig_h)  
            if 0 <= x < orig_w and 0 <= y < orig_h:  
                img_vis = cv2.circle(img_vis, (x, y), 8, (0, 255, 0), -1)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, img_vis)
        return img_vis

def main():
    model_path = "/home/tiennv/FPT/yolov9/counting_people/PET/PET/weight/model_simp_clean.onnx"
    img_folder = "/home/tiennv/FPT/yolov9/PET/data4render/images"
    vis_dir = "onnx_img_output"
    device = "cuda"
    input_size = (512, 1024)
    threshold = 0.5

    os.makedirs(vis_dir, exist_ok=True)
    model = PETModel(model_path, device, input_size)
    
    img_names = [f for f in os.listdir(img_folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"))]
    
    for img_name in img_names:
        img_path = os.path.join(img_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not load image at {img_path}")
            continue
            
        points, point_count, norm_image, orig_size = model.run(img, threshold)
        print(f"Image path: {img_path}\tCount: {point_count}")
            
        if vis_dir:
            save_path = os.path.join(vis_dir, f"{os.path.splitext(img_name)[0]}.jpg")
            model.visualize(norm_image, points, orig_size, save_path)

if __name__ == "__main__":
    main()
            