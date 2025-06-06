# src/grad_cam.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from src.model import FungiClassifier

# ─── STEP 1: Set up argument parsing ───────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Generate a Grad-CAM heatmap for a single image using a trained FungiClassifier (ResNet-18)."
)
parser.add_argument(
    "--image",
    required=True,
    help="Path to the input image file (e.g. data/split_mind_funga/val/<class>/<img>.jpg)",
)
parser.add_argument(
    "--output",
    default="gradcam_output.jpg",
    help="Where to save the overlayed Grad-CAM result (default: gradcam_output.jpg)",
)
parser.add_argument(
    "--weights",
    default="models/resnet18_fungi.pth",
    help="Path to the .pth weights file (default: models/resnet18_fungi.pth)",
)
parser.add_argument(
    "--target_class_idx",
    type=int,
    default=None,
    help="If you want Grad-CAM for a specific class index, pass it here. Otherwise, uses the model’s top prediction.",
)
parser.add_argument(
    "--device",
    choices=["cpu", "cuda", "mps"],
    default=None,
    help="Device to run on. By default, auto‐detects ('cuda' > 'mps' > 'cpu').",
)
args = parser.parse_args()


# ─── STEP 2: Utility to load image + preprocess ────────────────────────────────
def load_image(img_path):
    """
    Load a PIL image, apply the same transforms (resize + normalize) you used in training.
    Returns both the original PIL image (for overlay) and a torch.Tensor (1×3×224×224).
    """
    pil_img = Image.open(img_path).convert("RGB")

    preprocess = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    tensor = preprocess(pil_img).unsqueeze(0)  # add batch dim
    return pil_img, tensor


# ─── STEP 3: Define a small GradCAM helper ─────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        """
        model: your nn.Module (FungiClassifier)
        target_layer: the convolutional layer you want to hook (e.g. model.model.layer4)
        """
        self.model = model
        self.target_layer = target_layer

        # Placeholders for gradients and activations
        self.gradients = None
        self.activations = None

        # Register hooks
        def forward_hook(module, input, output):
            # output: feature map of shape [B, C, H, W]
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0] is gradients w.r.t. the activations
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def __call__(self, input_tensor, class_idx=None):
        """
        input_tensor: (1×3×224×224) preprocessed image
        class_idx: if None, uses the top predicted class
        Returns: heatmap (H×W, numpy float in [0,1]) and the model’s raw output scores
        """
        self.model.zero_grad()
        output = self.model(input_tensor)  # shape: [1, num_classes]
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Compute gradient of the score for class_idx w.r.t. activations
        score = output[0, class_idx]
        score.backward(retain_graph=True)

        grads = self.gradients[0]       # shape: [C, H, W]
        acts = self.activations[0]      # shape: [C, H, W]

        # Global‐average‐pool the gradients over (H, W)
        weights = grads.mean(dim=(1, 2))  # shape: [C]

        # Weighted combination of forward activations
        cam = (weights.view(-1, 1, 1) * acts).sum(dim=0)  # shape: [H, W]
        cam = F.relu(cam)  # zero out negative values

        # Normalize to [0, 1]
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        # Convert to numpy
        return cam.cpu().numpy(), output.cpu().detach().numpy()


# ─── STEP 4: Main routine ──────────────────────────────────────────────────────
def main():
    # Select device
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    print(f"Using device: {device}")

    # 1) Load your model architecture and weights
    #    NOTE: FungiClassifier wraps a torchvision.resnet18 backbone
    #    so the last conv layer is `model.model.layer4`.
    state = torch.load(args.weights, map_location="cpu")
    num_classes = state["model.fc.weight"].shape[0]

    model = FungiClassifier(num_classes=num_classes)
    model.load_state_dict(state)
    model.to(device).eval()

    # 2) Choose the target conv layer: ResNet-18’s last conv block is `layer4`
    target_layer = model.model.layer4

    # 3) Initialize GradCAM with model + target layer
    gradcam = GradCAM(model, target_layer)

    # 4) Load image + preprocess
    pil_img, input_tensor = load_image(args.image)
    input_tensor = input_tensor.to(device)

    # 5) Run GradCAM
    cam_map, output_scores = gradcam(input_tensor, class_idx=args.target_class_idx)

    # If user didn’t specify a target class, report which class the model predicted
    pred_idx = int(output_scores.argmax(axis=1)[0]) if args.target_class_idx is None else args.target_class_idx
    print(f"Predicted class index = {pred_idx}  (use --target_class_idx to override)")

    # 6) Resize cam to original image size
    cam_map_resized = np.uint8(255 * cam_map)
    cam_map_resized = Image.fromarray(cam_map_resized).resize(pil_img.size, resample=Image.BILINEAR)
    cam_map_resized = np.array(cam_map_resized) / 255.0  # back to [0,1] float, but now same H×W as pil_img

    # 7) Overlay heatmap on PIL image
    heatmap = plt.get_cmap("jet")(cam_map_resized)[:, :, :3]  # ndarray H×W×3 in [0,1]
    heatmap = np.uint8(255 * heatmap)

    orig_np = np.array(pil_img)
    if orig_np.dtype != np.uint8:
        orig_np = np.uint8(255 * orig_np)

    overlay = np.uint8(0.5 * orig_np + 0.5 * heatmap)

    # 8) Save result
    out_img = Image.fromarray(overlay)
    out_img.save(args.output)
    print(f"Grad-CAM overlay saved to: {args.output}")

    # Optionally, show side‐by‐side original and heatmap
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(pil_img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(cam_map_resized, cmap="jet")
    axes[1].set_title("Grad-CAM Map")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
