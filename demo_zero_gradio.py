#!/usr/bin/env python
# ------------------------------------------------------------------
# Demo script for the DAVE zero‑shot counting model, exposed via Gradio.
# Pulls the following env‑vars:
#   - GRADIO_PORT      : port on which the UI will listen (default 7860)
#   - MODEL_PATH       : directory with the .pth checkpoints
#   - IMAGE_SIZE       : default input size (default 384)
#   - TWO_PASSES       : "1" to enable two‑pass refinement
# ------------------------------------------------------------------
import os
import torch
import gradio as gr
from pathlib import Path
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from utils.arg_parser import get_argparser  # kept for compatibility
from models.dave import build_model
from utils.data import pad_image
from torch.nn import DataParallel

# ------------------------------------------------------------------
# 1️⃣  Read configuration once (module import time)
# ------------------------------------------------------------------
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))
MODEL_PATH  = os.getenv("MODEL_PATH", "/app/DAVE/weights")
IMAGE_SIZE  = int(os.getenv("IMAGE_SIZE", "384"))
TWO_PASSES  = os.getenv("TWO_PASSES", "1") == "1"
DEVICE      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# 2️⃣  Helper: Build & load the model
# ------------------------------------------------------------------
def get_model(model_path: str, image_size: int, two_passes: bool) -> torch.nn.Module:
    gpu = 0
    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    # Base model
    model = DataParallel(
        build_model(
            {
                "image_size": image_size,
                "use_appearance": True,
                "use_objectness": True,
                "two_passes": two_passes,
            }
        ).to(device),
        device_ids=[gpu],
        output_device=gpu,
    )

    # Load the main DAVE checkpoint
    ckpt = torch.load(os.path.join(model_path, "DAVE_0_shot.pth"), map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)

    # Load the verification branch
    ver = torch.load(os.path.join(model_path, "verification.pth"), map_location=device)
    pretrained_dict_feat = {
        k.split("feat_comp.")[1]: v
        for k, v in ver["model"].items()
        if "feat_comp" in k
    }
    model.module.feat_comp.load_state_dict(pretrained_dict_feat)

    model.eval()
    return model

# ------------------------------------------------------------------
# 3️⃣  Inference routine
# ------------------------------------------------------------------
@torch.no_grad()
def predict(img: Image.Image) -> plt.Figure:
    model = get_model(MODEL_PATH, IMAGE_SIZE, TWO_PASSES)

    # Pre‑process
    preprocess = T.Compose(
        [
            T.ToTensor(),
            T.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
            T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    img_t = preprocess(img).unsqueeze(0).to(DEVICE)

    # Dummy bbox placeholder (matches the original script)
    dummy_bboxes = torch.zeros((1, 3, 4), device=DEVICE)

    # Forward pass
    density, _, _, pred_bboxes = model(img_t, bboxes=dummy_bboxes)

    # Two‑pass refinement (optional)
    if TWO_PASSES:
        boxes_pred = pred_bboxes.box
        scale_y = min(
            1, 50 / (boxes_pred[:, 2] - boxes_pred[:, 0]).mean().item()
        )
        scale_x = min(
            1, 50 / (boxes_pred[:, 3] - boxes_pred[:, 1]).mean().item()
        )
        if scale_x < 1 or scale_y < 1:
            sx = (int(IMAGE_SIZE * scale_x) // 8 * 8) / IMAGE_SIZE
            sy = (int(IMAGE_SIZE * scale_y) // 8 * 8) / IMAGE_SIZE
            resize_ = T.Resize((int(IMAGE_SIZE * sy), int(IMAGE_SIZE * sx)), antialias=True)
            img_resized = resize_(img_t)
            img_resized = pad_image(img_resized[0]).unsqueeze(0).to(DEVICE)
            density, _, _, pred_bboxes = model(img_resized, bboxes=dummy_bboxes)

    # Post‑process
    boxes = pred_bboxes.box.cpu().numpy()
    count = density.sum().item()

    # Build matplotlib figure
    fig, ax = plt.subplots(
        figsize=(img.width / 100, img.height / 100), dpi=100
    )
    ax.imshow(img)
    for box in boxes:
        x0, y0, x1, y1 = box
        rect = matplotlib.patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=2, edgecolor="red", facecolor="none",
        )
        ax.add_patch(rect)
    ax.set_title(f"Density count: {round(count, 1)}  |  Boxes: {len(boxes)}")
    ax.axis("off")
    return fig

# ------------------------------------------------------------------
# 4️⃣  Gradio wrapper
# ------------------------------------------------------------------
def create_interface() -> gr.Interface:
    return gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil"),
        outputs="plot",
        title="DAVE Zero‑Shot Counting",
        description=(
            "Upload an image – the model predicts a density map and bounding boxes.\n\n"
            "Environment variables used:\n"
            "- `MODEL_PATH` – directory that contains the .pth checkpoints\n"
            "- `GRADIO_PORT` – port on which the UI will listen\n"
            "- `IMAGE_SIZE` – default input size (default 384)\n"
            "- `TWO_PASSES` – `1` to enable the two‑pass refinement\n"
        ),
        allow_flagging="never",   # <‑‑ add this
    )

# ------------------------------------------------------------------
# 5️⃣  Main entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    iface = create_interface()
    iface.launch(
        share=False,
        server_port=GRADIO_PORT,
        server_name="0.0.0.0",
    )