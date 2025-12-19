import argparse
from pathlib import Path
import os
import torch
from torch.nn import DataParallel
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.arg_parser import get_argparser
from models.dave import build_model
from utils.data import pad_image

from typing import List, Optional

import cv2
import numpy as np
import torch.nn.functional as F

import gradio as gr
import io

import threading

import datetime
from PIL import Image
import torch
from pymongo import MongoClient
from bson.binary import Binary
from io import BytesIO

from tqdm import tqdm
import time


# ------------------------------------------------------------------
#  CONFIGURATION
ROOT_DIR   = "/VisDrone"
IMG_DIR    = os.path.join(ROOT_DIR, "images")
ANN_DIR    = os.path.join(ROOT_DIR, "annotations")

CATEGORIES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]

MONGO_URI = "mongodb://root:example1234@mongo:27017/"
DB_NAME   = "visdrone"
COL_NAME  = "results"

# ------------------------------------------------------------------
#  MONGO client
client   = MongoClient(MONGO_URI)
db       = client[DB_NAME]
coll     = db[COL_NAME]


# ------------------------------------------------------------------
#  Helper: extract ground‑truth counts from the .txt file
def get_gt_counts(ann_path):
    counts = {c: 0 for c in CATEGORIES}
    with open(ann_path, "r") as f:
        for line in f:
            if not line.strip():  # skip empty lines
                continue
            cls_id = int(line.split()[0]) - 1  # 1‑based → 0‑based
            if 0 <= cls_id < len(CATEGORIES):
                counts[CATEGORIES[cls_id]] += 1
    return counts


_model_lock = threading.Lock()


def resize(img, img_size):
    resize_img = T.Resize((img_size, img_size), antialias=True)
    w, h = img.size
    img = T.Compose([
        T.ToTensor(),
        resize_img,
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])(img)
    scale = torch.tensor([1.0, 1.0]) / torch.tensor([w, h]) * img_size
    return img, scale

def parse_categories(text: Optional[str]) -> List[str]:
    """
    Convert a comma-separated string into a list.
    - Trims whitespace around each item.
    - Ignores empty items (e.g. 'a,,b' → ['a', 'b']).
    - Handles None → [].
    """
    if (not text) or (text==""):
        return []

    # Split on commas, strip spaces, drop empties
    return [item.strip() for item in text.split(",") if item.strip()]

@torch.no_grad()
def demo(args):
    # global fig, ax

    prompts = parse_categories(args.prompts_txt)


    dpi = plt.rcParams['figure.dpi']
    plt.figure()
    gpu = 0
    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    model = DataParallel(
        build_model(args).to(device),
        device_ids=[gpu],
        output_device=gpu
    )
    model.load_state_dict(
        torch.load(os.path.join(args.model_path, 'DAVE_0_shot.pth'))['model'], strict=False
    )
    pretrained_dict_feat = {k.split("feat_comp.")[1]: v for k, v in
                            torch.load(os.path.join(args.model_path, 'verification.pth'))[
                                'model'].items() if 'feat_comp' in k}
    model.module.feat_comp.load_state_dict(pretrained_dict_feat)
    model.eval()

    bboxes = torch.zeros((1, 3, 4))

    @torch.no_grad()
    def one_img(image: Image.Image, prompts_txt: str):
        try:
            with _model_lock:
                scale_x = scale_y = 1
                # image = Image.open(img_path).convert("RGB")
                img, scale = resize(image, args.image_size)
                img = img.unsqueeze(0).to(device)

                # Convert the raw text to a Python list
                prompts = [p.strip() for p in prompts_txt.splitlines() if p.strip()]

                denisty_map, _, _, predicted_bboxes, clusters = model(img, bboxes=bboxes, classes=prompts)

                print(f"== Prompts: {prompts}")

                if args.two_passes:
                    boxes_predicted = predicted_bboxes.box
                    scale_y = min(1, 50 / (boxes_predicted[:, 2] - boxes_predicted[:, 0]).mean())
                    scale_x = min(1, 50 / (boxes_predicted[:, 3] - boxes_predicted[:, 1]).mean())

                    if scale_x < 1 or scale_y < 1:
                        scale_x = (int(args.image_size * scale_x) // 8 * 8) / args.image_size
                        scale_y = (int(args.image_size * scale_y) // 8 * 8) / args.image_size
                        resize_ = T.Resize((int(args.image_size * scale_x), int(args.image_size * scale_y)), antialias=True)
                        img_resized = resize_(img)

                        img_resized = pad_image(img_resized[0]).unsqueeze(0)

                    else:
                        scale_y = max(1, 11 / (boxes_predicted[:, 2] - boxes_predicted[:, 0]).mean())
                        scale_x = max(1, 11 / (boxes_predicted[:, 3] - boxes_predicted[:, 1]).mean())

                        if scale_y > 1.9:
                            scale_y = 1.9
                        if scale_x > 1.9:
                            scale_x = 1.9

                        scale_x = (int(args.image_size * scale_x) // 8 * 8) / args.image_size
                        scale_y = (int(args.image_size * scale_y) // 8 * 8) / args.image_size
                        resize_ = T.Resize((int(args.image_size * scale_x), int(args.image_size * scale_y)), antialias=True)
                        img_resized = resize_(img)

                    if scale_x != 1.0 or scale_y != 1.0:
                        denisty_map, _, _, predicted_bboxes, clusters = model(img, bboxes=bboxes, classes=prompts)

                pred_boxes = predicted_bboxes.box.cpu() / torch.tensor([scale_y*scale[0], scale_x*scale[1], scale_y*scale[0], scale_x*scale[1]])

                # ------------------------------------------------------------------
                # Overlay density map and bounding boxes using OpenCV (cv2)
                # ------------------------------------------------------------------
                # Convert the original PIL image to a NumPy array (BGR for cv2)
                img_cv = np.array(image)[:, :, ::-1]  # RGB → BGR

                # Prepare the density map for overlay:
                                # 1️⃣  Make sure the density map matches the image resolution
                dens = denisty_map.squeeze()          # shape: (1, H', W')
                if dens.shape[-1] != image.width or dens.shape[-2] != image.height:
                    dens = F.interpolate(denisty_map, size=(image.height, image.width),
                                        mode='bilinear', align_corners=False).squeeze()

                # 2️⃣  Normalise the density values to [0, 1] for display
                dens_np = dens.cpu().numpy()
                dens_np -= dens_np.min()
                dens_np /= dens_np.max() + 1e-6  # avoid division by zero

                # dens_np is already normalized to [0, 1]
                dens_uint8 = (dens_np * 255).astype(np.uint8)          # 0‑255 single channel
                # Apply a colormap (use JET as a close approximation to viridis)
                dens_colored = cv2.applyColorMap(dens_uint8, cv2.COLORMAP_JET)

                # Blend the colored density map with the original image
                overlay = cv2.addWeighted(dens_colored, 0.8, img_cv, 0.2, 0)

                # Draw the predicted bounding boxes (red, thickness=2)
                for box in pred_boxes:
                    x1, y1, x2, y2 = map(int, box.tolist())
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Convert the result back to a PIL image for Gradio
                out_img = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

                # Prepare the info string for the second output panel
                out_info = (
                    f"= Found {clusters} clusters\n"
                    f"= Density sum {denisty_map.sum().item()}\n"
                    f"= Count of boxes {len(pred_boxes)}"
                )
                return out_img, out_info, clusters, denisty_map.sum().item(), len(pred_boxes)
        except RuntimeError as ex:
            raise gr.Error("Not enough VRAM at the moment 💥! Try again in 5 mins.", duration=10)

    # ------------------------------------------------------------------
    #  Main loop
    # ------------------------------------------------------------------
    for img_name in tqdm(sorted(os.listdir(IMG_DIR)), desc="Processing images"):
        start_time = time.perf_counter()   # start timing

        if not img_name.lower().endswith(".jpg"):
            continue

        ann_name = os.path.splitext(img_name)[0] + ".txt"
        ann_path = os.path.join(ANN_DIR, ann_name)

        if not os.path.isfile(ann_path):
            print(f"⚠️  Annotation missing for {img_name}, skipping")
            continue

        image = Image.open(os.path.join(IMG_DIR, img_name))
        gt_counts = get_gt_counts(ann_path)

        per_cat = []            # list of dicts per category
        mae_density_sum = 0.0
        mae_box_sum     = 0.0
        mse_density_sum = 0.0
        mse_box_sum     = 0.0

        for cat in CATEGORIES:
            # Run the model for this category
            out_img, out_info, clusters, dens_sum, pred_box_len = one_img(image, cat)

            # Ground‑truth count for the category
            gt = gt_counts[cat]

            # MAE for density sum and for predicted box count
            mae_density = abs(gt - dens_sum)
            mae_box     = abs(gt - pred_box_len)

            # MSE (square the MAE terms)
            mse_density = mae_density ** 2
            mse_box     = mae_box ** 2

            # Store per‑category metrics
            per_cat.append({
                "category"        : cat,
                "gt_count"        : int(gt),
                "density_sum"     : float(dens_sum),
                "pred_box_len"    : int(pred_box_len),
                "mae_density"     : float(mae_density),
                "mae_box"         : float(mae_box),
                "mse_density"     : float(mse_density),
                "mse_box"         : float(mse_box),
                "clusters"        : clusters
            })

            # Accumulate for macro metrics
            mae_density_sum += mae_density
            mae_box_sum     += mae_box
            mse_density_sum += mse_density
            mse_box_sum     += mse_box

        n_cat = len(CATEGORIES)

        # Macro‑MAE for density and for box separately
        macro_mae_density = mae_density_sum / n_cat
        macro_mae_box     = mae_box_sum     / n_cat

        # Macro‑RMSE (sqrt of the average of the two MSEs)
        macro_rmse_density = (mse_density_sum / n_cat) ** 0.5
        macro_rmse_box = (mse_box_sum / n_cat) ** 0.5

        # Prepare MongoDB document
        doc = {
            "image_name"          : img_name,
            "image_path"          : os.path.join(IMG_DIR, img_name),
            "ground_truth_counts" : gt_counts,
            "per_category"        : per_cat,
            "macro_mae_density"   : macro_mae_density,
            "macro_mae_box"       : macro_mae_box,
            "macro_rmse_density"  : macro_rmse_density,
            "macro_rmse_box"      : macro_rmse_box,
            "timestamp"           : datetime.datetime.utcnow()
        }

        # Store the out‑image as binary (PNG)
        buf = BytesIO()
        out_img.save(buf, format="PNG")
        doc["out_image"] = Binary(buf.getvalue())

        # Store elapsed time in the document
        doc["processing_time_sec"] = time.perf_counter() - start_time
        
        coll.insert_one(doc)

        print(f"✅  Processed {img_name} (took {doc['processing_time_sec']:.3f}s)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DAVE', parents=[get_argparser()])
    parser.add_argument('--prompts_txt', type=str)
    # parser.add_argument('--img_path', type=Path)
    # parser.add_argument('--out_path', type=Path, default="material")
    # parser.add_argument('--show', action='store_true')
    parser.add_argument('--two_passes', action='store_true')
    args = parser.parse_args()
    demo(args)
