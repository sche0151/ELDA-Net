"""
ELDA-Net: Edge-Lightweight Detection and Adaptation Network
============================================================
Full experiment pipeline: training, evaluation, and lane detection inference.
Implements the complete system described in thesis Chapters 3-5.

Usage:
  python elda_net_experiment.py --mode train   --dataset TuSimple
  python elda_net_experiment.py --mode eval    --dataset TuSimple
  python elda_net_experiment.py --mode infer   --video demo.mp4 --output results/out.mp4
"""

import time
import os
import cv2
import sys
import torch
import numpy as np
import yaml
import argparse
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from model.unet import UNet
from utils.dataset import LaneDataset
from utils.preprocessing import preprocess_image
from utils.visualization import overlay_lanes
from utils.adaptive_estimator import AdaptiveLaneEstimator
from utils.metrics import compute_metrics, compute_metrics_batch
from utils.loss import ELDANetLoss

# ── Configuration ─────────────────────────────────────────────────────────────

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open(CONFIG_PATH, 'r', encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)


def get_data_paths():
    ds = CONFIG['dataset']
    if ds == 'TuSimple':
        return CONFIG['tusimple']['image_dir'], CONFIG['tusimple']['label_dir']
    elif ds == 'CULane':
        return CONFIG['culane']['image_dir'], CONFIG['culane']['label_dir']
    raise ValueError(f"Unknown dataset: {ds}")


def get_device():
    device_str = CONFIG.get('device', 'cuda')
    if device_str == 'cuda' and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        return torch.device('cpu')
    return torch.device(device_str)


def build_model(device):
    """Instantiate ELDA-Net U-Net with config hyperparameters."""
    model = UNet(
        in_channels=CONFIG.get('in_channels', 3),
        out_channels=CONFIG.get('out_channels', 1),
        dropout=CONFIG.get('dropout', 0.2),
    ).to(device)
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    print("[INFO] Starting ELDA-Net training...")
    device = get_device()
    image_dir, label_dir = get_data_paths()
    input_size = CONFIG['input_size']   # [H, W]

    dataset = LaneDataset(
        image_dir, label_dir, input_size, train=True,
        aug_cfg=CONFIG.get('augmentation'),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG.get('num_workers', 4),
        pin_memory=CONFIG.get('pin_memory', True),
        drop_last=True,
    )

    model     = build_model(device)
    criterion = ELDANetLoss(
        lambda_bce=CONFIG['loss']['lambda_bce'],
        lambda_dice=CONFIG['loss']['lambda_dice'],
        lambda_conf=CONFIG['loss']['lambda_conf'],
        lambda_poly=CONFIG['loss']['lambda_poly'],
    )

    opt_cfg   = CONFIG['optimizer']
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opt_cfg['lr'],
        betas=tuple(opt_cfg['betas']),
        weight_decay=opt_cfg.get('weight_decay', 1e-4),
    )

    sched_cfg = CONFIG['scheduler']
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=sched_cfg.get('T_max', 25),
        eta_min=sched_cfg.get('eta_min', 1e-5),
    )

    num_epochs = CONFIG['num_epochs']
    os.makedirs(os.path.dirname(CONFIG['model_save_path']), exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running = {'total': 0.0, 'seg': 0.0, 'dice': 0.0, 'conf': 0.0}
        n = 0

        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            seg_logits, conf_pred, poly_pred = model(imgs)
            loss, breakdown = criterion(seg_logits, conf_pred, poly_pred, masks)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            for k in running:
                running[k] += breakdown.get(k, 0.0)
            n += 1

        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Loss {running['total']/n:.4f} | "
              f"Seg {running['seg']/n:.4f} | "
              f"Dice {running['dice']/n:.4f} | "
              f"Conf {running['conf']/n:.4f} | "
              f"LR {lr_now:.2e}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = CONFIG['model_save_path'].replace('.pth', f'_{timestamp}.pth')
    torch.save(model.state_dict(), save_path)
    print(f"[INFO] Model saved to {save_path}")
    return save_path


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate():
    print("[INFO] Evaluating ELDA-Net...")
    device = get_device()
    image_dir, label_dir = get_data_paths()
    input_size = CONFIG['input_size']

    dataset = LaneDataset(image_dir, label_dir, input_size, train=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=CONFIG.get('num_workers', 4))

    model = build_model(device)
    ckpt  = CONFIG['model_save_path']
    if not os.path.exists(ckpt):
        print(f"[ERROR] Checkpoint not found: {ckpt}")
        print("        Run --mode train first, then update model_save_path in config.yaml")
        return
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    all_metrics = []
    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            seg_logits, conf_pred, _ = model(imgs)
            preds = (torch.sigmoid(seg_logits) > 0.5).cpu().numpy()
            m = compute_metrics(preds.ravel(), masks.cpu().numpy().ravel())
            all_metrics.append(m[:2])   # (f1, iou)

    avg = np.mean(all_metrics, axis=0)
    print(f"[RESULTS] F1:  {avg[0]:.4f}  (target: TuSimple 0.932 | CULane 0.876)")
    print(f"[RESULTS] IoU: {avg[1]:.4f}")


# ── Video Inference ───────────────────────────────────────────────────────────

def infer_on_video(video_path: str, output_path: str):
    print(f"[INFO] Running inference on: {video_path}")

    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return

    device = get_device()
    model  = build_model(device)
    ckpt   = CONFIG['model_save_path']
    if not os.path.exists(ckpt):
        print(f"[ERROR] Checkpoint not found: {ckpt}")
        return
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    alem_cfg   = CONFIG.get('alem', {})
    estimator  = AdaptiveLaneEstimator(alem_cfg)
    input_size = CONFIG['input_size']   # [H, W]
    norm_mean  = CONFIG.get('normalize', {}).get('mean', [0.485, 0.456, 0.406])
    norm_std   = CONFIG.get('normalize', {}).get('std',  [0.229, 0.224, 0.225])
    conf_threshold = alem_cfg.get('confidence_threshold', 0.70)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    writer = None

    frame_count = 0
    prev_time = time.time()
    fps_smooth = 0.0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        img_tensor = preprocess_image(frame, input_size,
                                      mean=norm_mean, std=norm_std)
        img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).to(device)

        with torch.no_grad():
            seg_logits, conf_pred, poly_pred = model(img_tensor)

        pred_mask = (torch.sigmoid(seg_logits).squeeze().cpu().numpy() > 0.5)
        confidence = conf_pred.item()
        poly_coeffs = poly_pred.squeeze().cpu().numpy()

        # ALEM refinement
        refined_mask = estimator.refine(pred_mask, confidence, poly_coeffs)

        # Resize refined mask back to original frame size for overlay
        h_orig, w_orig = frame.shape[:2]
        refined_display = cv2.resize(
            refined_mask.astype(np.uint8),
            (w_orig, h_orig), interpolation=cv2.INTER_NEAREST
        ).astype(np.float32)

        overlay = overlay_lanes(frame, refined_display,
                                confidence=confidence,
                                conf_threshold=conf_threshold)
        
        ## Added code here to display FPS ------------------------- 
        # Calculate FPS
        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now

        # Smooth FPS so it does not jump too much
        fps_smooth = 0.9 * fps_smooth + 0.1 * fps if fps_smooth > 0 else fps

        # Draw FPS on frame
        cv2.putText(
            overlay,
            f"FPS: {fps_smooth:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # Show live output window
        cv2.imshow("ELDA-Net Real-Time Inference", overlay)

        # Press q to quit early
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        ## Added code to display FPS ends here -------------------------

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, 20,
                                     (overlay.shape[1], overlay.shape[0]))
        writer.write(overlay)
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    
    if writer:
        writer.release()
    print(f"[INFO] Processed {frame_count} frames -> {output_path}")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ELDA-Net experiment pipeline')
    parser.add_argument('--mode',    choices=['train', 'eval', 'infer'], default='train')
    parser.add_argument('--video',   type=str, help='Input video path (infer mode)')
    parser.add_argument('--output',  type=str, help='Output video path (infer mode)')
    parser.add_argument('--dataset', choices=['TuSimple', 'CULane'],
                        help='Override dataset in config.yaml')
    args = parser.parse_args()

    if args.dataset:
        CONFIG['dataset'] = args.dataset

    if args.mode == 'train':
        train()
    elif args.mode == 'eval':
        evaluate()
    elif args.mode == 'infer':
        if not args.video or not args.output:
            print("[ERROR] --video and --output are required for infer mode.")
            sys.exit(1)
        infer_on_video(args.video, args.output)
