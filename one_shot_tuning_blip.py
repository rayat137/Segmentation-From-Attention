#!/usr/bin/env python3
"""
one_shot.py

BLIP ITM-based prompt-attention segmentation with optional 1-shot tuning.

Supports:
- Training-free eval (same as training_free.py), plus
- 1-shot tuning of:
  - model.text_encoder.embeddings.word_embeddings
  - selected cross-attention layers (cfg.model.layers)

Example:
  python one_shot.py --config configs/pascal_blip_oneshot.yaml
  python one_shot.py --config configs/pascal_blip_oneshot.yaml --multi_prompt
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import random
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from transformers import BertTokenizerFast, BlipConfig, BlipForImageTextRetrieval
import yaml

import transform as transform  # keep custom transform module
import dataset
import convcrf
from util import AverageMeter, intersectionAndUnionGPU


DEFAULT_CRF_CONF = {
    "filter_size": 15,
    "blur": 4,
    "merge": True,
    "norm": "none",
    "weight": "vector",
    "unary_weight": 1,
    "weight_init": 0.2,
    "trainable": False,
    "convcomp": False,
    "logsoftmax": True,
    "softmax": True,
    # NOTE: your original 1-shot default_conf had final_softmax=True
    # but gausscrf.forward is used the same way; keep consistent with your 1-shot:
    "final_softmax": True,
    "pos_feats": {"sdims": 3, "compat": 3},
    "col_feats": {"sdims": 80, "schan": 0.1, "compat": 10, "use_bias": False},
    "trainable_bias": False,
    "pyinn": False,
}


def load_config(path: str) -> SimpleNamespace:
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    def to_ns(x):
        if isinstance(x, dict):
            return SimpleNamespace(**{k: to_ns(v) for k, v in x.items()})
        if isinstance(x, list):
            return [to_ns(v) for v in x]
        return x

    return to_ns(cfg_dict)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    p.add_argument("--multi_prompt", action="store_true", help="Enable multi prompt mode (default: False)")
    return p.parse_args()


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def set_seed(seed: int) -> None:
    # Keep your training_free style, but also add deterministic knobs (harmless for eval)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class PromptIndex:
    prompts: List[str]
    class_names: List[str]
    class_to_span: Dict[int, Tuple[int, int]]  # cls_id -> (start, end_exclusive)


def build_prompts(class_specs: Sequence[dict], is_multi_prompt: bool) -> PromptIndex:
    """
    Convention:
      - background (cls 0) always uses synonyms (if present)
      - others use synonyms only if is_multi_prompt=True
    """
    prompts: List[str] = []
    class_names: List[str] = []
    class_to_span: Dict[int, Tuple[int, int]] = {}

    for cls_id, spec in enumerate(class_specs):
        start = len(prompts)
        class_names.append(spec["name"])

        prompts.append(f"Image of {spec['name']}.")

        use_synonyms = is_multi_prompt or cls_id == 0
        if use_synonyms:
            for syn in spec.get("related_word", []):
                prompts.append(f"Image of {syn}.")

        end = len(prompts)
        class_to_span[cls_id] = (start, end)

    return PromptIndex(prompts=prompts, class_names=class_names, class_to_span=class_to_span)


def enable_cross_attention_saving(model: BlipForImageTextRetrieval) -> int:
    num_layers = len(model.text_encoder.encoder.layer)
    for i in range(num_layers):
        model.text_encoder.encoder.layer[i].crossattention.self.save_attention = True
    return num_layers


def make_val_transform(image_size: int, mean: List[float], std: List[float]):
    return transform.Compose(
        [
            transform.Resize(image_size),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std),
        ]
    )


def compute_unary_from_attention(
    model: BlipForImageTextRetrieval,
    attention_mask: torch.Tensor,
    itm_scores: torch.Tensor,
    prompt_index: PromptIndex,
    layers: Sequence[int],
    num_classes: int,
    new_embedding_res: int,
    image_size: int,
) -> torch.Tensor:
    """
    Builds per-class unary maps from saved cross-attention.

    Convention (same as training_free):
    - cls 0 (background): SUM over its prompt span
    - others: MAX over their prompt spans
    """
    device = itm_scores.device
    ensemble = 0.0

    for layer in layers:
        attn = model.text_encoder.encoder.layer[layer].crossattention.self.get_attention_map()
        attn_max, _ = torch.max(attn, dim=1)

        # drop CLS patch
        attn_max = attn_max[:, :, 1:]
        attn_max = attn_max * attention_mask.unsqueeze(-1).to(device)
        attn_sum = torch.sum(attn_max, dim=1)

        attn_sum = attn_sum.unsqueeze(0)
        attn_sum = torch.nn.functional.softmax(attn_sum, dim=1) * itm_scores.unsqueeze(0).unsqueeze(-1)

        attn_map = attn_sum.reshape(-1, len(prompt_index.prompts), new_embedding_res, new_embedding_res)

        per_class = torch.zeros(
            attn_map.shape[0],
            num_classes,
            attn_map.shape[2],
            attn_map.shape[3],
            device=device,
        )

        for cls_id in range(num_classes):
            start, end = prompt_index.class_to_span[cls_id]
            if cls_id == 0:
                per_class[:, cls_id, :, :] = torch.sum(attn_map[:, start:end, :, :], dim=1)
            else:
                per_class[:, cls_id, :, :], _ = torch.max(attn_map[:, start:end, :, :], dim=1)

        per_class = torch.nn.functional.interpolate(
            per_class,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=True,
        )

        ensemble = ensemble + per_class

    ensemble = ensemble / max(len(layers), 1)
    denom = ensemble.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return ensemble / denom


def oneshot_tune(
    model: BlipForImageTextRetrieval,
    support_images: torch.Tensor,   # [N, 3, H, W]
    support_labels: torch.Tensor,   # [N, H, W]
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_index: PromptIndex,
    layers: Sequence[int],
    num_classes: int,
    new_embedding_res: int,
    image_size: int,
    ignore_label: int,
    num_iterations: int,
    lr: float,
    weight_decay: float,
    use_amp: bool,
    batch_gradient_accumulation: int = 1,
) -> None:
    """
    1-shot tuning that mirrors your original code:
      - optimize word embeddings + selected cross-attention layers
      - loss: NLL over log(unary)
      - update every image (equivalent to (cls+1)%1==0)
    """
    device = support_images.device

    cls_weights = torch.ones(num_classes, device=device)
    cls_weights[0] = 0.5
    criterion = nn.NLLLoss(ignore_index=ignore_label, weight=cls_weights)

    params = [{"params": model.text_encoder.embeddings.word_embeddings.parameters()}]
    for layer in layers:
        params.append({"params": model.text_encoder.encoder.layer[layer].crossattention.parameters()})

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    model.train()

    for it in range(num_iterations):
        unary_maps = []
        start_ind =0 
        for n in range(support_images.shape[0]):
            with autocast(enabled=use_amp):
                outputs = model(
                    pixel_values=support_images[n].unsqueeze(0),
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    interpolate_pos_encoding=True,
                )
                itm_scores_all = model.itm_head(outputs["question_embeds"]).softmax(dim=-1)[:, :, 1]
                itm_scores, _ = torch.max(itm_scores_all, dim=1)

                unary = compute_unary_from_attention(
                    model=model,
                    attention_mask=attention_mask,
                    itm_scores=itm_scores,
                    prompt_index=prompt_index,
                    layers=layers,
                    num_classes=num_classes,
                    new_embedding_res=new_embedding_res,
                    image_size=image_size,
                )
                unary_maps.append(unary)
                print(n+1)
                if (n+1)%batch_gradient_accumulation==0 or n+1==support_images.shape[0]:
                    unary = torch.cat(unary_maps,dim=0)
                    unary_maps.clear()
                    log_unary = torch.log(unary.clamp_min(1e-12))
                    loss = criterion(log_unary, support_labels[start_ind:n+1])           
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    print(f"Tune iter {it} | img {n+1}/{support_images.shape[0]} | loss {loss.item():.6f}")
                    start_ind = n+1
    model.eval()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    is_multi_prompt = args.multi_prompt

    # Run params from config
    data_set = str(cfg.data_set)
    data_root = str(cfg.data_root)

    model_name = str(cfg.model.name)
    image_size = int(cfg.model.image_size)
    layers = list(cfg.model.layers)
    batch_gradient_accumulation = int(cfg.oneshot.batch_gradient_accumulation)

    num_workers = int(cfg.dataloader.num_workers)

    crf_iters = int(cfg.crf.iters)

    seed = int(cfg.runtime.seed)
    use_amp = bool(cfg.runtime.amp)

    # Dataset constants
    mean = list(cfg.dataset.mean)
    std = list(cfg.dataset.std)
    ignore_label = int(cfg.dataset.ignore_label)

    # Class list
    class_specs = dataset.load_class_specs(cfg.class_name_dir)
    if not class_specs or class_specs[0].get("name", "") == "":
        raise ValueError("Invalid class_specs. Expect background as class 0 with a non-empty name.")
    num_classes = len(class_specs)

    # 1-shot config block
    oneshot = cfg.oneshot
    train_list = str(oneshot.train_list)
    test_list = str(oneshot.test_list)
    shot = int(oneshot.shot)
    split = int(oneshot.split)
    support_only_one_novel = bool(oneshot.support_only_one_novel)
    use_training_images_for_supports = bool(oneshot.use_training_images_for_supports)

    num_iterations = int(oneshot.num_iterations)
    lr = float(oneshot.lr)
    weight_decay = float(oneshot.weight_decay)
    total_runs = int(oneshot.total_runs)

    out_classwise_csv = str(cfg.output.classwise_csv)
    out_summary_csv = str(cfg.output.summary_csv)

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this setup (ConvCRF is used on GPU here).")

    prompt_index = build_prompts(class_specs, is_multi_prompt)
    print(f"Dataset: {data_set}")
    print(f"Classes: {num_classes} | Prompts: {len(prompt_index.prompts)}")
    print(f"Layers: {layers} | multi_prompt={is_multi_prompt}")

    # BLIP
    blip_cfg = BlipConfig.from_pretrained(model_name)
    blip_cfg.vision_config.image_size = image_size

    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BlipForImageTextRetrieval.from_pretrained(model_name).to(device)

    total_layers = enable_cross_attention_saving(model)
    if any(l < 0 or l >= total_layers for l in layers):
        raise ValueError(f"Invalid layer in {layers}. Model has {total_layers} layers.")

    # Encode prompts
    encoding = tokenizer(prompt_index.prompts, return_tensors="pt", truncation=True, padding=True)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    new_embedding_res = image_size // 16

    # transforms
    val_transform = make_val_transform(image_size=image_size, mean=mean, std=std)

    # episodic loader (same dataset you used for 1-shot)
    val_data = dataset.MultiClassValData(
        transform=val_transform,
        class_list=list(range(1, num_classes)),
        data_list_path_train=train_list,
        data_list_path_test=test_list,
        data_root=data_root,
        shot=shot,
        data_name=data_set,
        split=split,
        support_only_one_novel=support_only_one_novel,
        use_training_images_for_supports=use_training_images_for_supports,
    )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=1,
        drop_last=True,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=None,
    )

    unchanged_state_dict = copy.deepcopy(model.state_dict())

    runwise_miou: List[float] = []
    runwise_classwise_miou: List[np.ndarray] = []

    for run in range(total_runs):
        print(f"\n================ RUN {run}/{total_runs-1} ================")

        # CRF
        gausscrf = convcrf.GaussCRF(
            conf=DEFAULT_CRF_CONF,
            shape=[image_size, image_size],
            nclasses=num_classes,
            use_gpu=True,
        ).to(device)

        # ---- Generate support set (same call you used) ----
        spprt_imgs, s_label = val_loader.dataset.generate_support([], remove_them_from_query_data_list=True)
        support_images = spprt_imgs.to(device)
        support_labels = s_label.to(device)

        # ---- 1-shot tuning ----
        oneshot_tune(
            model=model,
            support_images=support_images,
            support_labels=support_labels,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_index=prompt_index,
            layers=layers,
            num_classes=num_classes,
            new_embedding_res=new_embedding_res,
            image_size=image_size,
            ignore_label=ignore_label,
            num_iterations=num_iterations,
            lr=lr,
            weight_decay=weight_decay,
            use_amp=use_amp,
            batch_gradient_accumulation=batch_gradient_accumulation,
        )

        # ---- Evaluate (same eval structure as training_free, but MultiClassValData yields 4 items) ----
        batch_time = AverageMeter()
        data_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()

        iter_num = 0
        total_intersection = 0.0
        total_union = 0.0
        end = time.time()

        with torch.inference_mode():
            for it, batch in enumerate(val_loader, start=1):
                data_time.update(time.time() - end)

                inp, target, _, _ = batch  # MultiClassValData returns 4
                inp = inp.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                with autocast(enabled=use_amp):
                    outputs = model(
                        pixel_values=inp,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        interpolate_pos_encoding=True,
                        use_itm_head=True,
                    )
                    itm_scores_all = model.itm_head(outputs["question_embeds"]).softmax(dim=-1)[:, :, 1]
                    itm_scores, _ = torch.max(itm_scores_all, dim=1)

                    unary = compute_unary_from_attention(
                        model=model,
                        attention_mask=attention_mask,
                        itm_scores=itm_scores,
                        prompt_index=prompt_index,
                        layers=layers,
                        num_classes=num_classes,
                        new_embedding_res=new_embedding_res,
                        image_size=image_size,
                    )

                unary = gausscrf.forward(unary=unary, img=inp, num_iter=crf_iters)
                _, pred = unary.max(dim=1)

                inter, uni, new_t = intersectionAndUnionGPU(pred, target, num_classes, ignore_label)
                inter = inter.cpu().squeeze().float().numpy()
                uni = uni.cpu().squeeze().float().numpy()
                new_t = new_t.cpu().squeeze().float().numpy()

                intersection_meter.update(inter)
                union_meter.update(uni)
                target_meter.update(new_t)

                total_intersection += intersection_meter.val
                total_union += union_meter.val

                iou_per_class = total_intersection / (total_union + 1e-16)
                mean_iou = float(np.nanmean(iou_per_class))

                batch_time.update(time.time() - end)
                end = time.time()

                iter_num += 1
                print(
                    f"Test: [{iter_num}/{len(val_loader)}] "
                    f"Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                    f"Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                    f"mIoU {mean_iou:.4f}"
                )

        iou_per_class = total_intersection / (total_union + 1e-16)
        mean_iou = float(np.nanmean(iou_per_class))
        print(f"RUN {run} Final mIoU: {mean_iou:.4f}")

        # record + reset (same as your original)
        runwise_miou.append(mean_iou)
        runwise_classwise_miou.append(iou_per_class.copy())

        model.load_state_dict(unchanged_state_dict)

    # ---- write outputs ----
    runwise_miou_arr = np.array(runwise_miou)
    runwise_classwise_miou_arr = np.array(runwise_classwise_miou)

    print("Mean mIoU across runs:", float(np.mean(runwise_miou_arr)))

    ensure_parent_dir(out_classwise_csv)
    with open(out_classwise_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["Classname"] + [f"IOU Run_{r}" for r in range(total_runs)]
        w.writerow(header)
        for ci, cname in enumerate(prompt_index.class_names):
            row = [cname] + [float(runwise_classwise_miou_arr[r, ci]) for r in range(total_runs)]
            w.writerow(row)
    print(f"Wrote: {out_classwise_csv}")

    ensure_parent_dir(out_summary_csv)
    with open(out_summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Run", "mIoU"])
        for r in range(total_runs):
            w.writerow([r, float(runwise_miou_arr[r])])
        w.writerow(["mean", float(np.mean(runwise_miou_arr))])
        w.writerow(["std", float(np.std(runwise_miou_arr))])
    print(f"Wrote: {out_summary_csv}")


if __name__ == "__main__":
    main()
