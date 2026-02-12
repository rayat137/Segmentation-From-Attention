#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.cuda.amp import autocast
from transformers import BertTokenizerFast, BlipConfig, BlipForImageTextRetrieval
import yaml

import transform as transform  # keep custom transform module
import dataset


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
    return p.parse_args()


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class PromptIndex:
    prompts: List[str]
    class_names: List[str]
    class_to_span: Dict[int, Tuple[int, int]]  # cls_id -> (start, end_exclusive)


def build_prompts(class_specs: Sequence[dict]) -> PromptIndex:

    prompts: List[str] = []
    class_names: List[str] = []
    class_to_span: Dict[int, Tuple[int, int]] = {}

    for cls_id, spec in enumerate(class_specs):
        start = len(prompts)
        class_names.append(spec["name"])

        prompts.append(f"Image of {spec['name']}.")

        # Compute infoScore in single prompt setting. 
        if cls_id == 0: 
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


class RunningWelford:
    """
    Running mean/variance per-dimension using Welford.
    Matches your original style (variance divided by n, not n-1).
    """
    def __init__(self, dim: int, device: torch.device):
        self.n = 0
        self.mean = torch.zeros(dim, device=device)
        self.M2 = torch.zeros(dim, device=device)

    def update(self, x: torch.Tensor) -> None:
        # x: [dim]
        self.n += 1
        delta = x - self.mean
        self.mean = self.mean + delta / self.n
        delta2 = x - self.mean
        self.M2 = self.M2 + delta * delta2

    def std(self) -> torch.Tensor:
        if self.n <= 0:
            return torch.zeros_like(self.mean)
        var = self.M2 / self.n
        return torch.sqrt(var.clamp_min(0.0))


def compute_infoscore_stats_for_layer(
    model: BlipForImageTextRetrieval,
    attention_mask: torch.Tensor,
    itm_scores: torch.Tensor,
    prompt_index: PromptIndex,
    layer: int,
    num_classes: int,
    new_embedding_res: int,
    image_size: int,
    eps: float,
) -> torch.Tensor:
    """
    Returns per-image marginal distribution for this layer: [B, C]
    using the same convention as training_free:
      - cls 0: SUM over its prompt span
      - cls >0: MAX over its prompt span
    """
    device = itm_scores.device

    attn = model.text_encoder.encoder.layer[layer].crossattention.self.get_attention_map()
    attn_max, _ = torch.max(attn, dim=1)  # [B, tokens, patches+1?]

    attn_max = attn_max[:, :, 1:]  # drop CLS patch
    attn_max = attn_max * attention_mask.unsqueeze(-1).to(device)
    attn_sum = torch.sum(attn_max, dim=1)  # [B, patches]

    attn_sum = attn_sum.unsqueeze(0)
    attn_sum = torch.nn.functional.softmax(attn_sum, dim=1) * itm_scores.unsqueeze(0).unsqueeze(-1)

    attn_map = attn_sum.reshape(-1, len(prompt_index.prompts), new_embedding_res, new_embedding_res)

    per_class = torch.nn.functional.interpolate(
        attn_map,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=True,
    )

    per_class = per_class / per_class.sum(dim=1, keepdim=True).clamp_min(eps)
    per_class = per_class.clamp_min(eps)

    # marginal over pixels
    marginal = torch.mean(per_class.flatten(2), dim=-1)  # [B, C]
    return marginal


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # Run params from config
    data_set = str(cfg.data_set)
    data_root = str(cfg.data_root)
    val_list = str(cfg.val_list)

    model_name = str(cfg.model.name)
    image_size = int(cfg.model.image_size)

    batch_size = int(cfg.dataloader.batch_size)
    num_workers = int(cfg.dataloader.num_workers)

    seed = int(cfg.runtime.seed)
    use_amp = bool(cfg.runtime.amp)

    # Dataset constants from config (mean/std/ignore_label)
    mean = list(cfg.dataset.mean) if hasattr(cfg, "dataset") else list(cfg.voc.mean)
    std = list(cfg.dataset.std) if hasattr(cfg, "dataset") else list(cfg.voc.std)
    ignore_label = int(cfg.dataset.ignore_label) if hasattr(cfg, "dataset") else int(cfg.voc.ignore_label)

    # Optional output
    out_csv = None
    if hasattr(cfg, "output") and hasattr(cfg.output, "csv"):
        out_csv = str(cfg.output.csv)

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this setup.")

    # Load class specs from JSON
    class_specs = dataset.load_class_specs(str(cfg.class_name_dir))
    if not class_specs or class_specs[0].get("name", "") == "":
        raise ValueError("Invalid class_specs. Expect background as class 0 with a non-empty name.")
    num_classes = len(class_specs)

    prompt_index = build_prompts(class_specs)
    print(f"Dataset: {data_set}")
    print(f"Classes: {num_classes} | Prompts: {len(prompt_index.prompts)}")
    num_prompts = len(prompt_index.prompts)
    # BLIP
    blip_cfg = BlipConfig.from_pretrained(model_name)
    blip_cfg.vision_config.image_size = image_size
    blip_cfg.output_hidden_states = True

    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BlipForImageTextRetrieval.from_pretrained(model_name).to(device)
    model.eval()

    num_layers = enable_cross_attention_saving(model)
    print(f"BLIP text encoder layers: {num_layers}")

    # Dataset / loader
    val_transform = make_val_transform(image_size=image_size, mean=mean, std=std)
    val_data = dataset.BaseData(
        mode="val",
        data_root=data_root,
        data_list=val_list,
        data_set=data_set,
        transform=val_transform,
        class_name_dir=str(cfg.class_name_dir),
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    # Encode prompts
    encoding = tokenizer(prompt_index.prompts, return_tensors="pt", truncation=True, padding=True)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    new_embedding_res = image_size // 16
    eps = 1e-6

    # Accumulators (layer-wise)
    layer_sum_image_entropy = torch.zeros(num_layers, device=device)                 # [L]
    layer_sum_marginal = torch.zeros(num_layers, num_prompts, device=device)        # [L,C]
    layer_welford = [RunningWelford(num_prompts, device=device) for _ in range(num_layers)]

    total_images = 0
    end = time.time()

    with torch.inference_mode():
        for it, batch in enumerate(val_loader, start=1):
            inp, _target = batch
            inp = inp.to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                outputs = model(
                    pixel_values=inp,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    interpolate_pos_encoding=True,
                    use_itm_head=True,
                )
                itm_scores_all = model.itm_head(outputs["question_embeds"]).softmax(dim=-1)[:, :, 1]
                itm_scores, _ = torch.max(itm_scores_all, dim=1)  # [B]

                for layer in range(num_layers):
                    marginal = compute_infoscore_stats_for_layer(
                        model=model,
                        attention_mask=attention_mask,
                        itm_scores=itm_scores,
                        prompt_index=prompt_index,
                        layer=layer,
                        num_classes=num_classes,
                        new_embedding_res=new_embedding_res,
                        image_size=image_size,
                        eps=eps,
                    )  # [B,C]

                    ent = -torch.sum(marginal * torch.log(marginal.clamp_min(eps)), dim=-1)  # [B]
                    layer_sum_image_entropy[layer] += ent.sum()
                    layer_sum_marginal[layer, :] += marginal.sum(dim=0)

                    for b in range(marginal.shape[0]):
                        layer_welford[layer].update(marginal[b])

            total_images += inp.shape[0]
            end = time.time()

            if it % 50 == 0:
                mean_ent = layer_sum_image_entropy / max(total_images, 1)
                ds_marg = layer_sum_marginal / max(total_images, 1)
                ds_ent = -torch.sum(ds_marg * torch.log(ds_marg.clamp_min(eps)), dim=-1)

                cov = []
                for l in range(num_layers):
                    std_l = layer_welford[l].std()
                    mu_l = layer_welford[l].mean.clamp_min(eps)
                    cov.append(torch.sum(std_l / mu_l))
                cov = torch.stack(cov)

                infoscore = (ds_ent / mean_ent.clamp_min(eps)) * cov
                best = torch.argsort(infoscore, descending=True)[:10]
                print(f"[iter {it}] top layers by InfoScore: {best.tolist()}")

    # Final stats
    mean_image_entropy = layer_sum_image_entropy / max(total_images, 1)  # [L]
    dataset_marginal = layer_sum_marginal / max(total_images, 1)         # [L,C]
    dataset_entropy = -torch.sum(dataset_marginal * torch.log(dataset_marginal.clamp_min(eps)), dim=-1)  # [L]

    cov = []
    for l in range(num_layers):
        std_l = layer_welford[l].std()
        mu_l = layer_welford[l].mean.clamp_min(eps)
        cov.append(torch.sum(std_l / mu_l))
    cov = torch.stack(cov)  # [L]

    entropy_ratio = dataset_entropy / mean_image_entropy.clamp_min(eps)
    infoscore = entropy_ratio * cov

    print("\n==== Final ====")
    print("Total images:", total_images)
    print("Mean Image Entropy per layer:", mean_image_entropy.detach().cpu())
    print("Dataset Entropy per layer:", dataset_entropy.detach().cpu())
    print("CoV per layer:", cov.detach().cpu())
    print("EntropyRatio per layer:", entropy_ratio.detach().cpu())
    print("InfoScore per layer:", infoscore.detach().cpu())

    vals, idxs = torch.sort(infoscore, descending=True)
    print("\nSorted layers by InfoScore (desc):")
    print(idxs.detach().cpu().tolist())



if __name__ == "__main__":
    main()
