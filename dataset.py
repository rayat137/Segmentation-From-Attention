# dataset.py
from __future__ import annotations

import json
import math
import os
import random
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import transform


# ----------------------------
# Class name JSON loading
# ----------------------------

def load_class_specs(class_name_dir: str) -> List[dict]:
    """
    Reads a JSON file containing a list[dict] where each dict has at least:
      - "id": int
      - "name": str
    Returns the list in the file order.
    """
    if not class_name_dir:
        raise ValueError("class_name_dir is required (path to JSON class spec list).")
    if not os.path.isfile(class_name_dir):
        raise FileNotFoundError(f"class_name_dir not found: {class_name_dir}")

    with open(class_name_dir, "r") as f:
        specs = json.load(f)

    if not isinstance(specs, list) or not all(isinstance(x, dict) for x in specs):
        raise ValueError(f"Invalid JSON format in {class_name_dir}: expected list[dict].")

    # Basic validation
    for i, s in enumerate(specs):
        if "name" not in s:
            raise ValueError(f"Missing 'name' in class spec index {i} ({class_name_dir}).")
        if "id" not in s:
            # allow implicit ids, but it's safer to keep them explicit
            s["id"] = i

    return specs


def class_names_from_specs(specs: List[dict]) -> List[str]:
    return [s["name"] for s in specs]


# ----------------------------
# BaseData (val/train image+mask)
# ----------------------------

class BaseData(Dataset):
    """
    Simple dataset used by training_free.py:
      returns (image, label) where label is grayscale with:
        - 0 = background
        - 255 = ignore
        - 1..K are foreground classes (Pascal) OR 0..79 (COCO) depending on your masks

    This class does NOT remap labels.
    It just loads the image/label and applies transform.
    """

    def __init__(
        self,
        mode: str,
        data_root: str,
        data_list: str,
        data_set: str,
        transform=None,
        class_name_dir: str | None = None,
    ):
        assert mode in ["train", "val"], f"mode must be train/val, got {mode}"
        assert data_set in ["pascal", "coco"], f"data_set must be pascal/coco, got {data_set}"

        self.mode = mode
        self.data_root = data_root
        self.data_list_path = data_list
        self.data_set = data_set
        self.transform = transform

        # Load class specs (optional but recommended)
        self.class_specs: List[dict] = []
        self.class_names: List[str] = []
        if class_name_dir is not None:
            self.class_specs = load_class_specs(class_name_dir)
            self.class_names = class_names_from_specs(self.class_specs)

        self.data_list: List[Tuple[str, str]] = []
        if not os.path.isfile(self.data_list_path):
            raise RuntimeError(f"Image list file does not exist: {self.data_list_path}")

        list_read = open(self.data_list_path).readlines()
        for line in list_read:
            line = line.strip()
            if not line:
                continue
            line_split = line.split(" ")
            image_name = os.path.join(self.data_root, line_split[0])
            label_name = os.path.join(self.data_root, line_split[1])
            self.data_list.append((image_name, label_name))

    def __len__(self):
        if self.data_set == 'coco':
            return 10000 
        return len(self.data_list)


    def __getitem__(self, index: int):
        image_path, label_path = self.data_list[index]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image)

        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise RuntimeError(f"Failed to read label: {label_path}")

        if image.shape[:2] != label.shape[:2]:
            raise RuntimeError(f"Image/label shape mismatch: {image_path} vs {label_path}")

        if self.transform is not None:
            image, label = self.transform(image, label)

        return image, label


# ----------------------------
# Helpers used by MultiClassValData
# ----------------------------

def get_image_and_label(image_path: str, label_path: str) -> Tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.float32(image)

    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    if label is None:
        raise RuntimeError(f"Failed to read label: {label_path}")

    if image.shape[:2] != label.shape[:2]:
        raise RuntimeError(f"Image/label shape mismatch: {image_path} vs {label_path}")

    return image, label


def make_dataset(
    data_root: str,
    data_list: str,
    class_list: List[int],
    remove_images_with_undesired_classes: bool = False,
    keep_small_area_classes: bool = False,
) -> Tuple[List[Tuple[str, str]], Dict[int, List[Tuple[str, str]]]]:
    """
    Filters a list file into:
      - image_label_list: all images that contain >=1 class from class_list
      - class_file_dict: per-class list of images containing that class
    Assumes:
      - background = 0
      - ignore = 255
    """
    if not os.path.isfile(data_list):
        raise RuntimeError(f"Image list file does not exist: {data_list}")

    list_read = open(data_list).readlines()

    image_label_list: List[Tuple[str, str]] = []
    class_file_dict: Dict[int, List[Tuple[str, str]]] = defaultdict(list)

    process_partial = partial(
        process_image,
        data_root=data_root,
        class_list=class_list,
        remove_images_with_undesired_classes=remove_images_with_undesired_classes,
        keep_small_area_classes=keep_small_area_classes,
    )

    with Pool(os.cpu_count() // 2) as pool:
        for sublist, subdict in pool.map(process_partial, list_read):
            image_label_list += sublist
            for k, v in subdict.items():
                class_file_dict[k] += v
        pool.close()
        pool.join()

    return image_label_list, class_file_dict


def process_image(
    line: str,
    data_root: str,
    class_list: List[int],
    remove_images_with_undesired_classes: bool,
    keep_small_area_classes: bool,
) -> Tuple[List[Tuple[str, str]], Dict[int, List[Tuple[str, str]]]]:
    line = line.strip()
    if not line:
        return [], {}

    line_split = line.split(" ")
    image_name = os.path.join(data_root, line_split[0])
    label_name = os.path.join(data_root, line_split[1])
    item = (image_name, label_name)

    label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
    if label is None:
        return [], {}

    label_class = np.unique(label).tolist()
    if 0 in label_class:
        label_class.remove(0)
    if 255 in label_class:
        label_class.remove(255)

    new_label_class: List[int] = []
    for c in label_class:
        if c in class_list:
            tmp = np.zeros_like(label)
            tmp[label == c] = 1
            if tmp.sum() >= 16 * 32 * 32 or keep_small_area_classes:
                new_label_class.append(c)
        elif remove_images_with_undesired_classes:
            new_label_class = []
            break

    label_class = new_label_class

    image_label_list: List[Tuple[str, str]] = []
    class_file_dict: Dict[int, List[Tuple[str, str]]] = defaultdict(list)

    if len(label_class) > 0:
        image_label_list.append(item)
        for c in label_class:
            class_file_dict[c].append(item)

    return image_label_list, class_file_dict


# ----------------------------
# MultiClassValData (kept)
# ----------------------------

class MultiClassValData(Dataset):
    def __init__(
        self,
        transform: transform.Compose,
        class_list: List[int],
        data_list_path_train: str,
        data_list_path_test: str,
        **kwargs,
    ):
        self.support_only_one_novel = kwargs["support_only_one_novel"]
        self.use_training_images_for_supports = kwargs["use_training_images_for_supports"]
        assert (not self.use_training_images_for_supports) or data_list_path_train

        self.data_name = kwargs["data_name"]  # 'pascal' or 'coco'
        self.shot = kwargs["shot"]
        self.data_root = kwargs["data_root"]
        self.class_list = class_list

        split = kwargs["split"]
        split_query_list = f"{self.data_name}_split{split}_query.npy"
        split_support_list = f"{self.data_name}_split{split}_support.npy"

        random.seed(1)

        if os.path.exists(split_query_list):
            np_dict = np.load(split_query_list, allow_pickle=True).item()
            self.query_data_list = np_dict["query_list"]
        else:
            self.query_data_list, _ = make_dataset(
                self.data_root,
                data_list_path_test,
                self.class_list,
                keep_small_area_classes=True,
            )
            np.save(split_query_list, {"query_list": self.query_data_list})

        self.complete_query_data_list = self.query_data_list.copy()
        print("Total number of kept images (query):", len(self.query_data_list))

        if os.path.exists(split_support_list):
            np_dict = np.load(split_support_list, allow_pickle=True).item()
            support_data_list = np_dict["support_data_list"]
            self.support_sub_class_file_list = np_dict["support_sub_cls_list"]
        else:
            support_list_path = data_list_path_train if self.use_training_images_for_supports else data_list_path_test
            support_data_list, self.support_sub_class_file_list = make_dataset(
                self.data_root,
                support_list_path,
                self.class_list,
                keep_small_area_classes=False,
            )
            np.save(
                split_support_list,
                {"support_data_list": support_data_list, "support_sub_cls_list": self.support_sub_class_file_list},
            )

        print("Total number of kept images (support):", len(support_data_list))
        self.transform = transform

    def __len__(self):

        if self.data_name == 'coco':
            return 10000 
        return len(self.query_data_list)

    def set_base_to_zero(self, label: torch.Tensor) -> torch.Tensor:
        mask = torch.isin(label, torch.tensor(self.class_list, device=label.device)) | (label == 255)
        label = label.clone()
        label[~mask] = 0
        return label

    def __getitem__(self, index: int):
        image_path, label_path = self.query_data_list[index]
        qry_img, label_np = get_image_and_label(image_path, label_path)

        if self.transform is not None:
            qry_img, label = self.transform(qry_img, label_np)
        else:
            # fallback to tensor conversion if needed (but your code always passes transform)
            qry_img = torch.from_numpy(qry_img).permute(2, 0, 1).float()
            label = torch.from_numpy(label_np).long()

        label = self.set_base_to_zero(label)
        valid_pixels = (label != 255).float()
        return qry_img, label, valid_pixels, image_path

    def generate_support(self, query_image_path_list: List[str], remove_them_from_query_data_list: bool = False):
        image_list, label_list = [], []
        support_image_path_list, support_label_path_list = [], []

        for c in self.class_list:
            file_class_chosen = self.support_sub_class_file_list[c]
            num_file = len(file_class_chosen)
            indices_list = list(range(num_file))
            random.shuffle(indices_list)

            current_path_list = []
            for idx in indices_list:
                if len(current_path_list) >= self.shot:
                    break

                image_path, label_path = file_class_chosen[idx]
                if image_path in (query_image_path_list + current_path_list):
                    continue

                image, label_np = get_image_and_label(image_path, label_path)

                if self.support_only_one_novel:
                    present = set(np.unique(label_np)) - {0, 255}
                    if len(present) > 1:
                        continue

                image_list.append(image)
                label_list.append(label_np)
                current_path_list.append(image_path)
                support_image_path_list.append(image_path)
                support_label_path_list.append(label_path)

            found = len(current_path_list)
            if found == 0:
                print(f"No support candidate for class {c} out of {num_file} images")
                idx = math.floor(random.random() * num_file)
                image_path, label_path = file_class_chosen[idx]
                image, label_np = get_image_and_label(image_path, label_path)

                present = set(np.unique(label_np)) - {0, c, 255}
                for nc in present:
                    label_np[label_np == nc] = 0

                image_list.append(image)
                label_list.append(label_np)
                found = 1

            if found < self.shot:
                print(f"Found {found} images for class {c} instead of {self.shot} shots")
                while found < self.shot:
                    idx = math.floor(random.random() * num_file)
                    image_path, label_path = file_class_chosen[idx]
                    image, label_np = get_image_and_label(image_path, label_path)

                    present = set(np.unique(label_np)) - {0, c, 255}
                    for nc in present:
                        label_np[label_np == nc] = 0

                    image_list.append(image)
                    label_list.append(label_np)
                    found += 1

        transformed_images, transformed_labels = [], []
        for i, l in zip(image_list, label_list):
            ti, tl = self.transform(i, l)
            transformed_images.append(ti.unsqueeze(0))
            transformed_labels.append(tl.unsqueeze(0))

        spprt_imgs = torch.cat(transformed_images, 0)
        spprt_labels = torch.cat(transformed_labels, 0)

        if remove_them_from_query_data_list and not self.use_training_images_for_supports:
            self.query_data_list = self.complete_query_data_list.copy()
            for i, l in zip(support_image_path_list, support_label_path_list):
                self.query_data_list.remove((i, l))

        return spprt_imgs, spprt_labels
