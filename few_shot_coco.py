# torchvision.transforms as transform
import transform as transform
import dataset 
import torch
import torch.nn as nn
import numpy as np
from transformers import BlipConfig, BlipForImageTextRetrieval, BlipVisionConfig, BlipProcessor, BertTokenizerFast
import math
from torchvision import transforms
from PIL import ImageEnhance
from util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, get_model_para_number, fast_intersection_and_union, setup_seed, get_logger, get_save_path, \
                                    is_same_model, fix_bn, sum_list, check_makedirs
import time
from PIL import Image
import os
import random
import csv
import copy
import convcrf
from torch.autograd import Variable
from torch.cuda.amp import autocast, GradScaler


default_conf = {
    'filter_size': 15, #Default 11 #Best 15
    'blur': 4,  #DEFAULT 4
    'merge': True,
    'norm': 'none',
    'weight': 'vector',
    "unary_weight": 1,  #Default 1 
    "weight_init": 0.2, #Default 0.2

    'trainable': False,
    'convcomp': False,
    'logsoftmax': True,  # use logsoftmax for numerical stability; DFEAULT TRUE
    'softmax': True, #  DFEAULT TRUE
    'final_softmax': True,

    'pos_feats': {
        'sdims': 3,
        'compat': 3,
    },
    'col_feats': {
        'sdims': 80,
        'schan': 0.1,   # schan depend on the input scale.
                       # use schan = 13 for images in [0, 255]
                       # for normalized images in [-0.5, 0.5] try schan = 0.1
        'compat': 10,
        'use_bias': False
    },
    "trainable_bias": False,

    "pyinn": False
}



def mask_gradients(param, unique_ids):
    """
    Masks the gradients of a tensor so only specific indices are optimized.

    Args:
        param (torch.Tensor): The tensor whose gradients need to be masked.
        unique_ids (torch.Tensor): The indices to keep; all other gradients will be set to zero.
    """
    grad = param.grad  # Access the gradient of the parameter
    if grad is not None:  # Ensure gradients exist
        mask = torch.zeros_like(grad)  # Create a mask of the same shape as the gradient
        mask[unique_ids, :] = 1  # Set 1s for the rows corresponding to unique IDs
        grad *= mask  # Element-wise multiply the gradient by the mask


class_names = [  {"color": [0, 0, 0], "isthing": 1, "id": 0, "name": "sky", "related_word":[ "wall", "tree", "wood", "grass", "road", "sea", "river", "mountain", "sands", "desk", "building", "cloud", "lamp", "door", "window", "wardrobe", "ceiling", "shelf", "curtain", "stair", "floor", "hill", "rail", "fence"]}, 
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person", "related_word":["people", "man", "woman", "child", "children", "boy", "girl"]},    #"people", "man", "woman", "child", "children", "boy", "girl"
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bicycle", "related_word":["bicycles", "bike"]},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "car", "related_word":["cars"]},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "motorcycle", "related_word":["motorbike", "motorcycles"]},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "airplane", "related_word":["aeroplane", "airplanes", "aircraft"]},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "bus",  "related_word":["buses"] },
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "train"},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "truck", "related_word":["trucks"]},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "boat","related_word":["boats", "yacht", "ship", "speedboat"]},
    {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "traffic light",  "alternative":"trafficlight"},
    {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "fire hydrant",  "alternative":"firehydrant"},
    {"color": [220, 220, 0], "isthing": 1, "id": 13, "name": "stop sign", "alternative":"stopsign"},
    {"color": [175, 116, 175], "isthing": 1, "id": 14, "name": "parking meter", "alternative":"parkingmeter"},
    {"color": [250, 0, 30], "isthing": 1, "id": 15, "name": "bench",  "related_word":["benches"]},
    {"color": [165, 42, 42], "isthing": 1, "id": 16, "name": "bird", "related_word":["birds"]},
    {"color": [255, 77, 255], "isthing": 1, "id": 17, "name": "cat",  "related_word":["cats"]},
    {"color": [0, 226, 252], "isthing": 1, "id": 18, "name": "dog", "related_word":["dogs"]},
    {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse", "related_word":["horses"]},
    {"color": [0, 82, 0], "isthing": 1, "id": 20, "name": "sheep"},
    {"color": [120, 166, 157], "isthing": 1, "id": 21, "name": "cow", "related_word":["cattle", "cows"]},
    {"color": [110, 76, 0], "isthing": 1, "id": 22, "name": "elephant", "related_word":["elephants"]},
    {"color": [174, 57, 255], "isthing": 1, "id": 23, "name": "bear", "related_word":["bears"]},
    {"color": [199, 100, 0], "isthing": 1, "id": 24, "name": "zebra", "related_word":["zebras"]},
    {"color": [72, 0, 118], "isthing": 1, "id": 25, "name": "giraffe", "related_word":["girrafes"]},
    {"color": [255, 179, 240], "isthing": 1, "id": 27, "name": "backpack", "related_word":["knapsack", "rucksack"]},
    {"color": [0, 125, 92], "isthing": 1, "id": 28, "name": "umbrella", "related_word":["umbrellas"] },
    {"color": [209, 0, 151], "isthing": 1, "id": 31, "name": "handbag", "related_word":["briefcase"]},
    {"color": [188, 208, 182], "isthing": 1, "id": 32, "name": "necktie"},
    {"color": [0, 220, 176], "isthing": 1, "id": 33, "name": "suitcase"},
    {"color": [255, 99, 164], "isthing": 1, "id": 34, "name": "frisbee"},
    {"color": [92, 0, 73], "isthing": 1, "id": 35, "name": "skis","related_word":["ski"]},
    {"color": [133, 129, 255], "isthing": 1, "id": 36, "name": "snowboard",  "related_word":["snowboards"]},
    {"color": [78, 180, 255], "isthing": 1, "id": 37, "name": "sports ball", "related_word":["ball", "football", "soccer ball", "tennis ball"], "alternative":"sportsball"},
    {"color": [0, 228, 0], "isthing": 1, "id": 38, "name": "kite", "related_word":["kites"]},
    {"color": [174, 255, 243], "isthing": 1, "id": 39, "name": "baseball bat", "alternative":"baseballbat"},
    {"color": [45, 89, 255], "isthing": 1, "id": 40, "name": "baseball glove", "alternative":"baseballglove"},
    {"color": [134, 134, 103], "isthing": 1, "id": 41, "name": "skateboard", "related_word":["skateboards"]},
    {"color": [145, 148, 174], "isthing": 1, "id": 42, "name": "surfboard", "related_word":["surfboards"]},
    {"color": [255, 208, 186], "isthing": 1, "id": 43, "name": "tennis racket", "related_word":["racket"],  "alternative":"tennisracket"},
    {"color": [197, 226, 255], "isthing": 1, "id": 44, "name": "bottle",  "related_word":["bottles"]},
    {"color": [171, 134, 1], "isthing": 1, "id": 46, "name": "wine glass", "related_word":["wine glasses"], "alternative":"wineglass"},
    {"color": [109, 63, 54], "isthing": 1, "id": 47, "name": "cup",  "related_word":["cups"] },
    {"color": [207, 138, 255], "isthing": 1, "id": 48, "name": "fork" },
    {"color": [151, 0, 95], "isthing": 1, "id": 49, "name": "knife", "related_word":["knives"]},
    {"color": [9, 80, 61], "isthing": 1, "id": 50, "name": "spoon" }, 
    {"color": [84, 105, 51], "isthing": 1, "id": 51, "name": "bowl"},
    {"color": [74, 65, 105], "isthing": 1, "id": 52, "name": "banana", "related_word":["bananas"]},
    {"color": [166, 196, 102], "isthing": 1, "id": 53, "name": "apple", "related_word":["apples"] },
    {"color": [208, 195, 210], "isthing": 1, "id": 54, "name": "sandwich", "related_word":["sandwiches"]},
    {"color": [255, 109, 65], "isthing": 1, "id": 55, "name": "orange", "related_word":["oranges"]},
    {"color": [0, 143, 149], "isthing": 1, "id": 56, "name": "broccoli"},
    {"color": [179, 0, 194], "isthing": 1, "id": 57, "name": "carrot","related_word":["carrots"]},
    {"color": [209, 99, 106], "isthing": 1, "id": 58, "name": "hot dog"},
    {"color": [5, 121, 0], "isthing": 1, "id": 59, "name": "pizza"},
    {"color": [227, 255, 205], "isthing": 1, "id": 60, "name": "donut", "related_word":["donuts", "doughnut"]},
    {"color": [147, 186, 208], "isthing": 1, "id": 61, "name": "cake", "related_word":["cakes"]},
    {"color": [153, 69, 1], "isthing": 1, "id": 62, "name": "chair", "related_word":["chairs", "dining chair"]},
    {"color": [3, 95, 161], "isthing": 1, "id": 63, "name": "couch", "related_word":["sofa"]},
    {"color": [163, 255, 0], "isthing": 1, "id": 64, "name": "potted plant", "related_word":["plants", "indoor plants", "house plants"], "alternative":"pottedplant"},
    {"color": [119, 0, 170], "isthing": 1, "id": 65, "name": "bed"},
    {"color": [0, 182, 199], "isthing": 1, "id": 67, "name": "dining table",  "alternative":"diningtable"},
    {"color": [0, 165, 120], "isthing": 1, "id": 70, "name": "toilet"},
    {"color": [183, 130, 88], "isthing": 1, "id": 72, "name": "tv", "related_word":["television", "television set"]},
    {"color": [95, 32, 0], "isthing": 1, "id": 73, "name": "laptop" },
    {"color": [130, 114, 135], "isthing": 1, "id": 74, "name": "mouse", "related_word":["computer mouse"], "alternative":"mouse"},
    {"color": [110, 129, 133], "isthing": 1, "id": 75, "name": "remote"},
    {"color": [166, 74, 118], "isthing": 1, "id": 76, "name": "keyboard"},
    {"color": [219, 142, 185], "isthing": 1, "id": 77, "name": "cell phone", "related_word":["smartphone", "mobile phone"]},
    {"color": [79, 210, 114], "isthing": 1, "id": 78, "name": "microwave"},
    {"color": [178, 90, 62], "isthing": 1, "id": 79, "name": "oven"},
    {"color": [65, 70, 15], "isthing": 1, "id": 80, "name": "toaster"},
    {"color": [127, 167, 115], "isthing": 1, "id": 81, "name": "sink"},
    {"color": [59, 105, 106], "isthing": 1, "id": 82, "name": "refrigerator", "related_word":["fridge"]},
    {"color": [142, 108, 45], "isthing": 1, "id": 84, "name": "book", "related_word":["books"]},
    {"color": [196, 172, 0], "isthing": 1, "id": 85, "name": "clock"},
    {"color": [95, 54, 80], "isthing": 1, "id": 86, "name": "vase"},
    {"color": [128, 76, 255], "isthing": 1, "id": 87, "name": "scissors"},
    {"color": [201, 57, 1], "isthing": 1, "id": 88, "name": "teddy bear", "related_word":["teddy"]},
    {"color": [246, 0, 122], "isthing": 1, "id": 89, "name": "hair dryer", "related_word":["blow dryer"]},
    {"color": [191, 162, 208], "isthing": 1, "id": 90, "name": "toothbrush"}]

# class_names = [ {"color": [0, 0, 0], "isthing": 1, "id": 0, "name": "sky", "related_word":[ "wall", "tree", "wood", "grass", "road", "sea", "river", "mountain", "sands", "desk", "building", "cloud", "lamp", "door", "window", "wardrobe", "ceiling", "shelf", "curtain", "stair", "floor", "hill", "rail", "fence"]},  #background
#     {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},    #"people", "man", "woman", "child", "children", "boy", "girl"
#     {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bicycle"},
#     {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "car", "related_word":["cars"]},
#     {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "motorcycle"},
#     {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "airplane"},
#     {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "bus"},
#     {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "train"},
#     {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "truck"},
#     {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "boat"},
#     {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "traffic light",  "alternative":"trafficlight"},
#     {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "fire hydrant",  "alternative":"firehydrant"},
#     {"color": [220, 220, 0], "isthing": 1, "id": 13, "name": "stop sign", "alternative":"stopsign"},
#     {"color": [175, 116, 175], "isthing": 1, "id": 14, "name": "parking meter", "alternative":"parkingmeter"},
#     {"color": [250, 0, 30], "isthing": 1, "id": 15, "name": "bench"},
#     {"color": [165, 42, 42], "isthing": 1, "id": 16, "name": "bird"},
#     {"color": [255, 77, 255], "isthing": 1, "id": 17, "name": "cat"},
#     {"color": [0, 226, 252], "isthing": 1, "id": 18, "name": "dog"},
#     {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse"},
#     {"color": [0, 82, 0], "isthing": 1, "id": 20, "name": "sheep"},
#     {"color": [120, 166, 157], "isthing": 1, "id": 21, "name": "cow"},
#     {"color": [110, 76, 0], "isthing": 1, "id": 22, "name": "elephant"},
#     {"color": [174, 57, 255], "isthing": 1, "id": 23, "name": "bear"},
#     {"color": [199, 100, 0], "isthing": 1, "id": 24, "name": "zebra"},
#     {"color": [72, 0, 118], "isthing": 1, "id": 25, "name": "giraffe"},
#     {"color": [255, 179, 240], "isthing": 1, "id": 27, "name": "backpack",},
#     {"color": [0, 125, 92], "isthing": 1, "id": 28, "name": "umbrella" },
#     {"color": [209, 0, 151], "isthing": 1, "id": 31, "name": "handbag"},
#     {"color": [188, 208, 182], "isthing": 1, "id": 32, "name": "necktie"},
#     {"color": [0, 220, 176], "isthing": 1, "id": 33, "name": "suitcase"},
#     {"color": [255, 99, 164], "isthing": 1, "id": 34, "name": "frisbee"},
#     {"color": [92, 0, 73], "isthing": 1, "id": 35, "name": "ski"},
#     {"color": [133, 129, 255], "isthing": 1, "id": 36, "name": "snowboard"},
#     {"color": [78, 180, 255], "isthing": 1, "id": 37, "name": "sports ball", "alternative":"sportsball"},
#     {"color": [0, 228, 0], "isthing": 1, "id": 38, "name": "kite"},
#     {"color": [174, 255, 243], "isthing": 1, "id": 39, "name": "baseball bat", "alternative":"baseballbat"},
#     {"color": [45, 89, 255], "isthing": 1, "id": 40, "name": "baseball glove", "alternative":"baseballglove"},
#     {"color": [134, 134, 103], "isthing": 1, "id": 41, "name": "skateboard"},
#     {"color": [145, 148, 174], "isthing": 1, "id": 42, "name": "surfboard"},
#     {"color": [255, 208, 186], "isthing": 1, "id": 43, "name": "tennis racket",  "alternative":"tennisracket"},
#     {"color": [197, 226, 255], "isthing": 1, "id": 44, "name": "bottle"},
#     {"color": [171, 134, 1], "isthing": 1, "id": 46, "name": "wine glass", "alternative":"wineglass"},
#     {"color": [109, 63, 54], "isthing": 1, "id": 47, "name": "cup"},
#     {"color": [207, 138, 255], "isthing": 1, "id": 48, "name": "fork" },
#     {"color": [151, 0, 95], "isthing": 1, "id": 49, "name": "knife"},
#     {"color": [9, 80, 61], "isthing": 1, "id": 50, "name": "spoon" }, 
#     {"color": [84, 105, 51], "isthing": 1, "id": 51, "name": "bowl"},
#     {"color": [74, 65, 105], "isthing": 1, "id": 52, "name": "banana"},
#     {"color": [166, 196, 102], "isthing": 1, "id": 53, "name": "apple" },
#     {"color": [208, 195, 210], "isthing": 1, "id": 54, "name": "sandwich"},
#     {"color": [255, 109, 65], "isthing": 1, "id": 55, "name": "orange"},
#     {"color": [0, 143, 149], "isthing": 1, "id": 56, "name": "broccoli"},
#     {"color": [179, 0, 194], "isthing": 1, "id": 57, "name": "carrot"},
#     {"color": [209, 99, 106], "isthing": 1, "id": 58, "name": "hot dog"},
#     {"color": [5, 121, 0], "isthing": 1, "id": 59, "name": "pizza"},
#     {"color": [227, 255, 205], "isthing": 1, "id": 60, "name": "donut"},
#     {"color": [147, 186, 208], "isthing": 1, "id": 61, "name": "cake"},
#     {"color": [153, 69, 1], "isthing": 1, "id": 62, "name": "chair",},
#     {"color": [3, 95, 161], "isthing": 1, "id": 63, "name": "couch"},
#     {"color": [163, 255, 0], "isthing": 1, "id": 64, "name": "potted plant", "alternative":"pottedplant"},
#     {"color": [119, 0, 170], "isthing": 1, "id": 65, "name": "bed"},
#     {"color": [0, 182, 199], "isthing": 1, "id": 67, "name": "dining table",  "alternative":"diningtable"},
#     {"color": [0, 165, 120], "isthing": 1, "id": 70, "name": "toilet"},
#     {"color": [183, 130, 88], "isthing": 1, "id": 72, "name": "tv", },
#     {"color": [95, 32, 0], "isthing": 1, "id": 73, "name": "laptop" },
#     {"color": [130, 114, 135], "isthing": 1, "id": 74, "name": "mouse", "alternative":"mouse"},
#     {"color": [110, 129, 133], "isthing": 1, "id": 75, "name": "remote"},
#     {"color": [166, 74, 118], "isthing": 1, "id": 76, "name": "keyboard"},
#     {"color": [219, 142, 185], "isthing": 1, "id": 77, "name": "cell phone"},
#     {"color": [79, 210, 114], "isthing": 1, "id": 78, "name": "microwave"},
#     {"color": [178, 90, 62], "isthing": 1, "id": 79, "name": "oven"},
#     {"color": [65, 70, 15], "isthing": 1, "id": 80, "name": "toaster"},
#     {"color": [127, 167, 115], "isthing": 1, "id": 81, "name": "sink"},
#     {"color": [59, 105, 106], "isthing": 1, "id": 82, "name": "refrigerator"},
#     {"color": [142, 108, 45], "isthing": 1, "id": 84, "name": "book"},
#     {"color": [196, 172, 0], "isthing": 1, "id": 85, "name": "clock"},
#     {"color": [95, 54, 80], "isthing": 1, "id": 86, "name": "vase"},
#     {"color": [128, 76, 255], "isthing": 1, "id": 87, "name": "scissors"},
#     {"color": [201, 57, 1], "isthing": 1, "id": 88, "name": "teddy bear",},
#     {"color": [246, 0, 122], "isthing": 1, "id": 89, "name": "hair dryer"},
#     {"color": [191, 162, 208], "isthing": 1, "id": 90, "name": "toothbrush"}]



#import pdb; pdb.set_trace()
query_string = [] 
classnames  = []
old_to_new_mapping = {}
new_to_old_mapping = {}
idx_counter = 0
for i, item in enumerate(class_names):
    query_string.append("Image of {0}.".format(item["name"] ))
    #NOTE: JUST TO TEST SYNONMYS
    if "related_word" in item:
            for syn in item["related_word"]:
                query_string.append("Image of {0}.".format(syn)) 
                idx_counter+=1
    classnames.append(item["name"])
    old_to_new_mapping[i] = idx_counter
    new_to_old_mapping[idx_counter] = i
    idx_counter+=1


print(query_string)
value_scale = 255
mean = [0.485, 0.456, 0.406]
#mean = [item * value_scale for item in mean]
std = [0.229, 0.224, 0.225]
#std = [item * value_scale for item in std]
data_set = 'coco'
data_root = '/ubc/cs/research/shield/datasets/MSCOCO/coco2014/'  ##NOTE: This is COCO-80
val_list = '/ubc/cs/research/shield/projects/rayat137/code/DiaM/lists/coco/val.txt'
train_list = '/ubc/cs/research/shield/projects/rayat137/code/DiaM/lists/coco/train.txt'

#torch.manual_seed(1)

resize_image_size = 512

val_transform = transform.Compose([
            transform.Resize(resize_image_size),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])


config = BlipConfig.from_pretrained("Salesforce/blip-itm-large-coco")
config.vision_config.image_size = resize_image_size
processor = BertTokenizerFast.from_pretrained("Salesforce/blip-itm-large-coco")
model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-coco")
#import pdb; pdb.set_trace()
pos_embedding_weights = model.vision_model.embeddings.position_embedding
new_embedding_res = config.vision_config.image_size // 16


unchanged_state_dict = copy.deepcopy(model.state_dict())
layers = [0, 3]
print("Chosen Layers: ", layers)

# num_layers = len(model.text_encoder.encoder.layer)

#import pdb; pdb.set_trace()
for i in layers: 
    model.text_encoder.encoder.layer[i].crossattention.self.save_attention = True 

# model.text_encoder.encoder.layer[0].crossattention.self.get_cross_attention.register_forward_hook(hook_fn)


#model.text_model.
model = model.cuda()

val_batch_size  = 1

seed = 124511 #124511
print(seed)
# Set the seed for NumPy
np.random.seed(seed)
random.seed(seed)
# Set the seed for PyTorch on CPU
torch.manual_seed(seed)

torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


val_data = dataset.MultiClassValData(transform=val_transform,
                                    class_list=list(range(1,81)),
                                    data_list_path_train=train_list,
                                    data_list_path_test=val_list,
                                    data_root=data_root,
                                    shot=1,
                                    data_name='coco',
                                    split=0,
                                    support_only_one_novel=True,
                                    use_training_images_for_supports=False)

val_sampler = None
#import pdb; pdb.set_trace()
val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, drop_last=True, shuffle=False, num_workers=1, pin_memory=True, sampler=val_sampler)

encoding = processor(query_string, return_tensors="pt", truncation=True, padding=True)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']


#import pdb; pdb.set_trace()

to_pil = transforms.ToPILImage()


mean = torch.Tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
#mean = [item * value_scale for item in mean]
std = torch.Tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

###NOTE: DO THE OPTIMIZATION PART HERE
device = torch.device('cuda:{}'.format(0))
num_iterations = 2 #2 #5 BEST SO FAR for batch size 4; 2 best so far for size 2

cls_weights = torch.ones(81)
cls_weights[0] = 0.5  # Best 0.25 so far for 1-shot
#import pdb; pdb.set_trace()
print(cls_weights)
criterion = nn.NLLLoss(ignore_index=255, weight=cls_weights.cuda())
#criterion = nn.CrossEntropyLoss(ignore_index=255, weight=cls_weights.cuda())
#params = []
torch.autograd.set_detect_anomaly(True)
#import pdb; pdb.set_trace()

scaler = GradScaler()
total_runs = 5
num_classes = 81
runwise_miou = []
runwise_classwise_miou = []

for run in range(total_runs):
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    iter_num = 0
    end = time.time()
    val_start = end

    #import pdb; pdb.set_trace()
    sum_IoU = 0
    iter = 0
    total_intersection = 0
    total_union = 0
    total_accuracy_sum = 0
    #layer_weights = nn.Parameter(torch.randn(num_layers, device='cuda', requires_grad=True))
    params = [{'params':model.text_encoder.embeddings.word_embeddings.parameters()}]
    #params.append({'params':layer_weights})

    for layer in layers: #range(num_layers):
        params.append({'params':model.text_encoder.encoder.layer[layer].crossattention.parameters()})
    optimizer = torch.optim.AdamW(params, lr=0.0002, weight_decay=0.05)  # 0.0002
    torch.autograd.set_detect_anomaly(True)
    gausscrf = convcrf.GaussCRF(conf=default_conf, shape=[config.vision_config.image_size,config.vision_config.image_size], nclasses=81, use_gpu=True).cuda()
    #scaler = GradScaler()
    spprt_imgs, s_label = val_loader.dataset.generate_support([], remove_them_from_query_data_list=True)
    images = spprt_imgs#.cuda()
    target = s_label#.cuda()
    #import pdb; pdb.set_trace()
    relu = torch.nn.ReLU()
    num_images = images.shape[0]
    
    model.train()
    for iterations in range(num_iterations):
        attention_maps = []
        start_ind = 0
        #print(layer_weights)
        for cls in range(num_images):
            with autocast(dtype=torch.float16):
                ensemble_attention_map = 0
                input_image = images[cls].unsqueeze(0).cuda()
                outputs = model(pixel_values=input_image, input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda(),  interpolate_pos_encoding=True)
                #outputs = model(pixel_values=images[cls].unsqueeze(0), input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda(),  interpolate_pos_encoding=True)
                itm_scores_all = model.itm_head(outputs['question_embeds']).softmax(dim=-1)[:,:,1]
                itm_scores, _ = torch.max(itm_scores_all,dim=1)
                
                for layer in layers:
                    average_attention_map = model.text_encoder.encoder.layer[layer].crossattention.self.get_attention_map()
                    max_attention_map, _ = torch.max(average_attention_map, dim =1)    
                    max_attention_map = max_attention_map[:,:,1:] * attention_mask.unsqueeze(-1).cuda() #* max_gradients[:,:,1:] 
                    max_attention_map = torch.sum(max_attention_map, dim =1) ## Words * Num_Patches

                    max_attention_map = max_attention_map.unsqueeze(0)
                    max_attention_map = torch.nn.functional.softmax(max_attention_map ,dim=1) *  itm_scores.unsqueeze(0).unsqueeze(-1)
                    max_attention_map = max_attention_map.reshape(-1, len(query_string),  new_embedding_res, new_embedding_res)
                    new_average_attention_map = torch.zeros(max_attention_map.shape[0], num_classes, max_attention_map.shape[2], max_attention_map.shape[3], device='cuda')
                
                    for cx in range(num_classes):  
                        if cx == 0:
                            start_idx = 0
                            end_idx = old_to_new_mapping[cx] + 1
                        else:
                            start_idx = old_to_new_mapping[cx-1] + 1
                            end_idx = old_to_new_mapping[cx] + 1
                        if cx==0:
                            new_average_attention_map[:,cx,:,:] = torch.sum(max_attention_map[:,start_idx:end_idx,:,:] , dim=1) 
                        else:
                            new_average_attention_map[:,cx,:,:], _ = torch.max(max_attention_map[:,start_idx:end_idx,:,:] , dim=1)

                    
                    average_attention_map = torch.nn.functional.interpolate(new_average_attention_map, size=(config.vision_config.image_size,config.vision_config.image_size),  mode='bilinear', align_corners=True )

                    ensemble_attention_map += average_attention_map #* relu(layer_weights[layer])


                ensemble_attention_map = ensemble_attention_map / len(layers)
                #ensemble_attention_map = (ensemble_attention_map   *  itm_scores.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))/torch.sum(relu(layer_weights))
                ensemble_attention_map = ensemble_attention_map/ (ensemble_attention_map.sum(dim=1).unsqueeze(1) +1e-14 ) 
                attention_maps.append(ensemble_attention_map)
            #import pdb; pdb.set_trace()
            if (cls+1)%2==0 or cls+1==num_images:# (cls+1)%10==0:
                #import pdb; pdb.set_trace()
                with autocast(dtype=torch.float16):
                    average_attention_map = torch.cat(attention_maps,dim=0)
                    attention_maps.clear()
                    average_attention_map = torch.log(average_attention_map)
                    output = average_attention_map
                    target_images =  target[start_ind:cls+1].cuda()
                    loss = criterion(output, target_images)
                #loss =  criterion(output, target[start_ind:cls+1]) #+  0.1 * torch.norm(layer_weights, p=1)  + 0.1 * torch.norm(layer_weights, p=2) # 1, 0.1, 0.1
                start_ind = cls+1
                print("Iteration: {0}; Start_ind:{1}; Loss: {2}".format(iterations, start_ind, loss))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # scaler.scale(loss).backward()
                # scaler.unscale_(optimizer)

                # scaler.step(optimizer)
                # scaler.update()

    ###NOTE: CODE FOR EVALUATION
    model.eval()
    gausscrf.eval()
    for i, logits in enumerate(val_loader):
        iter_num += 1
        data_time.update(time.time() - end)
        input, target,_ , _  = logits 
        input = input.cuda()
        target = target.cuda()
        tensor_image = input.cpu() * std + mean
        
        #print(num_layers)
        with torch.no_grad():
            with autocast():
                ensemble_attention_map = 0
                outputs = model(pixel_values=input, input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda(), interpolate_pos_encoding=True)
                itm_scores_all = model.itm_head(outputs['question_embeds']).softmax(dim=-1)[:,:,1]
                itm_scores, _ = torch.max(itm_scores_all,dim=1) 
                for layer in layers:
                    average_attention_map = model.text_encoder.encoder.layer[layer].crossattention.self.get_attention_map()
                    max_attention_map, _ = torch.max(average_attention_map, dim =1)    
                    max_attention_map = max_attention_map[:,:,1:] * attention_mask.unsqueeze(-1).cuda() 
                    max_attention_map = torch.sum(max_attention_map, dim =1) ## Words * Num_Patches

                    max_attention_map = max_attention_map.unsqueeze(0)
                    max_attention_map = torch.nn.functional.softmax(max_attention_map ,dim=1) *  itm_scores.unsqueeze(0).unsqueeze(-1)
                    max_attention_map = max_attention_map.reshape(-1, len(query_string),  new_embedding_res, new_embedding_res)
                    
                    new_average_attention_map = torch.zeros(max_attention_map.shape[0], num_classes, max_attention_map.shape[2], max_attention_map.shape[3], device='cuda')
            
                    for cx in range(num_classes):  
                        if cx == 0:
                            start_idx = 0
                            end_idx = old_to_new_mapping[cx] + 1
                        else:
                            start_idx = old_to_new_mapping[cx-1] + 1
                            end_idx = old_to_new_mapping[cx] + 1
                        if cx==0:
                            new_average_attention_map[:,cx,:,:] = torch.sum(max_attention_map[:,start_idx:end_idx,:,:] , dim=1) 
                        else:
                            new_average_attention_map[:,cx,:,:], _ = torch.max(max_attention_map[:,start_idx:end_idx,:,:] , dim=1)

                    
                    average_attention_map = torch.nn.functional.interpolate(new_average_attention_map, size=(config.vision_config.image_size,config.vision_config.image_size),  mode='bilinear', align_corners=True )
                    ensemble_attention_map += average_attention_map #* relu(layer_weights[layer])


                ensemble_attention_map = ensemble_attention_map / len(layers) #torch.sum(relu(layer_weights))
                #ensemble_attention_map = ensemble_attention_map   *  itm_scores.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)/torch.sum(relu(layer_weights))
                ensemble_attention_map = ensemble_attention_map/ ensemble_attention_map.sum(dim=1).unsqueeze(1)
                ensemble_attention_map = gausscrf.forward(unary=ensemble_attention_map, img=input,  num_iter=20)  #BEst so far 10

                output = torch.argmax(ensemble_attention_map, dim =1)

        #print(output.unique(), target.unique())
        intersection, union, new_target = intersectionAndUnionGPU(output, target, num_classes, 255)
        intersection, union, new_target = intersection.cpu().squeeze().float().numpy(), union.cpu().squeeze().float().numpy(), new_target.cpu().squeeze().float().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(new_target)

        total_intersection+=intersection_meter.val
        total_union+=union_meter.val

        IoU_per_class = total_intersection/ (total_union+1e-10)
        #print(IoU_per_class)
        mean_IoU = np.nanmean(IoU_per_class)

        #sum_IoU += IoU
        iter += 1
        #loss_meter.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        #mean_IoU = sum_IoU/ (iter * val_batch_size + 1e-10)
        print('Test: [{}/{}] '
                'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                #'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                'mIoU {mean_IoU:.4f}.'.format(iter_num, len(val_loader),
                                                data_time=data_time,
                                                batch_time=batch_time,
                                                #loss_meter=loss_meter,
                                                mean_IoU=mean_IoU))
        
        print(runwise_miou)

    IoU_per_class = total_intersection/ (total_union+1e-10)
    #print(IoU_per_class)
    mean_IoU = np.nanmean(IoU_per_class)

    print('Test: [{}/{}] '
                'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                #'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                'mIoU {mean_IoU:.4f}.'.format(iter_num, len(val_loader),
                                                data_time=data_time,
                                                batch_time=batch_time,
                                                #loss_meter=loss_meter,
                                                mean_IoU=mean_IoU))
    

    model.load_state_dict(unchanged_state_dict)
    #import pdb; pdb.set_trace()
    runwise_miou.append(mean_IoU)
    print(runwise_miou)
    runwise_classwise_miou.append(IoU_per_class)


runwise_miou = np.array(runwise_miou)
print(np.mean(runwise_miou))
runwise_classwise_miou = np.array(runwise_classwise_miou)


filename = 'Final_Results/BLIP_COCO_80_SYNONYMS_FT_2_iter_3_shots_0.5_back_lr_0.0002_no_syns_b4_full.csv'
with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    data_csv = ['Classname']
    for run in range(total_runs):
        data_csv.append('IOU Run_{0}'.format(run))
    writer.writerow(data_csv)  # Writing header

    for i in range(len(classnames)):
        data_csv = [classnames[i]]
        for run in range(total_runs):
            data_csv.append(runwise_classwise_miou[run, i])
        writer.writerow(data_csv)