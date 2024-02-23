import shutil
import sys

from torchvision import transforms

from models.eval_network import EvalKpSFR
from models.inference_core import InferenceCore
from robust.models.model import EncDec

sys.path.append('..')
from options import CustomOptions
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import os.path as osp
import time
from PIL import Image
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import utils
import metrics
import skimage.segmentation as ss

import random
import cv2
from PIL import Image
from shapely.geometry import Point, Polygon, MultiPoint
import utils
import torch

# Get input arguments
opt = CustomOptions(train=False)
opt = opt.parse()

# Setup GPU
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
print('CUDA Visible Devices: %s' % opt.gpu_ids)
device = torch.device('cuda:0')
print('device: %s' % device)


def postprocessing(scores, pred, num_classes, nms_thres):
    # TODO: decode the heatmaps into keypoint sets using non-maximum suppression
    pred_cls_dict = {k: [] for k in range(1, num_classes)}

    for cls in range(1, num_classes):
        pred_inds = pred == cls

        # implies the current class does not appear in this heatmaps
        if not np.any(pred_inds):
            continue

        values = scores[pred_inds]
        max_score = values.max()
        max_index = values.argmax()

        indices = np.where(pred_inds)
        coords = list(zip(indices[0], indices[1]))

        # the only keypoint with max confidence is greater than threshold or not
        if max_score >= nms_thres:
            pred_cls_dict[cls].append(max_score)
            pred_cls_dict[cls].append(coords[max_index])

    return pred_cls_dict


def class_mapping(rgb):
    # TODO: class mapping
    template = utils.gen_template_grid()  # grid shape (91, 3), (x, y, label)
    src_pts = rgb.copy()
    cls_map_pts = []

    for ind, elem in enumerate(src_pts):
        coords = np.where(elem[2] == template[:, 2])[0]  # find correspondence
        cls_map_pts.append(template[coords[0]])
    dst_pts = np.array(cls_map_pts, dtype=np.float32)

    return src_pts[:, :2], dst_pts[:, :2]


def test():
    num_classes = 92
    non_local = bool(opt.use_non_local)
    layers = 18

    frame_id = 0

    cap = cv2.VideoCapture('dataset/6secTest.mp4')
    model = EncDec(layers, num_classes, non_local).to(device)

    class_weights = torch.ones(num_classes) * 100
    class_weights[0] = 1
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights)  # TODO: put class weight

    # Set data path
    denorm = utils.UnNormalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    exp_name_path = osp.join(opt.checkpoints_dir, opt.name)
    test_visual_dir = osp.join(exp_name_path, 'imgs', 'test_visual')
    os.makedirs(test_visual_dir, exist_ok=True)

    iou_visual_dir = osp.join(test_visual_dir, 'iou')
    os.makedirs(iou_visual_dir, exist_ok=True)

    homo_visual_dir = osp.join(test_visual_dir, 'homography')
    os.makedirs(homo_visual_dir, exist_ok=True)

    field_model = Image.open(
        osp.join(opt.template_path, 'worldcup_field_model.png'))

    if len(opt.ckpt_path) > 0:
        load_weights_path = opt.ckpt_path
        print('Loading weights: ', load_weights_path)
        assert osp.isfile(load_weights_path), 'Error: no checkpoints found'
        checkpoint = torch.load(load_weights_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        print('Checkpoint Epoch: ', epoch)

    while True:
        ret_val, image = cap.read()

        image = cv2.resize(image, (1280, 720))


        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])  # ImageNet
        ])

        image_tensor = preprocess(image)

        image = image_tensor.to(device)
        if len(image.size()) == 3:
            image = image.unsqueeze(0)
        pred_heatmap = model(image)

        pred_heatmap = torch.softmax(pred_heatmap, dim=1)
        scores, pred_heatmap = torch.max(pred_heatmap, dim=1)
        scores = scores[0].detach().cpu().numpy()
        pred_heatmap = pred_heatmap[0].detach().cpu().numpy()
        pred_cls_dict = postprocessing(scores, pred_heatmap, num_classes, opt.nms_thres)

        image = utils.im_to_numpy(denorm(image[0]))
        pred_keypoints = np.zeros_like(
            pred_heatmap, dtype=np.uint8)
        pred_rgb = []
        for ind, (pk, pv) in enumerate(pred_cls_dict.items()):
            if pv:
                pred_keypoints[pv[1][0], pv[1][1]] = pk  # (H, W)
                # camera view point sets (x, y, label) in rgb domain not heatmap domain
                pred_rgb.append([pv[1][1] * 4, pv[1][0] * 4, pk])
        pred_rgb = np.asarray(pred_rgb, dtype=np.float32)  # (?, 3)
        pred_homo = None
        if pred_rgb.shape[0] >= 4:  # at least four points
            src_pts, dst_pts = class_mapping(pred_rgb)
            pred_homo, _ = cv2.findHomography(
                src_pts.reshape(-1, 1, 2), dst_pts.reshape(-1, 1, 2), cv2.RANSAC, 10)
            if pred_homo is not None:
                pred_part_mask = calc_iou_part(
                    pred_homo, image, field_model)

        if True:
            # if False:
            if pred_rgb.shape[0] >= 4 and pred_homo is not None:
                plt.imsave(osp.join(iou_visual_dir, 'test_' + str(frame_id) + '_pred_iou_part.png'), pred_part_mask)
                np.save(osp.join(homo_visual_dir, 'test_' + str(frame_id) + '_pred_iou_part.png'), pred_homo)

        frame_id += 1
        print(frame_id)


def calc_iou_part(pred_h, frame, template, frame_w=1280, frame_h=720, template_w=115,
                  template_h=74):
    # TODO: calculate iou part
    # === render ===
    render_w, render_h = template.size  # (1050, 680)
    dst = np.array(template)

    # Create three channels (680, 1050, 3)
    dst = np.stack((dst,) * 3, axis=-1)

    scaling_mat = np.eye(3)
    scaling_mat[0, 0] = render_w / template_w
    scaling_mat[1, 1] = render_h / template_h

    frame = np.uint8(frame * 255)  # 0-1 map to 0-255
    pred_mask_render = cv2.warpPerspective(
        frame, scaling_mat @ pred_h, (render_w, render_h), cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0))

    # === blending ===
    dstf = dst.astype(float) / 255
    pred_mask_renderf = pred_mask_render.astype(float) / 255
    pred_resultf = cv2.addWeighted(dstf, 0.3, pred_mask_renderf, 0.7, 0.0)
    pred_result = np.uint8(pred_resultf * 255)

    # field template binary mask
    field_mask = np.ones((frame_h, frame_w, 3), dtype=np.uint8) * 255
    pred_mask = cv2.warpPerspective(field_mask, pred_h, (template_w, template_h),
                                    cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
    pred_mask[pred_mask > 0] = 255

    pred_white_area = (pred_mask[:, :, 0] == 255) & (
            pred_mask[:, :, 1] == 255) & (pred_mask[:, :, 2] == 255)
    pred_fill = pred_mask.copy()
    pred_fill[pred_white_area, 0] = 0
    pred_fill[pred_white_area, 1] = 255
    pred_fill[pred_white_area, 2] = 0

    return pred_result


if __name__ == '__main__':
    test()
