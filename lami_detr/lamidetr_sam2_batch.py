import os
import sys
import cv2
import json
import torch
import numpy as np
from tqdm import tqdm

from PIL import Image
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from detectron2.config import LazyConfig, instantiate
from detectron2.layers import batched_nms
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import transforms as T
from detectron2.structures import Boxes, BoxMode, Instances

# 添加路径到 sys.path，以便导入 lami_detr 模块
# 脚本在 lami_detr 目录下，需要将父目录添加到路径，才能导入 lami_detr 模块
# 注意：将路径添加到末尾（append），而不是开头（insert(0, ...)），
# 这样可以确保已安装的包（如 detrex）优先被导入，避免本地目录覆盖已安装的包
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # LaMI-DETR 目录
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def plot_results(image, scores, boxes, labels, masks=None, thre=0):
    """在图像上绘制检测结果并返回"""
    fig = plt.figure(figsize=(16, 10))
    ax = plt.gca()
    colors = COLORS * 100
    
    # 绘制检测结果
    for score, (xmin, ymin, xmax, ymax), label, color in zip(scores, boxes, labels, colors):
        if score < thre:
            continue
        
        # 边界框和标签
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 fill=False, color=color, linewidth=3))
        ax.text(xmin, ymin, f'{label}: {score:0.2f}', 
                fontsize=15, bbox=dict(facecolor='white', alpha=0.8))

    plt.imshow(image)
    plt.axis('off')
    
    # 直接转换为图像数组
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    
    return img_array

def predict(image, names, visual_descs, score_thre=0.5, nms_score=0.5):
    """执行目标检测并返回结果"""
    # 如果 visual_descs 是空字典或 names 为空，直接返回原图
    if not visual_descs or not names:
        return image, ""
    
    # 图像预处理
    img_cv = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    img_tensor = preprocess_image(img_cv)
    
    # 模型推理
    with torch.no_grad():
        outputs = model(img_tensor.to(cfg.train.device), names, visual_descs, [image.size])[0]
    
    # 获取预测结果
    instances = outputs['instances']
    boxes = instances.pred_boxes.tensor.cpu()
    scores = instances.scores.cpu()
    labels = instances.pred_classes.cpu()
    label_names = [names[i] for i in labels]
    
    # NMS和阈值过滤
    keep_inds = batched_nms(boxes, scores, labels, nms_score)
    boxes, scores, labels = [x[keep_inds] for x in (boxes, scores, labels)]
    label_names = [label_names[i] for i in keep_inds]
    mask = scores > score_thre
    boxes, scores, labels = [x[mask] for x in (boxes, scores, labels)]
    label_names = [label_names[i] for i, flag in enumerate(mask) if flag]
    
    # 生成结果
    output_text = ",".join(set(label_names))
    if len(boxes) > 0:
        result_img = plot_results(img_cv, scores, boxes, label_names, thre=score_thre)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_img), output_text
    return image, output_text

def preprocess_image(image):
    """图像预处理"""
    b, g, r = cv2.split(image)
    img_eq = cv2.merge([cv2.equalizeHist(x) for x in [b, g, r]])
    img_aug, _ = T.apply_transform_gens(augmentation, img_eq)
    return torch.as_tensor(np.ascontiguousarray(img_aug.transpose(2, 0, 1))).unsqueeze(0)


class CustomDataset(Dataset):
    def __init__(self, image_dir, tag_dir):
        self.image_dir = image_dir
        self.tag_dir = tag_dir
        self.data = os.listdir(image_dir)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = self.data[idx]
        images = Image.open(os.path.join(self.image_dir, img_name))
        json_name = img_name.split('.')[0]+'.json'
        visual_descs = json.load(open(os.path.join(self.tag_dir, json_name)))
        names = list(visual_descs.keys())
        res = {
            'img_name': img_name,
            'images': images,
            'names': names,
            'visual_descs': visual_descs}
        return res

    def collate_fn(self, batch):
        return {
            key: [sample[key] for sample in batch]
            for key in ["img_name", "images", "names", "visual_descs"]
        }

if __name__ == '__main__':
    config_file = "configs/infer_dino_convnext_large.py"
    cfg = LazyConfig.load(config_file)
    augmentation = instantiate(cfg.dataloader.test.mapper.augmentation)
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model.eval()
    DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    dataset = CustomDataset('../examples/richhf/images', 
        '../examples/richhf/visual_descs')
    test_dataloader = DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=1, shuffle=False)
    output = '../examples/richhf/results'
    
    # 创建输出目录
    os.makedirs(output, exist_ok=True)
    
    for batch in tqdm(test_dataloader):
        # import pdb;pdb.set_trace()
        img_out, text_out = predict(batch['images'][0], batch['names'][0], batch['visual_descs'][0], 0.4, 0.7)
        # 生成安全的文件名，避免特殊字符
        img_name_base = batch['img_name'][0].split('.')[0]
        output_filename = f"{img_name_base}_result.png"
        img_out.save(os.path.join(output, output_filename))
        pass
