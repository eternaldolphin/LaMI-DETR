import os
import sys
import cv2
import json
import torch
import numpy as np
import argparse

from PIL import Image
import matplotlib.pyplot as plt

from detectron2.config import LazyConfig, instantiate
from detectron2.layers import batched_nms
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import transforms as T

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
    
    print(f"原始检测框数量: {len(boxes)}")
    if len(scores) > 0:
        print(f"最高置信度: {scores.max().item():.4f}, 平均置信度: {scores.mean().item():.4f}")
    
    # NMS和阈值过滤
    keep_inds = batched_nms(boxes, scores, labels, nms_score)
    boxes, scores, labels = [x[keep_inds] for x in (boxes, scores, labels)]
    label_names = [label_names[i] for i in keep_inds]
    print(f"NMS后检测框数量: {len(boxes)}")
    if len(scores) > 0:
        print(f"NMS后最高置信度: {scores.max().item():.4f}")
    
    mask = scores > score_thre
    boxes, scores, labels = [x[mask] for x in (boxes, scores, labels)]
    label_names = [label_names[i] for i, flag in enumerate(mask) if flag]
    print(f"阈值过滤后检测框数量: {len(boxes)} (使用阈值: {score_thre})")
    
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LaMI-DETR 单图推理脚本')
    parser.add_argument('--image', type=str, required=True, help='输入图片路径')
    parser.add_argument('--visual_desc', type=str, required=True, help='视觉描述JSON文件路径')
    parser.add_argument('--output', type=str, default=None, help='输出图片路径（可选，默认为输入图片名_result.png）')
    parser.add_argument('--config', type=str, default='configs/infer_dino_convnext_large.py', 
                        help='配置文件路径（默认: configs/infer_dino_convnext_large.py）')
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help='模型检查点路径（可选，默认使用配置文件中的路径）')
    parser.add_argument('--score_thre', type=float, default=0.4, 
                        help='置信度阈值（默认: 0.4）')
    parser.add_argument('--nms_score', type=float, default=0.7, 
                        help='NMS阈值（默认: 0.7）')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"图片文件不存在: {args.image}")
    if not os.path.exists(args.visual_desc):
        raise FileNotFoundError(f"视觉描述文件不存在: {args.visual_desc}")
    
    # 加载模型和配置
    config_file = args.config
    cfg = LazyConfig.load(config_file)
    augmentation = instantiate(cfg.dataloader.test.mapper.augmentation)
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model.eval()
    checkpoint = args.checkpoint if args.checkpoint else cfg.train.init_checkpoint
    DetectionCheckpointer(model).load(checkpoint)
    
    # 加载图片
    image = Image.open(args.image).convert('RGB')
    
    # 加载视觉描述
    with open(args.visual_desc, 'r') as f:
        visual_descs = json.load(f)
    names = list(visual_descs.keys())
    
    # 执行推理
    img_out, text_out = predict(image, names, visual_descs, args.score_thre, args.nms_score)
    
    # 确定输出路径
    if args.output is None:
        img_name_base = os.path.splitext(os.path.basename(args.image))[0]
        output_dir = os.path.dirname(args.image)
        output_path = os.path.join(output_dir, f"{img_name_base}_result.png")
    else:
        output_path = args.output
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    
    # 保存结果
    img_out.save(output_path)
    print(f"检测到的类别: {text_out}")
    print(f"结果已保存到: {output_path}")
