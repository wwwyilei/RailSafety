from __future__ import absolute_import, division, print_function
import os
import torch
import cv2
from tqdm import tqdm
import supervision as sv
import numpy as np
from datetime import datetime
import json
from PIL import Image

# 导入必要的模型相关库
from segmentanything.segment_anything.predictor2 import SamPredictor
from DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2
from EdgeSAM.setup_edge_sam import build_edge_sam
from inference.models.yolo_world.yolo_world import YOLOWorld
from src.utils.interface import Detector

# 配置路径
config_paths = {
    "EdgeSAM_CHECKPOINT": os.path.join("D:\\GitHub\\Monovit", "weights", "edge_sam_3x.pth"),
    "DEPTH_MODEL": os.path.join("D:\\GitHub\\Monovit", "models"),
    "RAIL_MODEL": os.path.join("D:\\GitHub\\Monovit", "weights", "chromatic-laughter-5")
}

# 模型配置
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
}

# 设置设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 类别定义
CLASSES = ['person', 'car', 'rail-track']

def load_model():
    """加载所有必要的模型"""
    yolo_model = YOLOWorld(model_id="yolo_world/v2-l")
    edge_sam = build_edge_sam(checkpoint=config_paths["EdgeSAM_CHECKPOINT"]).to(DEVICE)
    sam_predictor = SamPredictor(edge_sam)
    
    # 初始化rail detector
    rail_detector = Detector(
        model_path=config_paths["RAIL_MODEL"],
        crop_coords="auto",
        runtime="pytorch",
        device=DEVICE
    )
    
    return yolo_model, sam_predictor, rail_detector

def create_rail_mask(image, rail_detector):
    """创建铁轨的二值mask"""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    h, w = image.shape[:2]
    
    # 获取铁轨路径预测
    for _ in range(50):
        rail_path = rail_detector.detect(pil_image) 
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if rail_path and len(rail_path) == 2:
        left_rail = np.array(rail_path[0])
        right_rail = np.array(rail_path[1])
        
        if len(left_rail) > 1 and len(right_rail) > 1:
            rail_polygon = np.vstack((left_rail, right_rail[::-1]))
            rail_polygon = rail_polygon.reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [rail_polygon], 1)
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
    
    return mask > 0

def mask_to_polygon(mask):
    """Convert binary mask to polygon format"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                 cv2.RETR_EXTERNAL, 
                                 cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        # 简化轮廓点
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) > 2:  # 只添加有效的多边形
            polygon = approx.flatten().tolist()
            polygons.append(polygon)
    return polygons

def create_coco_format():
    """创建COCO格式的基础结构"""
    return {
        "info": {
            "year": datetime.now().year,
            "version": "1.0",
            "description": "Pre-annotated dataset",
            "contributor": "Auto-annotator",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [{"id": 1, "name": "Unknown"}],
        "categories": [
            {"id": 1, "name": "person", "supercategory": "person"},
            {"id": 2, "name": "car", "supercategory": "vehicle"},
            {"id": 3, "name": "rail-track", "supercategory": "infrastructure"}
        ],
        "images": [],
        "annotations": []
    }

def visualize_annotations(image, annotations, image_id):
    """
    可视化单张图片的标注结果
    Args:
        image: 原始图片
        annotations: COCO格式的标注数据
        image_id: 当前图片的ID
    """
    # 创建图片副本
    viz_image = image.copy()
    
    # 设置不同类别的颜色
    colors = {
        1: (0, 255, 0),    # 人：绿色
        2: (0, 255, 255),  # 车：黄色
        3: (255, 0, 0)     # 铁轨：蓝色
    }
    
    # 绘制该图片的所有标注
    for ann in annotations["annotations"]:
        if ann["image_id"] == image_id:
            # 获取类别和颜色
            category_id = ann["category_id"]
            color = colors[category_id]
            
            # 如果是目标检测框（人或车）
            if category_id in [1, 2]:
                x, y, w, h = map(int, ann["bbox"])
                cv2.rectangle(viz_image, (x, y), (x + w, y + h), color, 2)
                
                # 添加类别和置信度标签
                conf = ann["attributes"]["confidence"]
                label = f"{CLASSES[category_id-1]} {conf:.2f}"
                cv2.putText(viz_image, label, (x, y - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 如果是铁轨分割
            elif category_id == 3:
                # 创建分割遮罩
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                for segment in ann["segmentation"]:
                    pts = np.array(segment).reshape((-1, 2)).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1)
                
                # 将分割结果叠加到图像上
                viz_image[mask == 1] = viz_image[mask == 1] * 0.7 + np.array(color) * 0.3
    
    return viz_image

def main():
    # 设置输入输出路径
    IMAGES_DIRECTORY = os.path.join("D:\\video",'validation')
    IMAGES_EXTENSIONS = ['jpg', 'jpeg', 'png']
    OUTPUT_DIRECTORY = os.path.join("D:\\GitHub\\Monovit", 'pre_validation')
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    # 加载模型
    print("Loading models...")
    yolo_model, sam_predictor, rail_detector = load_model()
    
    # 获取图片路径
    image_paths = sv.list_files_with_extensions(directory=IMAGES_DIRECTORY, 
                                              extensions=IMAGES_EXTENSIONS)
    
    # 初始化COCO格式数据
    coco_data = create_coco_format()
    annotation_id = 1
    
    # 处理每张图片
    print("Processing images...")
    for img_id, image_path in enumerate(tqdm(image_paths)):
        try:
            # 加载图片
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            
            # 添加图片信息
            height, width = image.shape[:2]
            image_info = {
                "id": img_id,
                "file_name": os.path.basename(image_path),
                "width": width,
                "height": height,
                "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            coco_data["images"].append(image_info)
            
            # 目标检测（人）
            yolo_model.set_classes(['person'])
            person_results = yolo_model.infer(image, confidence=0.1, iou=0.4)
            person_detections = sv.Detections.from_inference(person_results)
            
            # 目标检测（车）
            yolo_model.set_classes(['car'])
            car_results = yolo_model.infer(image, confidence=0.3, iou=0.4)
            car_detections = sv.Detections.from_inference(car_results)
            
            # 保存人的检测结果
            if len(person_detections) > 0:
                for xyxy, conf in zip(person_detections.xyxy, person_detections.confidence):
                    x1, y1, x2, y2 = map(float, xyxy)
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    
                    detection_annotation = {
                        "id": annotation_id,
                        "image_id": img_id,
                        "category_id": 1,  # person
                        "bbox": [x1, y1, width, height],
                        "area": float(area),
                        "iscrowd": 0,
                        "segmentation": [],  # 空的分割，但必须包含
                        "attributes": {
                            "confidence": float(conf),
                            "verified": False
                        }
                    }
                    coco_data["annotations"].append(detection_annotation)
                    annotation_id += 1
            
            # 保存车的检测结果
            if len(car_detections) > 0:
                for xyxy, conf in zip(car_detections.xyxy, car_detections.confidence):
                    x1, y1, x2, y2 = map(float, xyxy)
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    
                    detection_annotation = {
                        "id": annotation_id,
                        "image_id": img_id,
                        "category_id": 2,  # car
                        "bbox": [x1, y1, width, height],
                        "area": float(area),
                        "iscrowd": 0,
                        "segmentation": [],  # 空的分割，但必须包含
                        "attributes": {
                            "confidence": float(conf),
                            "verified": False
                        }
                    }
                    coco_data["annotations"].append(detection_annotation)
                    annotation_id += 1
            
            # 铁轨分割
            rail_mask = create_rail_mask(image, rail_detector)
            if rail_mask is not None:
                # 转换mask为polygon格式的分割
                rail_polygons = mask_to_polygon(rail_mask)
                
                if rail_polygons:  # 只在找到有效多边形时保存
                    # 计算边界框
                    contours = cv2.findContours(rail_mask.astype(np.uint8), cv2.RETR_EXTERNAL, 
                                             cv2.CHAIN_APPROX_SIMPLE)[0]
                    if len(contours) > 0:
                        x, y, w, h = cv2.boundingRect(contours[0])
                        
                        rail_annotation = {
                            "id": annotation_id,
                            "image_id": img_id,
                            "category_id": 3,  # rail-track
                            "bbox": [float(x), float(y), float(w), float(h)],
                            "area": float(np.sum(rail_mask)),
                            "iscrowd": 0,
                            "segmentation": rail_polygons,
                            "attributes": {
                                "verified": False
                            }
                        }
                        coco_data["annotations"].append(rail_annotation)
                        annotation_id += 1

            # 在处理完所有标注后，添加可视化
            visualization = visualize_annotations(image, coco_data, img_id)
            
            # 保存可视化结果
            viz_dir = os.path.join(OUTPUT_DIRECTORY, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            viz_path = os.path.join(viz_dir, f"{os.path.basename(image_path)}")
            cv2.imwrite(viz_path, visualization)
        
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue
    
    # 保存COCO格式的标注
    output_path = os.path.join(OUTPUT_DIRECTORY, "annotations.json")
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"Pre-annotation completed. Results saved to {output_path}")
    print(f"Total images processed: {len(image_paths)}")
    print(f"Total annotations created: {annotation_id - 1}")

if __name__ == "__main__":
    main()