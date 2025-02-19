import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
import supervision as sv
from PIL import Image
import json
import matplotlib as mpl
import matplotlib.cm as cm
from typing import Dict, Any, Tuple, List
import time

# Import project modules
from inference.models.yolo_world.yolo_world import YOLOWorld
from EdgeSAM.setup_edge_sam import build_edge_sam
from segmentanything.segment_anything.predictor2 import SamPredictor
from src.utils.interface import Detector
from DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2

# Define classes
CLASSES = ['person', 'car', 'rail-track']

# Load calibration parameters
def load_calibration_params(filename='calibration_params.json'):
    """Load calibration parameters from json file"""
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                params = json.load(f)
            return params
        except Exception as e:
            print(f"Failed to load calibration parameters: {str(e)}")
            return None
    print("Calibration file not found. Using uncalibrated depth.")
    return None



def process_depth_map(depth, calibration_params=None):
    """Process depth map using calibration parameters"""
    if calibration_params is None:
        return depth
    

    depth = depth.copy()
    

    try:
        coefficients = calibration_params['calibration_function']['coefficients']
        a, b, c = coefficients
        

        calibrated_depth = a * np.exp(b * depth) + c  # use same exp function
        calibrated_depth = calibrated_depth
        

        depth_min = calibration_params['depth_range']['min']
        depth_max = calibration_params['depth_range']['max']
        calibrated_depth = np.clip(calibrated_depth, depth_min, depth_max)
        
        return calibrated_depth
        
    except Exception as e:
        print(f"Error applying calibration: {str(e)}")
        return depth


def calibration_function(x, a, b, c):
    return a * np.exp(b * x) + c

def apply_calibration(depth_map, calibration_params):
    return calibration_function(depth_map, *calibration_params)

def segment(sam_predictor: SamPredictor, rail_detector: Detector, image: np.ndarray, detections: sv.Detections) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    
    # get rail track mask
    rail_mask = create_rail_mask(image, rail_detector)
    result_masks.append(rail_mask)
    
    # process person and car masks using SAM
    for box in detections.xyxy:
        masks, scores, _ = sam_predictor.predict(box=box, multimask_output=True)
        index = np.argmax(scores)
        result_masks.append(masks[index])
    
    return np.array(result_masks)


def find_rail_point(rail_mask, projected_point):
    """Find corresponding point on rail mask closest to the projected point"""
    rail_points = np.argwhere(rail_mask)
    y_points = rail_points[np.abs(rail_points[:, 0] - projected_point[0]) < 5] # y position
    
    if len(y_points) == 0:
        return None
    
    distances = np.abs(y_points[:, 1] - projected_point[1])  # x position
    closest_idx = np.argmin(distances)
    
    return tuple(y_points[closest_idx])


def find_critical_points(person_mask, rail_mask):
    """Find key points for distance measurement"""
    image_width = person_mask.shape[1]
    midline_x = image_width // 2
    
    person_edge_points = get_mask_edges(person_mask)
    if len(person_edge_points) == 0:
        return None, None, None
    
    mask_points = np.argwhere(person_mask)
    bottom_y = np.max(mask_points[:, 0])
    bottom_points = mask_points[mask_points[:, 0] == bottom_y]
    bottom_x = int(np.mean(bottom_points[:, 1]))  # now use mean value, almost same with middle
    bottom_center = (bottom_y, bottom_x)
    
    distances_to_midline = np.abs(person_edge_points[:, 1] - midline_x)
    nearest_idx = np.argmin(distances_to_midline)
    nearest_point = person_edge_points[nearest_idx]
    
    projected_point = (bottom_y, nearest_point[1]) # use the x-coordinate of the nearest point and the y-coordinate of the bottom
    
    return nearest_point, projected_point, bottom_center


def get_mask_edges(mask):
    """Get mask edge points"""
    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel)  # remove 1 pixel
    edge = mask.astype(np.uint8) - eroded # get the pixel position
    return np.argwhere(edge)

def find_matching_depth_on_rail(rail_mask, depth_map, target_depth, person_foot_position, tolerance=18):
    """ Find the point on the railroad track that matches the depth of the target """
    rail_points = np.argwhere(rail_mask)
    rail_depths = depth_map[rail_points[:, 0], rail_points[:, 1]]
    
    absolute_diff = np.abs(rail_depths - target_depth)
    matching_indices = np.where(absolute_diff < tolerance)[0]
    
    if len(matching_indices) > 0:
        matching_points = rail_points[matching_indices]
        matching_depths = rail_depths[matching_indices]
        
        distances = np.sqrt(np.sum((matching_points - person_foot_position) ** 2, axis=1))
        nearest_index = np.argmin(distances)
        
        return (matching_points[nearest_index], 
                matching_depths[nearest_index],
                absolute_diff[matching_indices[nearest_index]])
    
    return None, None, None


def create_rail_mask(image, rail_detector):
    """Create a binary mask for the rail track"""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    h, w = image.shape[:2]
    
    crop_coords = rail_detector.get_crop_coords()
    rail_path = rail_detector.detect(pil_image)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if rail_path and len(rail_path) == 2:
        left_rail = np.array(rail_path[0])
        right_rail = np.array(rail_path[1])
        
        if len(left_rail) > 1 and len(right_rail) > 1: # at least 2 point
            rail_polygon = np.vstack((left_rail, right_rail[::-1])) 
            rail_polygon = rail_polygon.reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [rail_polygon], 1)
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
    
    return mask > 0


def calculate_horizontal_distance(rail_mask, person_mask, depth_map, gauge_width=1.435):
    """
    To calculate the lateral distance:
    1. find the closest point and its bottom projection point -->find_critical_points
    2. find the matching point on the railroad track using the depth of the center of the foot. -->find_matching_depth_on_rail
    3. Calculate the distance from the projected point to the matching point.
    """
    nearest_point, projected_point, foot_center = find_critical_points(person_mask, rail_mask)
    
    if nearest_point is None or projected_point is None or foot_center is None:
        return None, None, None
    
    foot_depth = depth_map[foot_center[0], foot_center[1]]
    matching_point, matching_depth, _ = find_matching_depth_on_rail(
        rail_mask, 
        depth_map, 
        foot_depth,
        foot_center
    )
    
    if matching_point is None:
        return None, None, None
    
    rail_y = matching_point[0] #y coordinate
    rail_points = np.where(rail_mask[rail_y, :])[0]  # leaf and right position in railway
    
    if len(rail_points) < 2:
        return None, None, None
    
    track_width_pixels = rail_points[-1] - rail_points[0]  # pixel width
    scale = gauge_width / track_width_pixels 
    
    pixel_distance = abs(projected_point[1] - matching_point[1])
    meter_distance = pixel_distance * scale
    
    return meter_distance, projected_point, matching_point

def calculate_horizontal_distance_without_depth(rail_mask, person_mask, gauge_width=1.435):

    # Find the key points
    nearest_point, projected_point, foot_center = find_critical_points(person_mask, rail_mask)
    
    if nearest_point is None or projected_point is None or foot_center is None:
        return None, None, None
        
    # Find the nearest point on rail at the same height
    rail_points = np.argwhere(rail_mask)
    # Only consider points at the same y-coordinate (height)
    same_height_points = rail_points[rail_points[:, 0] == projected_point[0]]
    
    if len(same_height_points) == 0:
        return None, None, None
        
    # Calculate distances to all points at the same height
    distances = np.abs(same_height_points[:, 1] - projected_point[1])
    nearest_idx = np.argmin(distances)
    matching_point = tuple(same_height_points[nearest_idx])
    
    # Calculate scale using rail gauge width
    rail_y = matching_point[0]
    rail_points_at_height = np.where(rail_mask[rail_y, :])[0]
    
    if len(rail_points_at_height) < 2:
        return None, None, None
    
    track_width_pixels = rail_points_at_height[-1] - rail_points_at_height[0]
    scale = gauge_width / track_width_pixels
    
    # Calculate final distance
    pixel_distance = abs(projected_point[1] - matching_point[1])
    meter_distance = pixel_distance * scale
    
    return meter_distance, projected_point, matching_point

class MultiReferenceDistanceCalculator:
    def __init__(self, calibration_data: Dict[str, Any]):
        self.calibration = calibration_data
        self.generate_symmetric_references()

    def calculate_symmetric_point(self, point: Tuple[float, float], 
                                left_rail: List[Tuple[float, float]], 
                                right_rail: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Calculate the points of symmetry of the points about the centerline of the railroad track."""
        # Calculate the two points on the centerline of the railroad track #
        center_line_start = (
            (left_rail[0][0] + right_rail[0][0]) / 2,
            (left_rail[0][1] + right_rail[0][1]) / 2
        )
        center_line_end = (
            (left_rail[1][0] + right_rail[1][0]) / 2,
            (left_rail[1][1] + right_rail[1][1]) / 2
        )
        
        # Calculate the direction vector of the centerline
        direction = (
            center_line_end[0] - center_line_start[0],
            center_line_end[1] - center_line_start[1]
        )
        
        # Normalized Direction Vector
        length = np.sqrt(direction[0]**2 + direction[1]**2)
        direction = (direction[0]/length, direction[1]/length)
        
        # Calculate the perpendicular vector from the point to the centerline
        point_to_center = (
            point[0] - center_line_start[0],
            point[1] - center_line_start[1]
        )
        
        # Calculate the projection of the point on the centerline
        dot_product = (point_to_center[0] * direction[0] + 
                    point_to_center[1] * direction[1])
        projection = (
            center_line_start[0] + dot_product * direction[0],
            center_line_start[1] + dot_product * direction[1]
        )
        
        # Calculate the point of symmetry
        symmetric_point = (
            2 * projection[0] - point[0],
            2 * projection[1] - point[1]
        )
        
        return symmetric_point

    def generate_symmetric_references(self):
        """Generate a other-side symmetric reference line"""
        if not self.calibration.get('rail_lines'):
            return
            
        left_rail = self.calibration['rail_lines'].get('left')
        right_rail = self.calibration['rail_lines'].get('right')
        
        if not (left_rail and right_rail):
            return
            

        left_references = [ref for ref in self.calibration['reference_lines']]
        

        for ref in left_references:

            symmetric_ref = {
                'distance': ref['distance'], 
                'line1': [
                    self.calculate_symmetric_point(ref['line1'][0], left_rail, right_rail),
                    self.calculate_symmetric_point(ref['line1'][1], left_rail, right_rail)
                ],
                'line2': [
                    self.calculate_symmetric_point(ref['line2'][0], left_rail, right_rail),
                    self.calculate_symmetric_point(ref['line2'][1], left_rail, right_rail)
                ]
            }
            

            self.calibration['reference_lines'].append(symmetric_ref)

    def point_in_region(self, point, ref_line) -> bool:
        """Determine if the point is within the area defined by the reference line"""
        x, y = point
        region = np.array([
            ref_line['line1'][0],
            ref_line['line1'][1],
            ref_line['line2'][1],
            ref_line['line2'][0]
        ], dtype=np.int32)
        result = cv2.pointPolygonTest(region.reshape((-1,1,2)), 
                                    (float(x), float(y)), False)
        return result >= 0

    def line_intersection(self, p1, p2, p3, p4) -> Tuple[float, float]:
        """Calculate the intersection of two line segments"""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denominator) < 1e-10:
            return None
            
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (x, y)
        return None

    def get_line_region_intersection(self, point1, point2, ref_line) -> List[Tuple[float, float]]:
        """Get the intersection of the line segment with the region"""

        edges = [
            (ref_line['line1'][0], ref_line['line1'][1]),  # left
            (ref_line['line1'][1], ref_line['line2'][1]),  # up
            (ref_line['line2'][1], ref_line['line2'][0]),  # right
            (ref_line['line2'][0], ref_line['line1'][0])   # bottom
        ]
        

        intersections = []
        for edge_start, edge_end in edges:
            intersection = self.line_intersection(point1, point2, edge_start, edge_end)
            if intersection is not None:
                intersections.append(intersection)
        

        if intersections:
            intersections.sort(key=lambda p: 
                ((p[0]-point1[0])**2 + (p[1]-point1[1])**2)**0.5)
        return intersections
    
    def find_segments_in_regions(self, point1, point2) -> List[Tuple[Tuple[float, float], 
                                                                    Tuple[float, float], 
                                                                    Dict]]:
        """Segmentation of line segments into parts in different regions"""
        segments = []
        for ref_line in self.calibration['reference_lines']:
            p1_in = self.point_in_region(point1, ref_line)
            p2_in = self.point_in_region(point2, ref_line)
            
            if p1_in and p2_in:
                segments.append((point1, point2, ref_line))
            elif p1_in or p2_in:
                intersections = self.get_line_region_intersection(point1, point2, ref_line)
                if intersections:
                    if p1_in:
                        segments.append((point1, intersections[0], ref_line))
                    else:
                        segments.append((intersections[0], point2, ref_line))
            else:
                intersections = self.get_line_region_intersection(point1, point2, ref_line)
                if len(intersections) >= 2:
                    segments.append((intersections[0], intersections[1], ref_line))
        
        segments.sort(key=lambda s: 
            ((s[0][0]-point1[0])**2 + (s[0][1]-point1[1])**2)**0.5)
        return segments

    def compute_local_transform(self, ref_line):
        """Calculate the local perspective transformation matrix"""
        src_points = np.array([
            ref_line['line1'][0],
            ref_line['line1'][1],
            ref_line['line2'][0],
            ref_line['line2'][1]
        ], dtype=np.float32)
        
        dist = ref_line['distance']
        line1_height = np.linalg.norm(
            np.array(ref_line['line1'][1]) - np.array(ref_line['line1'][0])
        )
        
        dst_points = np.array([
            [0, 0],
            [0, line1_height * dist / line1_height],
            [dist, 0],
            [dist, line1_height * dist / line1_height]
        ], dtype=np.float32)
        
        return cv2.getPerspectiveTransform(src_points, dst_points)

    def transform_point(self, point, H):
        """Transform point coordinates using the transformation matrix"""
        p = np.array([[point[0], point[1], 1]], dtype=np.float32).T
        p_transformed = np.dot(H, p)
        p_transformed = p_transformed / p_transformed[2]
        return p_transformed[:2].flatten()


    def calculate_distance(self, point1, point2) -> Tuple[float, List[Dict]]:
        """Calculate the actual distance between two points, considering all areas through which the line passes"""

        segments = self.find_segments_in_regions(point1, point2)
        
        # Find the 2.0m reference line as the default reference for the external area
        outer_ref = next(ref for ref in self.calibration['reference_lines'] 
                        if abs(ref['distance'] - 2.0) < 0.01)
        
        if not segments:
            # If no segments are found, it means it is completely outside the area, use the 2.0m reference line
            H = self.compute_local_transform(outer_ref)
            p1_world = self.transform_point(point1, H)
            p2_world = self.transform_point(point2, H)
            return np.linalg.norm(p2_world - p1_world), [outer_ref]

        # Calculate the distance of each segment
        total_distance = 0
        used_refs = []
        last_point = point1
        
        

        for i, (seg_start, seg_end, ref_line) in enumerate(segments):
            # over use 2.0m reference
            if i == 0 and seg_start != point1:
                H = self.compute_local_transform(outer_ref)
                p1_world = self.transform_point(point1, H)
                p2_world = self.transform_point(seg_start, H)
                total_distance += np.linalg.norm(p2_world - p1_world)
                used_refs.append(outer_ref)
                
            # calculation
            H = self.compute_local_transform(ref_line)
            p1_world = self.transform_point(seg_start, H)
            p2_world = self.transform_point(seg_end, H)
            total_distance += np.linalg.norm(p2_world - p1_world)
            if ref_line not in used_refs:
                used_refs.append(ref_line)
                
            last_point = seg_end
            

        if last_point != point2:
            H = self.compute_local_transform(outer_ref)
            p1_world = self.transform_point(last_point, H)
            p2_world = self.transform_point(point2, H)
            total_distance += np.linalg.norm(p2_world - p1_world)
            if outer_ref not in used_refs:
                used_refs.append(outer_ref)

        return total_distance, used_refs    
    
    
class DetectionEvaluator:
    def __init__(self, gt_path, val_path, output_path):
        """Initialize evaluator with paths"""
        self.coco = COCO(gt_path)
        self.val_path = val_path
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        
        self.detection_metrics = {
            'person': {'tp': 0, 'fp': 0, 'fn': 0},
            'car': {'tp': 0, 'fp': 0, 'fn': 0}
        }
        self.rail_metrics = []
        self.evaluated_images = set()  # Keep track of evaluated images
        self.results = {
            'detections': {},
            'rail_segmentation': {}
        }
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        if len(box1) == 4:
            box1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
            
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return intersection / (box1_area + box2_area - intersection)

    def calculate_mask_iou(self, mask1, mask2):
        """Calculate IoU between two masks"""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0

    def evaluate_detections(self, detections, image_id):
        """Evaluate object detections against ground truth"""
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        
        gt_classes = {'person': [], 'car': []}
        for ann in anns:
            if ann['category_id'] == 1:
                gt_classes['person'].append(ann['bbox'])
            elif ann['category_id'] == 2:
                gt_classes['car'].append(ann['bbox'])
        
        matched_gt = {'person': set(), 'car': set()}
        detection_results = {
            'person': {'matches': [], 'scores': []},
            'car': {'matches': [], 'scores': []}
        }
        
        for det_box, det_cls, score in zip(detections.xyxy, detections.class_id, detections.confidence):
            cls_name = 'person' if det_cls == 0 else 'car'
            best_iou = 0.5
            best_gt_idx = -1
            
            for i, gt_box in enumerate(gt_classes[cls_name]):
                if i not in matched_gt[cls_name]:
                    iou = self.calculate_iou(gt_box, det_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
            
            if best_gt_idx >= 0:
                self.detection_metrics[cls_name]['tp'] += 1
                matched_gt[cls_name].add(best_gt_idx)
                detection_results[cls_name]['matches'].append(1)
            else:
                self.detection_metrics[cls_name]['fp'] += 1
                detection_results[cls_name]['matches'].append(0)
            
            detection_results[cls_name]['scores'].append(score)
        
        for cls_name in ['person', 'car']:
            self.detection_metrics[cls_name]['fn'] += len(gt_classes[cls_name]) - len(matched_gt[cls_name])
        
        self.results['detections'][image_id] = detection_results

    # def evaluate_rail_segmentation(self, pred_mask, image_id):
    #     """Evaluate rail segmentation against ground truth"""
    #     ann_ids = self.coco.getAnnIds(imgIds=image_id)
    #     anns = self.coco.loadAnns(ann_ids)
        
    #     gt_mask = None
    #     for ann in anns:
    #         if ann['category_id'] == 3:
    #             gt_mask = self.coco.annToMask(ann)
    #             break
        
    #     if gt_mask is None:
    #         return
            
    #     iou = self.calculate_mask_iou(pred_mask, gt_mask)
    #     self.rail_metrics.append(iou)
    #     self.results['rail_segmentation'][image_id] = iou
    def evaluate_rail_segmentation(self, pred_mask, image_id):
        """Evaluate rail segmentation against ground truth"""
        # Skip if this image has already been evaluated
        if image_id in self.evaluated_images:
            return
            
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        
        gt_mask = None
        for ann in anns:
            if ann['category_id'] == 3:  # Rail track category
                gt_mask = self.coco.annToMask(ann)
                break
        
        if gt_mask is None:
            return
            
        iou = self.calculate_mask_iou(pred_mask, gt_mask)
        self.rail_metrics.append(iou)
        self.results['rail_segmentation'][image_id] = iou
        self.evaluated_images.add(image_id)  # Mark this image as evaluated

            
    def print_metrics(self):
        """Print evaluation metrics"""
        print("\nDetection Metrics:")
        for cls_name, metrics in self.detection_metrics.items():
            precision = metrics['tp'] / (metrics['tp'] + metrics['fp']) if metrics['tp'] + metrics['fp'] > 0 else 0
            recall = metrics['tp'] / (metrics['tp'] + metrics['fn']) if metrics['tp'] + metrics['fn'] > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
            
            print(f"\n{cls_name.capitalize()}:")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"True Positives: {metrics['tp']}")
            print(f"False Positives: {metrics['fp']}")
            print(f"False Negatives: {metrics['fn']}")
        
        if self.rail_metrics:
            mean_iou = np.mean(self.rail_metrics)
            print(f"\nRail Segmentation:")
            print(f"Mean IoU: {mean_iou:.4f}")
            # print(f"Number of evaluated images: {len(self.evaluated_images)}")  # Use set length
            print(f"Number of evaluated images: {len(self.rail_metrics)}")

class EnhancedDetectionEvaluator(DetectionEvaluator):
    def __init__(self, gt_path, val_path, output_path, calibration_data):
        super().__init__(gt_path, val_path, output_path)

        self.reference_calculator = MultiReferenceDistanceCalculator(calibration_data)  

        # self.distance_metrics = {
        #     'errors': [],
        #     'reference_distances': [],
        #     'calculated_distances': []
        # }
        self.distance_metrics = {
            'valid_measurements': []  
        }

    def evaluate_horizontal_distance(self, person_mask, rail_mask, depth_map, image_name):
        """Evaluating lateral distances using new distance calculation methods"""
        try:
            # depth estimation method
            # calculated_distance, projected_point, matching_point = calculate_horizontal_distance(
            #     rail_mask, person_mask, depth_map
            # )
            
            # 无深度引导版本的结果
            calculated_distance, projected_point, matching_point = calculate_horizontal_distance_without_depth(
                rail_mask, person_mask
            )
            
            if calculated_distance is not None and projected_point is not None and matching_point is not None:
                # Calculate distances using the new reference line method
                point1 = (projected_point[1], projected_point[0])  # 转换坐标格式
                point2 = (matching_point[1], matching_point[0])
                ref_distance, used_refs = self.reference_calculator.calculate_distance(point1, point2)
                
                # Only record measurements with a reference distance of less than 4 m
                if ref_distance < 4.0:
                    self.distance_metrics['valid_measurements'].append({
                        'reference_distance': ref_distance,
                        'calculated_distance': calculated_distance,
                        'error': abs(ref_distance - calculated_distance),
                        'image_name': image_name,
                        'projected_point': projected_point,
                        'matching_point': matching_point,
                        'used_refs': used_refs
                    })
                    
                return ref_distance, calculated_distance, projected_point, matching_point
                
        except Exception as e:
            print(f"Error evaluating distance for {image_name}: {str(e)}")
            return None, None, None, None

    def print_distance_metrics(self):
        """Only measurements with a reference distance of less than 4 m"""
        if not self.distance_metrics['valid_measurements']:
            print("\nNo valid distance measurements (reference distance < 4m)")
            return
            
        valid_measurements = self.distance_metrics['valid_measurements']
        
        # Extract the error and reference distance of all valid measurements
        errors = [m['error'] for m in valid_measurements]
        reference_distances = [m['reference_distance'] for m in valid_measurements]
        
        print("\nDistance Measurement Metrics (for reference distances < 4m):")
        print(f"Mean Error: {np.mean(errors):.3f}m")
        print(f"Standard Deviation: {np.std(errors):.3f}m")
        print(f"Maximum Error: {np.max(errors):.3f}m")
        print(f"Total valid measurements: {len(errors)}")
        
        # Error distribution statistics
        error_ranges = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, float('inf'))]
        print("\nError Distribution:")
        for start, end in error_ranges:
            count = sum(1 for e in errors if start <= e < end)
            percentage = (count / len(errors)) * 100
            print(f"{start}-{end}m: {count} measurements ({percentage:.1f}%)")
        
        # Reference Distance Distribution Statistics
        distance_ranges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        print("\nReference Distance Distribution:")
        for start, end in distance_ranges:
            count = sum(1 for d in reference_distances if start <= d < end)
            percentage = (count / len(reference_distances)) * 100
            print(f"{start}-{end}m: {count} measurements ({percentage:.1f}%)")

        # Add error analysis
        large_errors = [(m['image_name'], m['error'], m['reference_distance']) 
                       for m in valid_measurements if m['error'] > 0.5]
        if large_errors:
            print("\nLarge Errors Analysis (errors > 0.5m):")
            for img_name, error, ref_dist in sorted(large_errors, key=lambda x: x[1], reverse=True)[:10]:
                print(f"Image: {img_name}, Error: {error:.3f}m, Reference Distance: {ref_dist:.3f}m")

 

    def evaluate_image(self, image, detections, rail_mask, depth_map, image_id, img_name):
        """Evaluate all metrics for a single image and store results"""
        try:
            # Basic evaluation
            self.evaluate_detections(detections, image_id)
            if rail_mask is not None:
                self.evaluate_rail_segmentation(rail_mask, image_id)
            
            # Calculate distances for all detected objects
            distance_results = []
            if hasattr(detections, 'mask') and detections.mask is not None:
                for i, (xyxy, conf, cls) in enumerate(zip(detections.xyxy, 
                                                        detections.confidence, 
                                                        detections.class_id)):
                    if CLASSES[cls] in ['person', 'car']:
                        if i < len(detections.mask):
                            object_mask = detections.mask[i]
                            calc_distance, projected_point, matching_point = calculate_horizontal_distance(
                                rail_mask, object_mask, depth_map
                            )
                            
                            if all(x is not None for x in [calc_distance, projected_point, matching_point]):
                                distance_results.append({
                                    'bbox': xyxy,
                                    'class_id': cls,
                                    'confidence': conf,
                                    'calc_distance': calc_distance,
                                    'projected_point': projected_point,
                                    'matching_point': matching_point
                                })
            
            # Visualize results
            self.save_visualization(
                image=image,
                detections=detections,
                rail_mask=rail_mask,
                image_id=image_id,
                img_name=img_name,
                distance_results=distance_results
            )
            
            print(f"\nImage: {img_name}")
            print(f"Total detections: {len(detections.xyxy)}")
            print(f"Successful distance measurements: {len(distance_results)}")
            
            return distance_results
        
        except Exception as e:
            print(f"Error evaluating image {img_name}: {str(e)}")
            return []


    def save_visualization(self, image, detections, rail_mask, image_id, img_name, distance_results):
        """Save visualization with distance measurements"""
        try:
            gt_image = image.copy()
            pred_image = image.copy()
            h, w = image.shape[:2]

            # Draw ground truth rail
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                if ann['category_id'] == 3 and 'segmentation' in ann:
                    points = np.array(ann['segmentation'][0]).reshape(-1, 2).astype(np.int32)
                    gt_rail = np.zeros_like(gt_image)
                    cv2.fillPoly(gt_rail, [points], (0, 255, 0))
                    gt_image = cv2.addWeighted(gt_image, 1.0, gt_rail, 0.3, 0)

            # Draw predicted rail
            if rail_mask is not None:
                rail_vis = np.zeros_like(pred_image)
                rail_vis[rail_mask > 0] = [255, 0, 0]
                pred_image = cv2.addWeighted(pred_image, 1.0, rail_vis, 0.3, 0)


            # Add segmentation results for each detected object
            if hasattr(detections, 'mask') and detections.mask is not None:
                for i, (mask, class_id) in enumerate(zip(detections.mask[1:], detections.class_id[1:])):
                    instance_vis = np.zeros_like(pred_image)
                    if CLASSES[class_id] == 'person':
                        color = [0, 0, 255]  # people for red
                    elif CLASSES[class_id] == 'car':
                        color = [255, 0, 0]  # car for blue
                    instance_vis[mask > 0] = color
                    pred_image = cv2.addWeighted(pred_image, 1.0, instance_vis, 0.3, 0)

            # if hasattr(detections, 'mask') and detections.mask is not None:
            #     for i, (mask, class_id) in enumerate(zip(detections.mask[1:], detections.class_id[1:])):
            #         instance_vis = np.zeros_like(pred_image)
            #         # 根据类别使用不同颜色
            #         color = [0, 255, 0] if CLASSES[class_id] == 'person' else [255, 0, 0]
            #         instance_vis[mask > 0] = color
            #         pred_image = cv2.addWeighted(pred_image, 1.0, instance_vis, 0.3, 0)
                    
            # Draw ground truth boxes and use predicted points for distance
            gt_index = 0
            for ann in anns:
                if ann['category_id'] in [1, 2]:
                    bbox = [int(x) for x in ann['bbox']]
                    class_name = 'person' if ann['category_id'] == 1 else 'car'
                    
                    # Draw ground truth box
                    cv2.rectangle(gt_image,
                                (bbox[0], bbox[1]),
                                (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                (0, 255, 0), 2)
                    cv2.putText(gt_image,
                            class_name,
                            (bbox[0], bbox[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)
                    
                    # Use predicted points to calculate reference distance
                    if gt_index < len(distance_results):
                        result = distance_results[gt_index]
                        proj_point = result['projected_point']
                        match_point = result['matching_point']
                        
                        # Calculate reference distance using the same points
                        ref_distance, _ = self.reference_calculator.calculate_distance(
                            (proj_point[1], proj_point[0]),
                            (match_point[1], match_point[0])
                        )
                        
                        if ref_distance < 4.0:  # Only measurements with a reference distance of less than 4m are displayed
                            # Draw measurement on ground truth image
                            cv2.circle(gt_image, (proj_point[1], proj_point[0]), 4, (255, 0, 255), -1)
                            cv2.circle(gt_image, (match_point[1], match_point[0]), 4, (0, 255, 0), -1)
                            cv2.line(gt_image,
                                (proj_point[1], proj_point[0]),
                                (match_point[1], match_point[0]),
                                (255, 255, 255), 2)
                            cv2.putText(gt_image,
                                    f"Ref: {ref_distance:.2f}m",
                                    ((proj_point[1] + match_point[1])//2,
                                    proj_point[0] + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (255, 255, 255), 2)
                            
                            # Record measurement in new data structure
                            self.distance_metrics['valid_measurements'].append({
                                'reference_distance': ref_distance,
                                'calculated_distance': result['calc_distance'],
                                'error': abs(ref_distance - result['calc_distance']),
                                'image_name': img_name,
                                'projected_point': proj_point,
                                'matching_point': match_point
                            })
                            
                    gt_index += 1

            for result in distance_results:
                # Draw detection box
                x1, y1, x2, y2 = map(int, result['bbox'])
                cv2.rectangle(pred_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Draw label
                label = f"{CLASSES[result['class_id']]} {result['confidence']:.2f}"
                cv2.putText(pred_image,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)
                
                # Draw horizontal distance measurement
                proj_point = result['projected_point']
                match_point = result['matching_point']
                calc_dist = result['calc_distance']
                
                if proj_point and match_point:
                    # Draw points
                    cv2.circle(pred_image, (proj_point[1], proj_point[0]), 4, (255, 0, 255), -1)
                    cv2.circle(pred_image, (match_point[1], match_point[0]), 4, (0, 255, 0), -1)
                    
                    # Draw horizontal line (ensure same y-coordinate)
                    y_coord = proj_point[0]  # Use the same y-coordinate for both points
                    cv2.line(pred_image,
                        (proj_point[1], y_coord),
                        (match_point[1], y_coord),
                        (255, 255, 255), 2)
                    
                    # Add distance label
                    text_pos_x = (proj_point[1] + match_point[1]) // 2
                    cv2.putText(pred_image,
                            f"Calc: {calc_dist:.2f}m",
                            (text_pos_x, y_coord - 10),  # Place text above the line
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2)
                
            # # Draw predictions and calculated distances
            # for result in distance_results:
            #     # Draw detection box
            #     x1, y1, x2, y2 = map(int, result['bbox'])
            #     cv2.rectangle(pred_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
            #     # Draw label
            #     label = f"{CLASSES[result['class_id']]} {result['confidence']:.2f}"
            #     cv2.putText(pred_image,
            #             label,
            #             (x1, y1 - 5),
            #             cv2.FONT_HERSHEY_SIMPLEX,
            #             0.5, (0, 0, 255), 2)
                
            #     # Draw distance measurement
            #     proj_point = result['projected_point']
            #     match_point = result['matching_point']
            #     calc_dist = result['calc_distance']
                
            #     # 计算参考距离
            #     ref_distance, _ = self.reference_calculator.calculate_distance(
            #         (proj_point[1], proj_point[0]),
            #         (match_point[1], match_point[0])
            #     )
                
            #     if ref_distance < 4.0:  # 只显示参考距离小于4m的测量
            #         cv2.circle(pred_image, (proj_point[1], proj_point[0]), 4, (255, 0, 255), -1)
            #         cv2.circle(pred_image, (match_point[1], match_point[0]), 4, (0, 255, 0), -1)
            #         cv2.line(pred_image,
            #             (proj_point[1], proj_point[0]),
            #             (match_point[1], match_point[0]),
            #             (255, 255, 255), 2)
            #         cv2.putText(pred_image,
            #                 f"Calc: {calc_dist:.2f}m",
            #                 ((proj_point[1] + match_point[1])//2,
            #                 proj_point[0] + 25),
            #                 cv2.FONT_HERSHEY_SIMPLEX,
            #                 0.8, (255, 255, 255), 2)



            # save

            header_height = 30
            header = np.full((header_height, w * 2, 3), 255, dtype=np.uint8)
            cv2.putText(header, "Ground Truth (Reference)", (w//6, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(header, "Predictions (Calculated)", (w + w//6, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            combined = np.hstack([gt_image, pred_image])
            final_image = np.vstack([header, combined])
            
            base_output_path = os.path.splitext(img_name)[0]

            combined_path = os.path.join(self.output_path, f"{base_output_path}_combined.jpg")
            cv2.imwrite(combined_path, final_image)
            
            # 2. save ground truth
            gt_path = os.path.join(self.output_path, f"{base_output_path}_gt.jpg")
            cv2.imwrite(gt_path, gt_image)
            
            # 3. save predicted
            pred_path = os.path.join(self.output_path, f"{base_output_path}_pred.jpg")
            cv2.imwrite(pred_path, pred_image)

        except Exception as e:
            print(f"Visualization error for {img_name}: {str(e)}")
            raise



def main():
    # Configuration
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    }
    
    config = {
        'gt_path': "D:/GitHub/Railway_Safety_system/annotations/instances_default_updated.json",
        'val_path': "D:\\video\\validation",
        'output_path': "D:/GitHub/Railway_Safety_system/evaluation_results_withoutdepth",
        'calibration_path': "D:/GitHub/Railway_Safety_system/station_calibration.json",

        'model_paths': {
            'edge_sam': "D:/GitHub/Railway_Safety_system/weights/edge_sam_3x.pth",
            'rail_model': "D:/GitHub/Railway_Safety_system/weights/chromatic-laughter-5",
            'depth_any' : "D:/GitHub/Railway_Safety_system/weights/depth_anything_v2_vits.pth"
        }
    }
    calibration_params = load_calibration_params('D:/GitHub/Railway_Safety_system/00340_calibration_params_after_340.json')

    # timing
    total_det_time = 0  # YOLO-World timing
    total_track_time = 0  # TEP-Net Railroad Track segmentation Time
    total_sam_time = 0  # SAM segmentation time
    total_depth_time = 0  # Depth estimation time
    total_frames = 0

    try:
        # Load calibration data
        with open(config['calibration_path'], 'r') as f:
            calibration_data = json.load(f)

        # Initialize evaluator with distance calculation capability
        evaluator = EnhancedDetectionEvaluator(
            config['gt_path'],
            config['val_path'],
            config['output_path'],
            calibration_data
        )

        # Initialize models
        print("Loading models...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # YOLO World
        yolo_model = YOLOWorld(model_id="yolo_world/v2-l")
        
        # EdgeSAM for rail segmentation
        edge_sam = build_edge_sam(checkpoint=config['model_paths']['edge_sam']).to(device)
        sam_predictor = SamPredictor(edge_sam)
        
        # Depth model
        depth_anything = DepthAnythingV2(**model_configs['vits'])
        # depth_anything.load_state_dict(torch.load('D:/GitHub/Railway_Safety_system/weights/depth_anything_v2_vits.pth', map_location='cpu'))
        depth_anything.load_state_dict(torch.load(config['model_paths']['depth_any'], map_location='cpu'))
        depth_anything = depth_anything.to(device).eval()
        
        # Rail detector
        rail_detector = Detector(
            model_path=config['model_paths']['rail_model'],
            crop_coords="auto",
            runtime="pytorch",
            device=device
        )

        # Process validation images
        print("\nProcessing validation images...")
        image_files = [f for f in os.listdir(config['val_path'])
                      if f.endswith(('.jpg', '.jpeg', '.png'))]

        print(f"Total number of image files: {len(image_files)}")
        processed_images = set()
        
        for img_name in tqdm(image_files):
            if img_name in processed_images:
                print(f"Warning: Processing {img_name} multiple times!")
            processed_images.add(img_name)
            
        # for img_name in tqdm(image_files):
            try:
                # Load and check image
                img_path = os.path.join(config['val_path'], img_name)
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Could not load image: {img_path}")
                    continue
                
                # Get image ID
                img_id = evaluator.coco.imgs[{i['file_name']: i['id'] 
                         for i in evaluator.coco.imgs.values()}[img_name]]['id']
                
                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                depth_start = time.time()
                # Depth prediction
                depth = depth_anything.infer_image(image, input_size=720)
                if isinstance(depth, torch.Tensor):
                    depth = depth.cpu().numpy()
                depth = depth.max() - depth  # 反转深度值
                depth = (depth - depth.min()) / (depth.max() - depth.min()) 

                # depth_map = depth
                # 应用校准
                if calibration_params is not None:
                    depth_map = process_depth_map(depth, calibration_params)
                else:
                    # 如果没有校准参数，也需要进行基础的归一化和范围映射
                    depth_map = (depth - depth.min()) / (depth.max() - depth.min())
                    print("Warning: Using uncalibrated depth")
                
                depth_end = time.time()



                total_depth_time += (depth_end - depth_start)

                # Object detection with separate confidence thresholds
                det_start = time.time()
                yolo_model.set_classes(['person'])
                person_results = yolo_model.infer(image, confidence=0.19, iou=0.4)
                person_detections = sv.Detections.from_inference(person_results)
                
                yolo_model.set_classes(['car'])
                car_results = yolo_model.infer(image, confidence=0.4, iou=0.4)
                car_detections = sv.Detections.from_inference(car_results)

                det_end = time.time()
                total_det_time += (det_end - det_start)


                # First combine person and car detections
                if len(person_detections) > 0 and len(car_detections) > 0:
                    combined_detections = sv.Detections(
                        xyxy=np.vstack((person_detections.xyxy, car_detections.xyxy)),
                        confidence=np.concatenate((person_detections.confidence, car_detections.confidence)),
                        class_id=np.concatenate((person_detections.class_id, car_detections.class_id + 1))
                    )
                elif len(person_detections) > 0:
                    combined_detections = person_detections
                elif len(car_detections) > 0:
                    combined_detections = car_detections
                    combined_detections.class_id = combined_detections.class_id + 1
                else:
                    combined_detections = sv.Detections.empty()

                # SAM timing
                sam_start = time.time()
                # Generate masks using SAM predictor
                all_masks = segment(sam_predictor=sam_predictor, 
                                  rail_detector=rail_detector,
                                  image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                                  detections=combined_detections)

                sam_end = time.time()
                total_sam_time += (sam_end - sam_start)

                total_frames += 1

                # Add rail track information
                rail_track_xyxy = np.array([[0, 0, image.shape[1], image.shape[0]]])
                rail_track_confidence = np.array([1.0])
                rail_track_class_id = np.array([CLASSES.index('rail-track')])
                
                # Create final detections with masks
                detections = sv.Detections.empty()
                detections.mask = all_masks
                detections.xyxy = np.vstack((rail_track_xyxy, combined_detections.xyxy))
                detections.confidence = np.concatenate((rail_track_confidence, combined_detections.confidence))
                detections.class_id = np.concatenate((rail_track_class_id, combined_detections.class_id))

                # Extract rail mask for distance calculations
                rail_mask = detections.mask[0] if detections.mask is not None else None

                # Process detections and calculate distances
                for i, (xyxy, class_id) in enumerate(zip(detections.xyxy, detections.class_id)):
                    if CLASSES[class_id] in ['person', 'car']:
                        try:
                            if detections.mask is not None:
                                person_mask = detections.mask[i]
                                
                                # Evaluate horizontal distance
                                ref_dist, calc_dist, proj_point, match_point = evaluator.evaluate_horizontal_distance(
                                    person_mask,
                                    rail_mask,
                                    depth_map,
                                    img_name
                                )
                                    
                        except Exception as e:
                            print(f"Error processing detection {i} in {img_name}: {str(e)}")
                            continue

                # Evaluate detections and rail segmentation
                evaluator.evaluate_detections(detections, img_id)
                if rail_mask is not None:
                    evaluator.evaluate_rail_segmentation(rail_mask, img_id)
                
                # # Save visualization
                # evaluator.save_visualization(
                #     image=image, 
                #     detections=detections,
                #     rail_mask=rail_mask,
                #     image_id=img_id,
                #     img_name=img_name,
                #     depth_map=depth_map
                # )

                # Replace the evaluation section in the main loop with:
                evaluator.evaluate_image(
                    image=image,
                    detections=detections,
                    rail_mask=rail_mask,
                    depth_map=depth_map,
                    image_id=img_id,
                    img_name=img_name
                )

            except Exception as e:
                print(f"\nError processing {img_name}: {str(e)}")
                continue

        # Printing Performance Metrics
        print("\nPerformance Metrics:")
        
        print("\nObject Detection (YOLO-World):")
        avg_det_time = total_det_time / total_frames
        det_fps = 1.0 / avg_det_time
        print(f"Average detection time: {avg_det_time*1000:.1f} ms")
        print(f"Detection FPS: {det_fps:.1f}")
        
        print("\nSegmentation (SAM):")
        avg_sam_time = total_sam_time / total_frames
        sam_fps = 1.0 / avg_sam_time
        print(f"Average segmentation time: {avg_sam_time*1000:.1f} ms")
        print(f"Segmentation FPS: {sam_fps:.1f}")

        print("\nDepth Estimation:")
        avg_depth_time = total_depth_time / total_frames
        depth_fps = 1.0 / avg_depth_time
        print(f"Average depth estimation time: {avg_depth_time*1000:.1f} ms")
        print(f"Depth Estimation FPS: {depth_fps:.1f}")
        
        print("\nOverall Processing:")
        total_avg_time = (total_det_time + total_track_time + total_sam_time + total_depth_time) / total_frames
        total_fps = 1.0 / total_avg_time
        print(f"Average total processing time: {total_avg_time*1000:.1f} ms")
        print(f"Total FPS: {total_fps:.1f}")

        # Print final metrics
        evaluator.print_metrics()
        evaluator.print_distance_metrics()


    except Exception as e:
        print(f"Critical error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()

