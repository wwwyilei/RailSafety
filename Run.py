from __future__ import absolute_import, division, print_function
import os
import torch
import cv2
from tqdm import tqdm
import supervision as sv
import numpy as np
from segmentanything.segment_anything.predictor2 import SamPredictor
from DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2
import matplotlib as mpl
import matplotlib.cm as cm
from EdgeSAM.setup_edge_sam import build_edge_sam
from inference.models.yolo_world.yolo_world import YOLOWorld
import time
from typing import List
import json
from PIL import Image
from src.utils.interface import Detector
from src.utils.visualization import draw_egopath


def load_config(config_path='D:\GitHub\Railway_Safety_system\config.json'):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


config = load_config()

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

        calibrated_depth = a * np.exp(b * depth) + c  
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

    
def load_model(config):
    """Load all required models with configuration"""
    # set device
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    
    # load yolo model
    yolo_model = YOLOWorld(model_id=config['models']['yolo']['model_id'])
    
    # load EdgeSAM model
    edge_sam = build_edge_sam(checkpoint=config['paths']['EdgeSAM_CHECKPOINT']).to(device)
    sam_predictor = SamPredictor(edge_sam)
    
    # load depth estimation model
    depth_anything = DepthAnythingV2(**config['models']['depth']['vits'])
    depth_anything.load_state_dict(
        torch.load(config['paths']['DEPTH_CHECKPOINT'], map_location='cpu')
    )
    depth_anything = depth_anything.to(device).eval()
    
    # load rail track model
    rail_detector = Detector(
        model_path=config['paths']['RAIL_MODEL'],
        crop_coords=config['models']['rail_detector']['crop_coords'],
        runtime=config['models']['rail_detector']['runtime'],
        device=device
    )
    
    return yolo_model, sam_predictor, depth_anything, rail_detector



def create_rail_mask(image, rail_detector):
    """Create a binary mask for the rail track using the rail detector"""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    h, w = image.shape[:2]
    
    # Get rail path prediction
    # for _ in range(3):
    crop_coords = rail_detector.get_crop_coords()
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

def segment(sam_predictor: SamPredictor, rail_detector: Detector, image: np.ndarray, detections: sv.Detections) -> np.ndarray:
    """Modified segment function for proper mask extraction"""
    sam_predictor.set_image(image)
    result_masks = []
    
    # First get rail track mask
    rail_mask = create_rail_mask(image, rail_detector)
    result_masks.append(rail_mask)
    
    # Then process all detected objects (person and car) using SAM
    for box in detections.xyxy:
        masks, scores, _ = sam_predictor.predict(box=box, multimask_output=True)
        index = np.argmax(scores)
        result_masks.append(masks[index])
    
    return np.array(result_masks)


def calculate_instance_depths(detections, depth_map):
    instance_depths = []
    classes = config['detection_settings']['classes']
    for mask, xyxy in zip(detections.mask, detections.xyxy):
        if classes[detections.class_id[len(instance_depths)]] in ['person', 'car']:
            foot_depth = get_foot_depth(mask, depth_map, xyxy)
            instance_depths.append(foot_depth if foot_depth is not None else np.nan)
        else:
            resized_mask = cv2.resize(mask.astype(np.uint8), (depth_map.shape[1], depth_map.shape[0]), interpolation=cv2.INTER_NEAREST)
            masked_depth = depth_map[resized_mask > 0]
            mean_depth = np.mean(masked_depth)
            instance_depths.append(mean_depth)
    return instance_depths

def get_foot_depth(mask, depth_map, xyxy):
    bottom_y = int(xyxy[3])
    left_x, right_x = int(xyxy[0]), int(xyxy[2])
    
    foot_region = mask[bottom_y-10:bottom_y+1, left_x:right_x]
    foot_depths = depth_map[bottom_y-10:bottom_y+1, left_x:right_x][foot_region]
    
    if len(foot_depths) > 0:
        return np.mean(foot_depths)
    else:
        return None

def calculate_overlap(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x1 < x2 and y1 < y2:
        return (x2 - x1) * (y2 - y1)
    return 0

def find_non_overlapping_position(image, box, xyxy, existing_boxes):
    height, width = image.shape[:2]
    box_width, box_height = box[2] - box[0], box[3] - box[1]
    
    # Try top position
    top_position = [int(xyxy[0]), max(0, int(xyxy[1] - box_height))]
    top_box = [top_position[0], top_position[1], top_position[0] + box_width, top_position[1] + box_height]
    
    # Try bottom position
    bottom_position = [int(xyxy[0]), min(height - box_height, int(xyxy[3]))]
    bottom_box = [bottom_position[0], bottom_position[1], bottom_position[0] + box_width, bottom_position[1] + box_height]
    
    # Check overlap
    top_overlap = sum(calculate_overlap(top_box, existing) for existing in existing_boxes)
    bottom_overlap = sum(calculate_overlap(bottom_box, existing) for existing in existing_boxes)
    
    return top_position if top_overlap <= bottom_overlap else bottom_position

def draw_labels_with_dynamic_positioning(image, detections, instance_depths, horizontal_distances, config):
    existing_boxes = []
    classes = config['detection_settings']['classes']
    
    for i, (xyxy, class_id, depth) in enumerate(zip(detections.xyxy, detections.class_id, instance_depths)):
        if classes[class_id] in ['person', 'car']:
            horizontal_distance = horizontal_distances[i] if i < len(horizontal_distances) else None
            label, color = format_label(classes[class_id], depth, horizontal_distance)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            line_spacing = 20
            
            max_width = max(cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in label)
            box_width = max_width + 10
            box_height = len(label) * line_spacing + 10
            
            position = find_non_overlapping_position(image, 
                [int(xyxy[0]), int(xyxy[1]), int(xyxy[0]) + box_width, int(xyxy[1]) + box_height],
                xyxy,
                existing_boxes)
            
            draw_label_with_box(image, label, position, color)
            
            existing_boxes.append([position[0], position[1], position[0] + box_width, position[1] + box_height])

def format_label(class_name, depth, horizontal_distance):
    depth_str = f"{depth:.2f}m" if not np.isnan(depth) else "N/A"
    if horizontal_distance is not None:
        dist_str = f"{horizontal_distance:.2f}m"
        if horizontal_distance < 1:
            color = (0, 0, 255)  # Red
        elif horizontal_distance < 2:
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 255, 0)  # Green
    else:
        dist_str = "N/A"
        color = (255, 255, 255)  # White
    return [f"Distance to rail: {dist_str}"], color
    # return [f"Distance to rail: {dist_str}",f"Depth: {depth_str}"], color

def draw_label_with_box(image, lines, position, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_spacing = 20

    max_width = 0
    for line in lines:
        (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_width = max(max_width, w)
    
    box_width = max_width + 10
    box_height = len(lines) * line_spacing + 10

    overlay = image.copy()
    cv2.rectangle(overlay, position, (position[0] + box_width, position[1] + box_height), (255, 255, 255), -1)
    alpha = 0.9
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    for i, line in enumerate(lines):
        y = position[1] + (i + 1) * line_spacing
        cv2.putText(image, line, (position[0] + 5, y), font, font_scale, (0, 0, 0), thickness)

    cv2.rectangle(image, position, (position[0] + box_width, position[1] + box_height), color, 2)

def find_matching_depth_on_rail(rail_mask, depth_map, target_depth, person_foot_position, tolerance=3):
    """
    Find the point on the railroad track that matches the depth of the target
    """
    # Get the coordinates of all points on the railroad tracks
    rail_points = np.argwhere(rail_mask)
    rail_depths = depth_map[rail_points[:, 0], rail_points[:, 1]]
    
    # Finding deep matches
    absolute_diff = np.abs(rail_depths - target_depth)
    matching_indices = np.where(absolute_diff < tolerance )[0]
    
    if len(matching_indices) > 0:
        matching_points = rail_points[matching_indices]
        matching_depths = rail_depths[matching_indices]
        
        # Calculate the distance to the character's foot position
        distances = np.sqrt(np.sum((matching_points - person_foot_position) ** 2, axis=1))
        nearest_index = np.argmin(distances)
        
        return (matching_points[nearest_index], 
                matching_depths[nearest_index],
                absolute_diff[matching_indices[nearest_index]])
    
    return None, None, None

def calculate_horizontal_distance(rail_mask, person_mask, depth_map, gauge_width=1.435):
    """
    To calculate the lateral distance:
    1. find the closest point and its bottom projection point
    2. find the matching point on the railroad track using the depth of the center of the foot.
    3. Calculate the distance from the projected point to the matching point.
    """
    # Find the key points
    nearest_point, projected_point, foot_center = find_critical_points(person_mask, rail_mask)
    
    if nearest_point is None or projected_point is None or foot_center is None:
        return None, None, None
    
    # Use the depth of the foot position to find the matching point
    foot_depth = depth_map[foot_center[0], foot_center[1]]
    matching_point, matching_depth, _ = find_matching_depth_on_rail(
        rail_mask, 
        depth_map, 
        foot_depth,  # Use foot depth
        foot_center  # Using the full foot_center coordinates
    )
    
    if matching_point is None:
        return None, None, None
    
    # Calculate scale
    rail_y = matching_point[0]
    rail_points = np.where(rail_mask[rail_y, :])[0]
    
    if len(rail_points) < 2:
        return None, None, None
    
    track_width_pixels = rail_points[-1] - rail_points[0]
    scale = gauge_width / track_width_pixels
    
    # Calculate the horizontal distance from the projected position of the nearest point to the matching point
    pixel_distance = abs(projected_point[1] - matching_point[1])
    meter_distance = pixel_distance * scale
    
    return meter_distance, projected_point, matching_point

def get_mask_edges(mask):

    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel)
    edge = mask.astype(np.uint8) - eroded
    return np.argwhere(edge)

def find_critical_points(person_mask, rail_mask):
    """
    Find the key point: the point closest to the center line and its bottom projection
    """
    image_width = person_mask.shape[1]
    midline_x = image_width // 2
    
    # Get mask edge points
    person_edge_points = get_mask_edges(person_mask)
    if len(person_edge_points) == 0:
        return None, None, None
    
    # Find the bottom position
    mask_points = np.argwhere(person_mask)
    bottom_y = np.max(mask_points[:, 0])
    bottom_points = mask_points[mask_points[:, 0] == bottom_y]
    bottom_x = int(np.mean(bottom_points[:, 1]))
    bottom_center = (bottom_y, bottom_x)
    
    # Find the point closest to the center line找到距离中线最近的点
    distances_to_midline = np.abs(person_edge_points[:, 1] - midline_x)
    nearest_idx = np.argmin(distances_to_midline)
    nearest_point = person_edge_points[nearest_idx]
    
    # Create the projection point (keep the x-coordinate and use the bottom y-coordinate)
    projected_point = (bottom_y, nearest_point[1])
    
    return nearest_point, projected_point, bottom_center

def find_rail_edges(rail_mask):
    """
    Find the left and right edges of the tracks
    
    """
    edges = []
    height = rail_mask.shape[0]
    width = rail_mask.shape[1]
    

    for y in range(height):
        row_points = np.where(rail_mask[y, :])[0]
        if len(row_points) >= 2:

            edges.append({
                'y': y,
                'left': row_points[0],
                'right': row_points[-1]
            })
    
    return edges





def main():
    # Load models
    model_load_start = time.time()
    yolo_model, sam_predictor, depth_anything, rail_detector = load_model(config)
    model_load_time = time.time() - model_load_start
    print(f"Model loading time: {model_load_time:.3f} seconds")

    calibration_params = load_calibration_params(config['paths'].get('CALIBRATION_FILE'))

    classes = config['detection_settings']['classes']
    detection_settings = config['detection_settings']
    class_specific_settings = detection_settings['class_specific_settings']
    gauge_width = config['rail_settings']['gauge_width']

    person_settings = class_specific_settings['person']
    car_settings = class_specific_settings['car']
    rail_track_idx = classes.index('rail-track')
    car_idx = classes.index('car')
    rail_track_idx = classes.index('rail-track')
    class_labels = config['detection_settings']['classes']
    output_dir = config['paths']['OUTPUT_DIRECTORY']

    # Get image paths
    image_paths = sv.list_files_with_extensions(
        directory=config['paths']['IMAGES_DIRECTORY'],
        extensions=config['image_settings']['extensions']
    )
    # Start total timing
    total_start_time = time.time()
    
    try:
        # Process images
        for image_path in tqdm(image_paths, desc="Processing Images"):
            try:
                per_image_start_time = time.time()
                
                # Image loading
                image_load_start = time.time()
                image_name = os.path.basename(image_path)
                image_path = str(image_path)
                image = cv2.imread(image_path)
                image_load_time = time.time() - image_load_start
                
                if image is None:
                    print(f"Failed to load image: {image_path}")
                    continue
                
                detection_start = time.time()
                
                # First detect people with lower confidence
                yolo_model.set_classes(['person'])
                person_results = yolo_model.infer(image, confidence= person_settings['confidence_threshold'],iou= person_settings['iou_threshold'])  # Lower confidence for people
                person_detections = sv.Detections.from_inference(person_results)
                
                # Then detect cars with original confidence
                yolo_model.set_classes(['car'])
                car_results = yolo_model.infer(image, confidence= car_settings['confidence_threshold'], iou= car_settings['iou_threshold'])  # Original confidence for cars
                car_detections = sv.Detections.from_inference(car_results)
                
                # Combine detections
                if len(person_detections) > 0 and len(car_detections) > 0:
                    detections = sv.Detections(
                        xyxy=np.vstack((person_detections.xyxy, car_detections.xyxy)),
                        confidence=np.concatenate((person_detections.confidence, car_detections.confidence)),
                        class_id=np.concatenate((person_detections.class_id, car_detections.class_id + 1))  # +1 for car class_id
                    )
                elif len(person_detections) > 0:
                    detections = person_detections
                elif len(car_detections) > 0:
                    detections = car_detections
                    detections.class_id = detections.class_id + 1  # Adjust class_id for cars
                else:
                    detections = sv.Detections.empty()
                
                detection_time = time.time() - detection_start
                
                # Segmentation
                segment_start = time.time()
                all_masks = segment(sam_predictor=sam_predictor, rail_detector=rail_detector, 
                                image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections=detections)
                
                # Combine detections
                rail_track_xyxy = np.array([[0, 0, image.shape[1], image.shape[0]]])
                rail_track_confidence = np.array([1.0])
                rail_track_class_id = np.array([rail_track_idx])
                
                combined_detections = sv.Detections.empty()
                combined_detections.mask = all_masks
                combined_detections.xyxy = np.vstack((rail_track_xyxy, detections.xyxy))
                combined_detections.confidence = np.concatenate((rail_track_confidence, detections.confidence))
                combined_detections.class_id = np.concatenate((rail_track_class_id, detections.class_id))
                segment_time = time.time() - segment_start


                # Depth prediction
                depth_start = time.time()
                depth = depth_anything.infer_image(image, input_size=720)
                if isinstance(depth, torch.Tensor):
                    depth = depth.cpu().numpy()
                depth = depth.max() - depth  # Inversion depth value
                depth = (depth - depth.min()) / (depth.max() - depth.min()) 

                # apply calibration
                if calibration_params is not None:
                    depth_map = process_depth_map(depth, calibration_params)
                else:
                    # no calibration parameters, normalization and range mapping
                    depth_map = (depth - depth.min()) / (depth.max() - depth.min())
                    print("Warning: Using uncalibrated depth")
                depth_time = time.time() - depth_start

                # Instance depths calculation
                instance_start = time.time()
                instance_depths = calculate_instance_depths(combined_detections, depth_map)
                instance_time = time.time() - instance_start
                
                # Visualization preparation
                vis_prep_start = time.time()
                bounding_box_annotator = sv.BoxAnnotator()
                mask_annotator = sv.MaskAnnotator()

                # Initialize annotated images
                annotated_image = image.copy()
                depth_visual = depth_map.copy()
                vmax = np.percentile(depth_visual, 95)
                normalizer = mpl.colors.Normalize(vmin=depth_visual.min(), vmax=vmax)
                mapper = cm.ScalarMappable(norm=normalizer, cmap='Spectral_r')
                depth_colormap = (mapper.to_rgba(depth_visual)[:, :, :3] * 255).astype(np.uint8)
                depth_colormap_resized = cv2.resize(depth_colormap, (image.shape[1], image.shape[0]), 
                                                  interpolation=cv2.INTER_NEAREST)
                depth_colormap_bgr = cv2.cvtColor(depth_colormap_resized, cv2.COLOR_RGB2BGR)
                annotated_depth_image = depth_colormap_bgr.copy()

                # Apply masks and bounding boxes
                labels = [f"{class_labels[class_id]}" for class_id in combined_detections.class_id]
                # labels = ["" for _ in combined_detections.class_id]
                annotated_image = mask_annotator.annotate(scene=annotated_image, detections=combined_detections)
                annotated_image = bounding_box_annotator.annotate(
                                scene=annotated_image, 
                                detections=combined_detections
                                # labels=labels
                            )
                annotated_depth_image = mask_annotator.annotate(scene=annotated_depth_image, detections=combined_detections)
                annotated_depth_image = bounding_box_annotator.annotate(
                    scene=annotated_depth_image, 
                    detections=combined_detections
                    # labels=labels
                )
                vis_prep_time = time.time() - vis_prep_start

                # Distance calculation and visualization
                dist_vis_start = time.time()
                horizontal_distances = []
                rail_mask = combined_detections.mask[0]

                for i, (xyxy, class_id, depth) in enumerate(zip(combined_detections.xyxy, 
                                                              combined_detections.class_id, 
                                                              instance_depths)):
                    horizontal_distance = None
                    if class_labels[class_id] in ['person', 'car']:
                        person_mask = combined_detections.mask[i]
                        
                        # Calculate distance
                        distance, projected_point, matching_point = calculate_horizontal_distance(
                            rail_mask, 
                            person_mask, 
                            depth_map
                        )
                        
                        if distance is not None:
                            horizontal_distance = distance
                            if projected_point is not None and matching_point is not None:
                                # Draw on original image
                                cv2.circle(annotated_image, (projected_point[1], projected_point[0]), 4, (255, 0, 255), -1)
                                cv2.circle(annotated_image, (matching_point[1], matching_point[0]), 4, (0, 255, 0), -1)
                                cv2.line(annotated_image, (projected_point[1], projected_point[0]), 
                                        (matching_point[1], matching_point[0]), (255, 255, 255), 2)
                                
                                # # Add distance label
                                # label = f"{distance:.2f}m"
                                # cv2.putText(annotated_image, label, 
                                #         ((projected_point[1] + matching_point[1])//2, projected_point[0] - 10),
                                #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                                
                                # Draw on depth image
                                cv2.circle(annotated_depth_image, (projected_point[1], projected_point[0]), 4, (255, 0, 255), -1)
                                cv2.circle(annotated_depth_image, (matching_point[1], matching_point[0]), 4, (0, 255, 0), -1)
                                cv2.line(annotated_depth_image, (projected_point[1], projected_point[0]), 
                                        (matching_point[1], matching_point[0]), (255, 255, 255), 2)
                                # cv2.putText(annotated_depth_image, label, 
                                #         ((projected_point[1] + matching_point[1])//2, projected_point[0] - 10),
                                #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    horizontal_distances.append(horizontal_distance)
                dist_vis_time = time.time() - dist_vis_start

                # Draw labels
                label_start = time.time()
                draw_labels_with_dynamic_positioning(annotated_image, combined_detections, instance_depths, horizontal_distances, config)
                draw_labels_with_dynamic_positioning(annotated_depth_image, combined_detections, instance_depths, horizontal_distances, config)
                label_time = time.time() - label_start

                # Save results
                save_start = time.time()
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                output_image_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_annotated.png")
                output_depth_image_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_depth_annotated.png")
                output_depth_data_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_depth.npy")

                cv2.imwrite(output_image_path, annotated_image)
                cv2.imwrite(output_depth_image_path, annotated_depth_image)
                np.save(output_depth_data_path, depth_visual)
                save_time = time.time() - save_start

                # Print detailed timing information
                per_image_time = time.time() - per_image_start_time
                print(f"\nDetailed timing for {image_name}:")
                print(f"Image loading: {image_load_time:.3f}s")
                print(f"Object detection: {detection_time:.3f}s")
                print(f"Segmentation: {segment_time:.3f}s")
                print(f"Depth prediction: {depth_time:.3f}s")
                print(f"Instance depth calculation: {instance_time:.3f}s")
                print(f"Visualization preparation: {vis_prep_time:.3f}s")
                print(f"Distance calculation and visualization: {dist_vis_time:.3f}s")
                print(f"Label drawing: {label_time:.3f}s")
                print(f"Result saving: {save_time:.3f}s")
                print(f"Total image processing time: {per_image_time:.3f}s")
                print("-" * 50)
                
            except Exception as e:
                print(f"Error processing {image_name}: {e}")
                continue

        total_time = time.time() - total_start_time
        print(f"\nOverall Statistics:")
        print(f"Model loading time: {model_load_time:.3f}s")
        print(f"Total processing time: {total_time:.3f}s")
        print(f"Average time per image: {total_time/len(image_paths):.3f}s")
        
    except Exception as e:
        print(f"Error in main process: {e}")

if __name__ == "__main__":
    main()



# def visualize_distance_calculation(image, person_mask, rail_mask, depth_map):

#     display = image.copy()
    
#     # Display Mask Outline
#     person_contours, _ = cv2.findContours(person_mask.astype(np.uint8), 
#                                         cv2.RETR_EXTERNAL, 
#                                         cv2.CHAIN_APPROX_SIMPLE)
#     cv2.drawContours(display, person_contours, -1, (255, 255, 255), 1)
    

#     midline_x = display.shape[1] // 2
#     cv2.line(display, (midline_x, 0), (midline_x, display.shape[0]), (128, 128, 128), 1, cv2.LINE_AA)
    

#     nearest_point, projected_point, foot_center = find_critical_points(person_mask, rail_mask)
    
#     if nearest_point is not None:

#         cv2.circle(display, (nearest_point[1], nearest_point[0]), 4, (0, 0, 255), -1)
#         cv2.putText(display, "Nearest", (nearest_point[1], nearest_point[0] - 10),
#                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        

#         cv2.circle(display, (foot_center[1], foot_center[0]), 4, (255, 0, 0), -1)
#         cv2.putText(display, "Foot", (foot_center[1], foot_center[0] - 10),
#                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        

#         cv2.line(display, (nearest_point[1], nearest_point[0]), 
#                 (projected_point[1], projected_point[0]), (0, 255, 255), 1, cv2.LINE_AA)
        
#         # Calculating distances and getting matches
#         distance, _, matching_point = calculate_horizontal_distance(rail_mask, person_mask, depth_map)
        
#         if matching_point is not None:
#             # grenn for railroad track match points
#             cv2.circle(display, (matching_point[1], matching_point[0]), 4, (0, 255, 0), -1)
#             cv2.putText(display, "Rail Match", (matching_point[1], matching_point[0] - 10),
#                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
#             # white line for the actual measured distance
#             cv2.line(display, (projected_point[1], projected_point[0]), 
#                     (matching_point[1], matching_point[0]), (255, 255, 255), 2)
            
#             # display distance value
#             label = f"{distance:.2f}m"
#             cv2.putText(display, label, 
#                       ((projected_point[1] + matching_point[1])//2, projected_point[0] - 20),
#                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#             cv2.putText(display, label, 
#                       ((projected_point[1] + matching_point[1])//2, projected_point[0] - 20),
#                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
#     return display