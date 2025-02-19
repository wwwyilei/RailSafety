import os
import torch
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, List, Optional, Dict, NamedTuple
from scipy.optimize import curve_fit
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from src.utils.interface import Detector
from DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2


PREDEFINED_CALIBRATION_POINTS = [
    (955, 1068, 0),
    (955, 1008, 0.6),
    (956, 968, 1.2),
    (957, 924, 1.8),
    (959, 891, 2.4),
    (959, 858, 3.0),
    (961, 833, 3.6),
    (961, 811, 4.2),
    (963, 791, 4.8),
    (961, 774, 5.4),
    (963, 759, 6.0),
    (964, 745, 6.6),
    (965, 734, 7.2),
    (966, 722, 7.8),
    (967, 710, 8.4),
    (966, 701, 9.0),
    (966, 692, 9.6),
    (966, 684, 10.2),
    (966, 677, 10.8),
    (965, 669, 11.4),
    (965, 661, 12.0),
    (966, 655, 12.6),
    (967, 650, 13.2),
    (967, 646, 13.8),
    (968, 642, 14.4),
    (968, 636, 15.0),
    (969, 631, 15.6),
    (970, 627, 16.2),
    (972, 622, 16.8),
    (969, 618, 17.4),
    (969, 614, 18.0),
    (966, 599, 21.0),
    (965, 586, 24.0),
    (965, 577, 27.0),
    (965, 569, 30.0),
    (964, 557, 36.0),
    (963, 546, 42.0),
    (962, 536, 48.0),
    (961, 528, 54.0),
    (961, 519, 60.0),
    (961, 507, 70.0),
    (959, 498, 80.0),
    (958, 489, 90.0)
]

# PREDEFINED_CALIBRATION_POINTS = [
#     (604, 716, 0.0),   
#     (604, 716, 0.6),
#     (604, 707, 1.2),
#     (604, 697, 1.8),
#     (604, 688, 2.4),
#     (605, 679, 3.0),
#     (606, 671, 3.6),
#     (608, 663, 4.2),
#     (609, 655, 4.8),
#     (610, 647, 5.2),
#     (611, 639, 5.8),
#     (613, 631, 6.4),
#     (615, 623, 7.0),
#     (617, 616, 7.6),
#     (618, 609, 8.0),
#     (619, 602, 8.6),
#     (620, 595, 9.2),
#     (622, 589, 9.8),
#     (623, 574, 13.0),
#     (624, 558, 16.0),
#     (623, 544, 19.0),
#     (621, 528, 22.0),
#     (614, 514, 25.0),
#     (609, 502, 28.0),
#     (605, 492, 31.0),
#     (601, 476, 40.0),
#     (598, 463, 50.0),
#     (594, 450, 60.0),
#     (592, 437, 70.0),
#     (592, 427, 80.0),
#     (590, 416, 90.0)
# ]
class RailMeasurement(NamedTuple):
    """Railway measurement data structure"""
    y: int               # y coordinate
    width: float         # track distance (pixels)
    depth: float         # depth value
    left_x: float       # left track x coordinate
    right_x: float      # right track x coordinate

class EnhancedRailwayCalibrator:
    def __init__(self,
                 depth_model,
                 rail_detector=None,
                 device: str = 'cuda',
                 rail_gauge: float = 1.435,
                 sleeper_distance: float = 0.6):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.depth_model = depth_model
        self.rail_detector = rail_detector
        self.rail_gauge = rail_gauge
        self.sleeper_distance = sleeper_distance
        self.manual_points = []
        self.calibration_cache = {}

    def get_depth_map(self, image: np.ndarray) -> np.ndarray:
        """Get initial depth map from image"""
        with torch.no_grad():
            depth = self.depth_model.infer_image(image, input_size=720)
        if isinstance(depth, torch.Tensor):
            depth = depth.cpu().numpy()
        
        depth = depth.max() - depth
        return (depth - depth.min()) / (depth.max() - depth.min())


    def detect_rails(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect railway tracks"""
        if self.rail_detector is None:
            return None, None
            
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        rail_path = None
        
        for _ in range(50):
            rail_path = self.rail_detector.detect(pil_image)
        
        if not rail_path or len(rail_path) != 2:
            return None, None
            
        left_rail = np.array(rail_path[0])
        right_rail = np.array(rail_path[1])
        
        if len(left_rail) < 2 or len(right_rail) < 2:
            return None, None
            
        left_rail = left_rail[left_rail[:, 1].argsort()]
        right_rail = right_rail[right_rail[:, 1].argsort()]
        
        return left_rail, right_rail


    # def _evaluate_detection_quality(self, left_rail, right_rail, width, height):
    #     """Assessing the quality of railroad track inspection"""
    #     try:
    #         # 1. Checking the reasonableness of the gauge
    #         distances = []
    #         for i in range(min(len(left_rail), len(right_rail))):
    #             dist = right_rail[i, 0] - left_rail[i, 0]
    #             if dist <= 0:  # Gauge should not be negative
    #                 return 0
    #             distances.append(dist)
            
    #         avg_distance = np.mean(distances)
    #         std_distance = np.std(distances)
            
    #         # 2. check coverage
    #         y_range = max(np.max(left_rail[:, 1]), np.max(right_rail[:, 1])) - \
    #                 min(np.min(left_rail[:, 1]), np.min(right_rail[:, 1]))
    #         y_coverage = y_range / height
            
    #         # 3. Calculating Smoothness
    #         left_smoothness = np.mean(np.abs(np.diff(left_rail[:, 0])))
    #         right_smoothness = np.mean(np.abs(np.diff(right_rail[:, 0])))
            
    #         # overall rating
    #         distance_score = 1.0 - (std_distance / avg_distance)  # Gauge Stability
    #         coverage_score = y_coverage  
    #         smoothness_score = 1.0 / (1.0 + (left_smoothness + right_smoothness) / 2)  
            
    #         total_score = (distance_score + coverage_score + smoothness_score) / 3
    #         return max(0, total_score)
            
    #     except Exception as e:
    #         print(f"Error in detection quality evaluation: {e}")
    #         return 0


    def load_predefined_points(self):
        """Load predefined calibration points"""
        self.manual_points = PREDEFINED_CALIBRATION_POINTS.copy()
        print(f"Loaded {len(self.manual_points)} predefined calibration points")


    # def sleeper_calibration(self, depth_map: np.ndarray) -> np.ndarray:
    #     """First stage: Calibrate depth using sleeper points"""
    #     if not self.manual_points:
    #         return depth_map
        
    #     points = np.array([(depth_map[y, x], depth) for x, y, depth in self.manual_points])
    #     pred_depths = points[:, 0]
    #     real_depths = points[:, 1]

    #     def calibration_func(x, a, b, c):
    #         return a * np.exp(b * x) + c

    #     try:
    #         popt, _ = curve_fit(calibration_func, pred_depths, real_depths)
    #         calibrated = calibration_func(depth_map, *popt)
    #         self.calibration_cache['sleeper_params'] = popt
    #         return calibrated
    #     except Exception as e:
    #         print(f"Warning: Exponential calibration failed ({str(e)}), using linear calibration")
    #         slope, intercept = np.polyfit(pred_depths, real_depths, 1)
    #         self.calibration_cache['sleeper_params'] = (slope, intercept)
    #         return depth_map * slope + intercept

    def sleeper_calibration(self, depth_map: np.ndarray) -> np.ndarray:
        """First stage: Calibrate depth using sleeper points"""
        if not self.manual_points:
            return depth_map
        
        points = np.array([(depth_map[y, x], depth) for x, y, depth in self.manual_points])
        pred_depths = points[:, 0]
        real_depths = points[:, 1]

        def calibration_func(x, a, b, c):
            return a * np.exp(b * x) + c

        try:
            # 1. exponential fit
            popt, _ = curve_fit(calibration_func, pred_depths, real_depths)
            calibrated = calibration_func(depth_map, *popt)
            
            # 2. Adjust the output range to ensure that the minimum value is 0
            min_depth = calibrated.min()
            calibrated = calibrated - min_depth  # Adjust the minimum value to 0
            
            # 3. Rescale to maintain relative proportions
            scale_factor = (real_depths.max() - 0) / (calibrated.max() - 0)
            calibrated = calibrated * scale_factor
            
            # 4. Update calibration parameters to reflect new transformations
            self.calibration_cache['sleeper_params'] = popt
            self.calibration_cache['scale_factor'] = scale_factor
            self.calibration_cache['depth_offset'] = min_depth
            
            return calibrated
            
        except Exception as e:
            print(f"Warning: Exponential calibration failed ({str(e)}), using linear calibration")
            # The case of linear calibration
            slope, intercept = np.polyfit(pred_depths, real_depths, 1)
            calibrated = depth_map * slope + intercept
            
            # Similarly adjust the output range of the linear calibration
            min_depth = calibrated.min()
            calibrated = calibrated - min_depth
            scale_factor = (real_depths.max() - 0) / (calibrated.max() - 0)
            calibrated = calibrated * scale_factor
            
            self.calibration_cache['sleeper_params'] = (slope, intercept)
            self.calibration_cache['scale_factor'] = scale_factor
            self.calibration_cache['depth_offset'] = min_depth
            
            return calibrated



    def rail_optimization(self, calibrated_depth: np.ndarray, 
                        left_rail: np.ndarray, 
                        right_rail: np.ndarray,
                        image: np.ndarray) -> np.ndarray:
        """Optimized depth map using standard gauge for railroad tracks"""
        height, width = calibrated_depth.shape
        
        # 1. Start with depth map optimization
        # 1.1 Collect measurement data
        y_coords = np.linspace(height-20, height//1.3, 60)
        measurements = []
        for y in y_coords:
            y = int(y)
            left_idx = np.argmin(np.abs(left_rail[:, 1] - y))
            right_idx = np.argmin(np.abs(right_rail[:, 1] - y))
            
            left_x = int(left_rail[left_idx, 0])
            right_x = int(right_rail[right_idx, 0])
            
            width_pixels = right_x - left_x
            avg_depth = np.mean(calibrated_depth[y, left_x:right_x])
            
            if 0 < avg_depth < 100:
                # measurements.append(RailMeasurement(y, width_pixels, avg_depth))
                measurements.append(RailMeasurement(
                    y=y,
                    width=width_pixels,
                    depth=avg_depth,
                    left_x=left_x,
                    right_x=right_x
                ))
        
        if not measurements:
            return calibrated_depth
        
        # 1.2 Intermediate results obtained by applying railroad track correction
        intermediate_depth = self._apply_rail_corrections(calibrated_depth, measurements)
        
        # 2. Refit parameters using optimized depth map
        if self.manual_points:
            # Getting the original depth map
            initial_depth = self.get_depth_map(image)
            
            # Collection point pairs: original depth values and optimized depth values
            points = np.array([(initial_depth[y, x], intermediate_depth[y, x]) 
                            for x, y, _ in self.manual_points])
            pred_depths = points[:, 0]  # Raw depth value
            optimized_depths = points[:, 1]  # optimized depth value

            def calibration_func(x, a, b, c):
                return a * np.exp(b * x) + c

            try:
                # Refit parameters
                popt, _ = curve_fit(calibration_func, pred_depths, optimized_depths)
                # Generate the final depth map using the new parameters
                final_depth = calibration_func(initial_depth, *popt)
                # Updating calibration parameters
                self.calibration_cache['sleeper_params'] = popt
                return final_depth
            except Exception as e:
                print(f"Warning: Final parameter fitting failed ({str(e)}), using intermediate result")
                return intermediate_depth
        
        return intermediate_depth

    def _apply_rail_corrections(self, depth_map: np.ndarray, measurements: List[RailMeasurement]) -> np.ndarray:
        """Apply railroad track correction to keep bottom depth essentially the same"""
        height, width = depth_map.shape
        result = depth_map.copy()
        
        # 1. Find the vanishing point.
        y_coords = [m.y for m in measurements]
        widths = [m.width for m in measurements]
        vp_y = self._estimate_vanishing_point_y(y_coords, widths)
        
        # 2. Select the reference point (the bottom point)
        ref_measurement = min(measurements, key=lambda m: abs(m.y - (height-20)))
        ref_depth = ref_measurement.depth
        ref_width = ref_measurement.width
        bottom_y = ref_measurement.y
        
        # 3. Design of distance-based modified weight curves
        def compute_distance_weight(y: float) -> float:
            """
            Calculating distance-based weights
            - Bottom region (top 30% distance) weights close to 0
            - Gradually increasing in the middle region
            - Weights are highest near the vanishing point
            """
            # Normalized distance, 0 for bottom, 1 for vanishing point
            normalized_dist = (bottom_y - y) / (bottom_y - vp_y)
            
            # Use the sigmoid function to create a smooth transition
            # Adjust these parameters to change the position and steepness of the transition
            transition_point = 0.3  # Where to start adding weights (30% from the bottom)
            steepness = 10.0       # Steepness of the transition
            
            weight = 1 / (1 + np.exp(-steepness * (normalized_dist - transition_point)))
            return weight
        
        # 4. Calculate the correction factor
        correction_factors = []
        distance_weights = []


        for m in measurements:
            # Calculating depth scaling using vanishing points
            distance_ratio = (vp_y - m.y) / (vp_y - bottom_y)
            # Calculating Depth Ratios Using Gauge
            width_ratio = ref_width / m.width
            
            # Combining the two ratios to calculate the desired depth
            expected_depth = ref_depth * (distance_ratio  + width_ratio )
            
            # Calculate the correction factor
            factor = expected_depth / m.depth
            factor = np.clip(factor, 0.5, 1.5)
            
            # Calculate distance weights (hold constant)
            distance_weight = compute_distance_weight(m.y)
            
            correction_factors.append(factor)
            distance_weights.append(distance_weight)

        # 5. smoothing
        correction_factors = np.array(correction_factors)
        # Smoothing with a larger window
        window_size = 7
        correction_factors = np.convolve(correction_factors, 
                                    np.ones(window_size)/window_size, 
                                    mode='valid')
        
        # 6. Apply
        weight_scale = 0.15  # Reduced overall correction strength
        for i, m in enumerate(measurements[window_size//2:-window_size//2]):
            factor = correction_factors[i]
            
            # Creating a spatial weight map
            y_grid, x_grid = np.mgrid[:height, :width]
            
            # x-direction weights: heavy near the center of the tracks
            rail_center = (m.left_x + m.right_x) // 2
            rail_width = m.right_x - m.left_x
            x_sigma = rail_width / 2  # Adaptive adjustment to gauge
            x_weight = np.exp(-0.5 * ((x_grid - rail_center)**2 / (x_sigma**2)))
            
            # y-direction weighting: taking distance into account
            distance_weight = compute_distance_weight(m.y)
            y_sigma = 30 * (1 + distance_weight)  # Use greater smoothing range from a distance
            y_weight = np.exp(-0.5 * ((y_grid - m.y)**2 / (y_sigma**2)))
            
            # Merger weights
            weight_map = y_weight * x_weight * distance_weight
            weight_map *= weight_scale
            
            # Application Amendments
            delta = (depth_map * (factor - 1.0)) * weight_map
            delta = np.clip(delta, -0.5, 0.5)  # Further limiting the magnitude of single amendments
            result += delta
        
        # 7. Ensure that the bottom depth remains essentially the same
        # Gradual transition to original depth using linear interpolation in the bottom region
        bottom_region = int(height * 0.2)  # Bottom 20% area
        for y in range(height-bottom_region, height):
            blend = (y - (height-bottom_region)) / bottom_region
            result[y] = blend * depth_map[y] + (1-blend) * result[y]
        
        return np.clip(result, 0, 100)


    def _estimate_vanishing_point_y(self, y_coords: List[int], widths: List[float]) -> float:
        # Estimating vanishing points using gauge changes
        y_coords = np.array(y_coords)
        widths = np.array(widths)
        
        # Assuming an approximately linear relationship between gauge and y-coordinate
        try:
            p = np.polyfit(y_coords, widths, 1)
            # Extrapolate to find the position where the gauge converges to 0
            vp_y = -p[1] / p[0]

            vp_y = np.clip(vp_y, 0, min(y_coords))
            return vp_y
        except:
            # If the fit fails, return a conservative estimate
            return min(y_coords)

    def calibrate(self, image: np.ndarray, use_predefined: bool = False) -> Dict:
        """Complete calibration process"""
        # Get initial depth map
        initial_depth = self.get_depth_map(image)
        
        # Load predefined points if requested
        if use_predefined:
            self.load_predefined_points()
        elif not self.manual_points:
            print("\nPlease mark calibration points...")
            self.collect_manual_points(image, initial_depth)
        
        if not self.manual_points:
            print("No calibration points available. Returning initial depth map.")
            return {'depth_map': initial_depth}
        
        # Stage 1: Sleeper-based calibration
        print("\nPerforming sleeper-based calibration...")
        calibrated_depth = self.sleeper_calibration(initial_depth)
        
        # Save parameters before optimization
        pre_optimization_params = self.calibration_cache['sleeper_params'].copy()
        
        # Stage 2: Rail-based optimization
        print("Detecting rails for optimization...")
        left_rail, right_rail = self.detect_rails(image)
        
        if left_rail is not None and right_rail is not None:
            print("Optimizing depth using rail geometry...")
            # get the original depth map.
            optimized_depth = self.rail_optimization(calibrated_depth, left_rail, right_rail, image)
        else:
            print("Rail detection failed. Using sleeper calibration only.")
            optimized_depth = calibrated_depth

        def adjust_depth_range(depth_map):
            depth_map = depth_map - depth_map.min()  # 确保最小值为0
            max_depth = max(d for _, _, d in self.manual_points)  # 获取真实深度的最大值
            return depth_map * (max_depth / depth_map.max())  # 缩放到正确的范围

        optimized_depth = adjust_depth_range(optimized_depth)     

        # Evaluate results
        initial_error = self._compute_error(initial_depth)
        calibrated_error = self._compute_error(calibrated_depth)
        optimized_error = self._compute_error(optimized_depth)
        
        print("\nCalibration Results:")
        print(f"Initial RMSE: {initial_error:.3f}m")
        print(f"After Sleeper Calibration RMSE: {calibrated_error:.3f}m")
        print(f"After Rail Optimization RMSE: {optimized_error:.3f}m")
        
        return {
            'initial_depth': initial_depth,
            'calibrated_depth': calibrated_depth,
            'optimized_depth': optimized_depth,
            'rail_paths': (left_rail, right_rail),
            'pre_optimization_params': pre_optimization_params, 
            'metrics': {
                'initial_rmse': initial_error,
                'calibrated_rmse': calibrated_error,
                'optimized_rmse': optimized_error
            }
        }

    def _compute_error(self, depth_map: np.ndarray) -> float:
        """Compute RMSE for current calibration points"""
        if not self.manual_points:
            return 0.0
        errors = []
        for x, y, true_depth in self.manual_points:
            predicted = depth_map[y, x]
            errors.append((predicted - true_depth) ** 2)
        return np.sqrt(np.mean(errors))

    def visualize_results(self, image: np.ndarray, result: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create visualization of calibration results and return multiple views"""
        # Create depth visualizations
        def depth_to_color(depth):
            normalized = (depth - depth.min()) / (depth.max() - depth.min())
            colored = (cm.jet(normalized)[:, :, :3] * 255).astype(np.uint8)
            return cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
        
        initial_viz = depth_to_color(result['initial_depth'])
        calibrated_viz = depth_to_color(result['calibrated_depth'])
        optimized_viz = depth_to_color(result['optimized_depth'])
        
        # Mark calibration points and rails on a copy of original image
        marked_original = image.copy()
        for x, y, depth in self.manual_points:
            cv2.circle(marked_original, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(marked_original, f'{depth:.1f}m', (x+5, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw detected rails if available
        if 'rail_paths' in result and result['rail_paths'][0] is not None:
            left_rail, right_rail = result['rail_paths']
            for rail in [left_rail, right_rail]:
                points = rail.astype(np.int32)
                cv2.polylines(marked_original, [points], False, (0, 255, 0), 2)
        
        # Create full visualization
        top_row = np.hstack((marked_original, initial_viz))
        bottom_row = np.hstack((calibrated_viz, optimized_viz))
        combined = np.vstack((top_row, bottom_row))
        
        # Create depth maps combination (right column only)
        depth_maps_combined = np.vstack((initial_viz, optimized_viz))
        
        # Add labels to combined visualization
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, "Original Image", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Initial Depth", (image.shape[1]+10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Calibrated Depth", (10, image.shape[0]+30), font, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Optimized Depth", (image.shape[1]+10, image.shape[0]+30), font, 1, (255, 255, 255), 2)
        
        # Add labels to depth maps combination
        cv2.putText(depth_maps_combined, "Initial Depth", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(depth_maps_combined, "Optimized Depth", (10, image.shape[0]+30), font, 1, (255, 255, 255), 2)
        
        return combined, marked_original, depth_maps_combined

    def collect_manual_points(self, image: np.ndarray, depth_map: np.ndarray) -> None:
        """Collect manual calibration points"""
        window_name = 'Manual Calibration (Q: finish, R: reset)'
        display_img = image.copy()
        self.manual_points = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                predicted_depth = depth_map[y, x]
                print(f"\nPredicted depth at ({x}, {y}): {predicted_depth:.2f}m")
                try:
                    real_depth = float(input("Enter real depth in meters: "))
                    self.manual_points.append((x, y, real_depth))
                    
                    cv2.circle(display_img, (x, y), 3, (0, 0, 255), -1)
                    cv2.putText(display_img, f'{real_depth:.1f}m', 
                            (x+5, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.imshow(window_name, display_img)
                except ValueError:
                    print("Invalid input. Please enter a number.")

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)
        
        cv2.putText(display_img, 
                    "Click points and enter real depths. Press 'q' to finish, 'r' to reset",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
        cv2.imshow(window_name, display_img)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.manual_points = []
                display_img = image.copy()
                cv2.putText(display_img, 
                        "Click points and enter real depths",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
                cv2.imshow(window_name, display_img)

        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Enhanced Railway Depth Calibration')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Input image path')
    parser.add_argument('--rail_model_path', type=str, 
                   default="D:/GitHub/Railway_Safety_system/weights/chromatic-laughter-5",  # Change to your actual model path
                   help='Rail detection model path')
    parser.add_argument('--depth_model_path', type=str,
                       default="D:/GitHub/Railway_Safety_system/weights/depth_anything_v2_vits.pth",
                       help='Depth estimation model path')
    parser.add_argument('--output_dir', type=str,
                       default="output_calibrated",
                       help='Output directory for results')
    parser.add_argument('--use_predefined', action='store_true',
                       help='Use predefined calibration points')
    parser.add_argument('--device', type=str,
                       default='cuda',
                       help='Device to run on (cuda/cpu)')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize models
    try:
        print("\nInitializing models...")
        
        # Initialize depth model
        model_configs = {
            'vits': {'encoder': 'vits', 
                    'features': 64, 
                    'out_channels': [48, 96, 192, 384]}
        }
        depth_model = DepthAnythingV2(**model_configs['vits'])
        depth_model.load_state_dict(torch.load(args.depth_model_path, 
                                             map_location='cpu'))
        depth_model = depth_model.to(args.device).eval()
        
        # Initialize rail detector
        rail_detector = Detector(
            model_path=args.rail_model_path,
            crop_coords="auto",
            runtime="pytorch",
            device=args.device
        )

        # Create calibrator
        calibrator = EnhancedRailwayCalibrator(
            depth_model=depth_model,
            rail_detector=rail_detector,
            device=args.device
        )

        # Read image
        print(f"\nReading image from {args.image_path}")
        image = cv2.imread(args.image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {args.image_path}")

        # Perform calibration
        print("\nStarting calibration process...")
        result = calibrator.calibrate(image, args.use_predefined)
        
        if result is None:
            raise ValueError("Calibration failed")

        # Create visualization
        print("\nCreating visualization...")
        visualization, marked_original, depth_maps = calibrator.visualize_results(image, result)
        
        # Save results
        base_name = Path(args.image_path).stem
        
        # Save visualization
        viz_path = output_dir / f"{base_name}_calibration_viz.jpg"
        marked_original_path = output_dir / f"{base_name}_marked_original.jpg"
        depth_maps_path = output_dir / f"{base_name}_depth_maps.jpg"
        
        cv2.imwrite(str(viz_path), visualization)
        cv2.imwrite(str(marked_original_path), marked_original)
        cv2.imwrite(str(depth_maps_path), depth_maps)

        # Save depth maps
        initial_path = output_dir / f"{base_name}_initial_depth.npy"
        calibrated_path = output_dir / f"{base_name}_calibrated_depth.npy"
        optimized_path = output_dir / f"{base_name}_optimized_depth.npy"
        
        np.save(str(initial_path), result['initial_depth'])
        np.save(str(calibrated_path), result['calibrated_depth'])
        np.save(str(optimized_path), result['optimized_depth'])
        
        # Save calibration parameters before optimization
        params_path_before = output_dir / f"{base_name}_calibration_params_before_340.json"
        calibration_data_before = {
            'calibration_function': {
                'type': 'exponential' if len(result['pre_optimization_params']) == 3 else 'linear',
                'coefficients': result['pre_optimization_params'].tolist()  
            },
            'depth_range': {
                'min': float(result['calibrated_depth'].min()),
                'max': float(result['calibrated_depth'].max())
            }
        }

        with open(str(params_path_before), 'w') as f:
            json.dump(calibration_data_before, f, indent=4)

        # Save calibration parameters after optimization
        params_path_after = output_dir / f"{base_name}_calibration_params_after_340.json"
        calibration_data_after = {
            'calibration_function': {
                'type': 'exponential' if len(calibrator.calibration_cache.get('sleeper_params', [])) == 3 else 'linear',
                'coefficients': calibrator.calibration_cache['sleeper_params'].tolist()  
            },
            'depth_range': {
                'min': float(result['optimized_depth'].min()),
                'max': float(result['optimized_depth'].max())
            }
        }

        with open(str(params_path_after), 'w') as f:
            json.dump(calibration_data_after, f, indent=4)

        # Display results
        print("\n=== Calibration Results ===")
        print("\nDepth Ranges:")
        print(f"Initial: {result['initial_depth'].min():.2f}m - {result['initial_depth'].max():.2f}m")
        print(f"Calibrated: {result['calibrated_depth'].min():.2f}m - {result['calibrated_depth'].max():.2f}m")
        print(f"Optimized: {result['optimized_depth'].min():.2f}m - {result['optimized_depth'].max():.2f}m")
        
        print("\nError Metrics:")
        print(f"Initial RMSE: {result['metrics']['initial_rmse']:.3f}m")
        print(f"Calibrated RMSE: {result['metrics']['calibrated_rmse']:.3f}m")
        print(f"Optimized RMSE: {result['metrics']['optimized_rmse']:.3f}m")
        
        print("\nFiles saved:")
        print(f"- Visualization: {viz_path}")
        print(f"- Depth maps: {initial_path}, {calibrated_path}, {optimized_path}")
        print(f"- Calibration params before optimization: {params_path_before}")
        print(f"- Calibration params after optimization: {params_path_after}")
        
        # Show visualization
        cv2.namedWindow('Calibration Results', cv2.WINDOW_NORMAL)
        cv2.imshow('Calibration Results', visualization)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()