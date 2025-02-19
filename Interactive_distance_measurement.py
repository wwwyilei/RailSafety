import cv2
import numpy as np
import json
from typing import List, Dict, Tuple, Any
import time

class MultiReferenceDistanceMapper:
    def __init__(self, calibration_data: Dict[str, Any], image_path: str):
        self.calibration = calibration_data
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from {image_path}")
        self.display_image = self.image.copy()
        
        self.drawing = False
        self.start_point = None
        

        self.generate_symmetric_references()
        

        self.validate_reference_lines()

    def calculate_symmetric_point(self, point: Tuple[float, float], 
                                left_rail: List[Tuple[float, float]], 
                                right_rail: List[Tuple[float, float]]) -> Tuple[float, float]:

        center_line_start = (
            (left_rail[0][0] + right_rail[0][0]) / 2,
            (left_rail[0][1] + right_rail[0][1]) / 2
        )
        center_line_end = (
            (left_rail[1][0] + right_rail[1][0]) / 2,
            (left_rail[1][1] + right_rail[1][1]) / 2
        )
        

        direction = (
            center_line_end[0] - center_line_start[0],
            center_line_end[1] - center_line_start[1]
        )
        

        length = np.sqrt(direction[0]**2 + direction[1]**2)
        direction = (direction[0]/length, direction[1]/length)
        

        point_to_center = (
            point[0] - center_line_start[0],
            point[1] - center_line_start[1]
        )
        

        dot_product = (point_to_center[0] * direction[0] + 
                    point_to_center[1] * direction[1])
        projection = (
            center_line_start[0] + dot_product * direction[0],
            center_line_start[1] + dot_product * direction[1]
        )
        

        symmetric_point = (
            2 * projection[0] - point[0],
            2 * projection[1] - point[1]
        )
        
        return symmetric_point

    def generate_symmetric_references(self):

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

        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denominator) < 1e-10:  # 防止除以0
            return None
            
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (x, y)
        return None

    def get_line_region_intersection(self, point1, point2, ref_line) -> List[Tuple[float, float]]:

            edges = [
                (ref_line['line1'][0], ref_line['line1'][1]),  # 左边
                (ref_line['line1'][1], ref_line['line2'][1]),  # 上边
                (ref_line['line2'][1], ref_line['line2'][0]),  # 右边
                (ref_line['line2'][0], ref_line['line1'][0])   # 下边
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



    def find_segments_in_regions(self, point1: Tuple[float, float], 
                               point2: Tuple[float, float]) -> List[Tuple[Tuple[float, float], 
                                                                        Tuple[float, float], 
                                                                        Dict]]:

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

    def calculate_distance(self, point1, point2) -> Tuple[float, List[Dict]]:
        """Calculate the actual distance between two points, considering all areas through which the line passes"""

        segments = self.find_segments_in_regions(point1, point2)
        

        outer_ref = next(ref for ref in self.calibration['reference_lines'] 
                        if abs(ref['distance'] - 2.0) < 0.01)
        
        if not segments:
        
            H = self.compute_local_transform(outer_ref)
            p1_world = self.transform_point(point1, H)
            p2_world = self.transform_point(point2, H)
            return np.linalg.norm(p2_world - p1_world), [outer_ref]


        total_distance = 0
        used_refs = []
        last_point = point1
        

        debug_image = self.display_image.copy()
        

        for i, (seg_start, seg_end, ref_line) in enumerate(segments):

            if i == 0 and seg_start != point1:
                H = self.compute_local_transform(outer_ref)
                p1_world = self.transform_point(point1, H)
                p2_world = self.transform_point(seg_start, H)
                total_distance += np.linalg.norm(p2_world - p1_world)
                used_refs.append(outer_ref)
                
            # Calculation of distances in the area
            H = self.compute_local_transform(ref_line)
            p1_world = self.transform_point(seg_start, H)
            p2_world = self.transform_point(seg_end, H)
            total_distance += np.linalg.norm(p2_world - p1_world)
            if ref_line not in used_refs:
                used_refs.append(ref_line)
                
            last_point = seg_end
            

            cv2.circle(debug_image, 
                    (int(seg_start[0]), int(seg_start[1])), 
                    3, (0, 0, 255), -1)
            cv2.circle(debug_image, 
                    (int(seg_end[0]), int(seg_end[1])), 
                    3, (0, 255, 0), -1)
        
        # Addressing the last possible extra-regional component
        if last_point != point2:
            H = self.compute_local_transform(outer_ref)
            p1_world = self.transform_point(last_point, H)
            p2_world = self.transform_point(point2, H)
            total_distance += np.linalg.norm(p2_world - p1_world)
            if outer_ref not in used_refs:
                used_refs.append(outer_ref)

        # Display debug image
        cv2.imshow('Debug View', debug_image)
        return total_distance, used_refs

    def calculate_point_distance(self, point, ref_line) -> float:
        """Calculate the distance from the point to the reference line area"""
        #  Returns 0 if the point is in the region
        if self.point_in_region(point, ref_line):
            return 0
            
        # Calculate the minimum distance from the point to the four sides of the region
        edges = [
            (ref_line['line1'][0], ref_line['line1'][1]),
            (ref_line['line1'][1], ref_line['line2'][1]),
            (ref_line['line2'][1], ref_line['line2'][0]),
            (ref_line['line2'][0], ref_line['line1'][0])
        ]
        
        min_dist = float('inf')
        for edge_start, edge_end in edges:
            dist = self.point_to_line_segment_distance(point, edge_start, edge_end)
            min_dist = min(min_dist, dist)
            
        return min_dist

    def point_to_line_segment_distance(self, point, line_start, line_end) -> float:
        """Calculate the distance from the point to the line segment"""
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        A = x - x1
        B = y - y1
        C = x2 - x1
        D = y2 - y1

        dot = A * C + B * D
        len_sq = C * C + D * D

        if len_sq == 0:
            return np.sqrt((x - x1)**2 + (y - y1)**2)
            
        param = dot / len_sq
        
        if param < 0:
            return np.sqrt((x - x1)**2 + (y - y1)**2)
        elif param > 1:
            return np.sqrt((x - x2)**2 + (y - y2)**2)
        else:
            x_proj = x1 + param * C
            y_proj = y1 + param * D
            return np.sqrt((x - x_proj)**2 + (y - y_proj)**2)

    def compute_local_transform(self, ref_line):
        """Calculate the local perspective transformation matrix"""
        src_points = np.array([
            ref_line['line1'][0], # Top-left
            ref_line['line1'][1], # Bottom-left
            ref_line['line2'][0], # Top-right
            ref_line['line2'][1] # Bottom-right
        ], dtype=np.float32)
        
        dist = ref_line['distance']
        line1_height = np.linalg.norm(
            np.array(ref_line['line1'][1]) - np.array(ref_line['line1'][0])
        )

        dst_points = np.array([
            [0, 0],      # Top-left
            [0, line1_height * dist / line1_height],   # Bottom-left
            [dist, 0],   # Top-right
            [dist, line1_height * dist / line1_height]  # Bottom-right
        ], dtype=np.float32)
        
        return cv2.getPerspectiveTransform(src_points, dst_points)

    def transform_point(self, point, H):
        """Transform point coordinates using the transformation matrix"""
        p = np.array([[point[0], point[1], 1]], dtype=np.float32).T
        p_transformed = np.dot(H, p)
        p_transformed = p_transformed / p_transformed[2]
        return p_transformed[:2].flatten()

    def validate_reference_lines(self):
        """Verify distance calculations for all reference pairs"""
        print("\nValidating reference lines:")
        for i, ref_line in enumerate(self.calibration['reference_lines']):
            # Calculate the distance between pairs of reference lines
            dist, used_refs = self.calculate_distance(
                ref_line['line1'][0], ref_line['line2'][0])
            
            print(f"Reference pair {i+1}:")
            print(f"  Expected distance: {ref_line['distance']}m")
            print(f"  Measured distance: {dist:.3f}m")
            print(f"  Error: {abs(dist - ref_line['distance']):.3f}m")
            print(f"  Using references: {[ref['distance'] for ref in used_refs]}")

    def draw_reference_lines(self):
        """Drawing reference lines and areas"""

        if self.calibration.get('rail_lines'):
            if self.calibration['rail_lines'].get('left'):
                left = self.calibration['rail_lines']['left']
                cv2.line(self.display_image,
                        tuple(map(int, left[0])),
                        tuple(map(int, left[1])),
                        (255, 0, 0), 2)
            if self.calibration['rail_lines'].get('right'):
                right = self.calibration['rail_lines']['right']
                cv2.line(self.display_image,
                        tuple(map(int, right[0])),
                        tuple(map(int, right[1])),
                        (0, 0, 255), 2)


        for ref_line in self.calibration['reference_lines']:

            if abs(ref_line['distance'] - 0.9) < 0.01:
                color1 = (0, 255, 0)   # 绿色
                color2 = (0, 255, 255) # 黄色
            else:
                color1 = (255, 0, 255) # 紫色
                color2 = (255, 255, 0) # 青色

            region = np.array([
                ref_line['line1'][0],
                ref_line['line1'][1],
                ref_line['line2'][1],
                ref_line['line2'][0]
            ], dtype=np.int32)
            

            overlay = self.display_image.copy()
            cv2.fillPoly(overlay, [region], color1, 8)
            cv2.addWeighted(overlay, 0.2, self.display_image, 0.8, 0, self.display_image)


            cv2.line(self.display_image,
                    tuple(map(int, ref_line['line1'][0])),
                    tuple(map(int, ref_line['line1'][1])),
                    color1, 2)
            cv2.line(self.display_image,
                    tuple(map(int, ref_line['line2'][0])),
                    tuple(map(int, ref_line['line2'][1])),
                    color2, 2)

            mid_point = tuple(map(int, np.mean([
                ref_line['line1'][0],
                ref_line['line2'][0]
            ], axis=0)))
            cv2.putText(self.display_image,
                       f"{ref_line['distance']}m",
                       mid_point,
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (255, 255, 255), 2)

    def mouse_callback(self, event, x, y, flags, param):
        """mouse events"""
        # 将缩放后的坐标转换回原始坐标
        scale = 0.9
        original_x = int(x / scale)
        original_y = int(y / scale)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (original_x, original_y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                display = self.display_image.copy()
                cv2.line(display, self.start_point, (original_x, original_y), (0, 255, 0), 2)
                
                dist, ref_lines = self.calculate_distance(self.start_point, (original_x, original_y))
                mid_point = ((self.start_point[0] + original_x) // 2,
                            (self.start_point[1] + original_y) // 2)
                
                cv2.putText(display,
                        f"Total: {dist:.2f}m",
                        mid_point,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
                
                for i, ref_line in enumerate(ref_lines):
                    info = f"Using {ref_line['distance']}m reference"
                    cv2.putText(display,
                            info,
                            (mid_point[0], mid_point[1] + 20 * (i + 1)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)
                
                # 显示时再次缩放
                height, width = display.shape[:2]
                window_width = int(width * scale)
                window_height = int(height * scale)
                display = cv2.resize(display, (window_width, window_height))
                cv2.imshow('Distance Measurement', display)
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                dist, ref_lines = self.calculate_distance(self.start_point, (original_x, original_y))
                cv2.line(self.display_image, self.start_point, (original_x, original_y),
                        (0, 255, 0), 2)
                
                mid_point = ((self.start_point[0] + original_x) // 2,
                            (self.start_point[1] + original_y) // 2)
                
                cv2.putText(self.display_image,
                        f"Total: {dist:.2f}m",
                        mid_point,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
                
                for i, ref_line in enumerate(ref_lines):
                    info = f"Using {ref_line['distance']}m reference"
                    cv2.putText(self.display_image,
                            info,
                            (mid_point[0], mid_point[1] + 20 * (i + 1)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)

    # def mouse_callback(self, event, x, y, flags, param):
    #     """mouse events"""
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         self.drawing = True
    #         self.start_point = (x, y)
            
    #     elif event == cv2.EVENT_MOUSEMOVE:
    #         if self.drawing:
    #             display = self.display_image.copy()
    #             cv2.line(display, self.start_point, (x, y), (0, 255, 0), 2)
                
    #             # Calculates and displays real-time distance
    #             dist, ref_lines = self.calculate_distance(self.start_point, (x, y))
    #             mid_point = ((self.start_point[0] + x) // 2,
    #                        (self.start_point[1] + y) // 2)
                
    #             # Display Total Distance
    #             cv2.putText(display,
    #                       f"Total: {dist:.2f}m",
    #                       mid_point,
    #                       cv2.FONT_HERSHEY_SIMPLEX,
    #                       0.5, (0, 255, 0), 2)
                
    #             # Displays information on all reference lines used
    #             for i, ref_line in enumerate(ref_lines):
    #                 info = f"Using {ref_line['distance']}m reference"
    #                 cv2.putText(display,
    #                           info,
    #                           (mid_point[0], mid_point[1] + 20 * (i + 1)),
    #                           cv2.FONT_HERSHEY_SIMPLEX,
    #                           0.5, (0, 255, 0), 2)
                
    #             cv2.imshow('Distance Measurement', display)
                
    #     elif event == cv2.EVENT_LBUTTONUP:
    #         if self.drawing:
    #             self.drawing = False
    #             dist, ref_lines = self.calculate_distance(self.start_point, (x, y))
    #             cv2.line(self.display_image, self.start_point, (x, y),
    #                     (0, 255, 0), 2)
                
    #             mid_point = ((self.start_point[0] + x) // 2,
    #                        (self.start_point[1] + y) // 2)
                
    #             # Display Total Distance
    #             cv2.putText(self.display_image,
    #                       f"Total: {dist:.2f}m",
    #                       mid_point,
    #                       cv2.FONT_HERSHEY_SIMPLEX,
    #                       0.5, (0, 255, 0), 2)
                
    #             # Display all reference line
    #             for i, ref_line in enumerate(ref_lines):
    #                 info = f"Using {ref_line['distance']}m reference"
    #                 cv2.putText(self.display_image,
    #                           info,
    #                           (mid_point[0], mid_point[1] + 20 * (i + 1)),
    #                           cv2.FONT_HERSHEY_SIMPLEX,
    #                           0.5, (0, 255, 0), 2)
                
                cv2.imshow('Distance Measurement', self.display_image)

    # def start_measurement(self):
    #     """start interactive measurements"""
    #     cv2.namedWindow('Distance Measurement')
    #     cv2.setMouseCallback('Distance Measurement', self.mouse_callback)

    #     self.draw_reference_lines()
        
    #     print("\nInstructions:")
    #     print("- Click and drag to measure distance")
    #     print("- Press 'r' to reset view")
    #     print("- Press 's' to save current view")  
    #     print("- Press 'q' to quit")
    #     print("\nNote: Multiple reference lines may be used for a single measurement")
        
    #     while True:
    #         cv2.imshow('Distance Measurement', self.display_image)
    #         key = cv2.waitKey(1) & 0xFF
            
    #         if key == ord('r'):
    #             self.display_image = self.image.copy()
    #             self.draw_reference_lines()
    #         elif key == ord('s'):  # 添加保存功能
    #             output_path = f"distance_measurement_{time.strftime('%Y%m%d_%H%M%S')}.png"
    #             cv2.imwrite(output_path, self.display_image)
    #             print(f"Image saved to: {output_path}")
    #         elif key == ord('q'):
    #             break
        
    #     cv2.destroyAllWindows()
    def start_measurement(self):
        """start interactive measurements"""
        scale = 0.9
        height, width = self.image.shape[:2]
        window_width = int(width * scale)
        window_height = int(height * scale)
        
        cv2.namedWindow('Distance Measurement')
        cv2.setMouseCallback('Distance Measurement', self.mouse_callback)
        
        self.draw_reference_lines()
        
        while True:
            # 仅在显示时缩放图像
            display = cv2.resize(self.display_image, (window_width, window_height))
            cv2.imshow('Distance Measurement', display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):
                self.display_image = self.image.copy()
                self.draw_reference_lines()
            elif key == ord('s'):
                output_path = f"distance_measurement_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(output_path, self.display_image)
                print(f"Image saved to: {output_path}")
            elif key == ord('q'):
                break
        
        cv2.destroyAllWindows()

def main():
    # Load the calibration file
    calibration_file = "D:/GitHub/Railway_Safety_system/station_calibration.json"  
    # image_path = "D:/video/validation/00388.png"  # station left
    image_path = "D:/video/validation/00885.png"  # station right
 
    try:

        with open(calibration_file, 'r') as f:
            calibration_data = json.load(f)
        
        # Creating a Distance Mapper
        mapper = MultiReferenceDistanceMapper(calibration_data, image_path)
        
        # Initiate interactive measurements
        mapper.start_measurement()
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()