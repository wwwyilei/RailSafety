import cv2
import numpy as np
import json
from typing import List, Dict, Tuple, Any, Optional

class StationCalibrationTool:
    def __init__(self, image_path: str):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from {image_path}")
        self.original = self.image.copy()
        self.display_image = self.image.copy()
        self.points: List[Tuple[int, int]] = []
        self.current_line: List[Tuple[int, int]] = []
        
        self.calibration_data = {
            'reference_lines': [],    # 存储两条已知距离的平行线
            'rail_lines': {          # 铁轨相关的线
                'center': None,      # 中心线
                'left': None,        # 左轨
                'right': None        # 右轨
            },
            'platform_edge': None,    # 站台边缘
            'point_distances': [],    # 存储点之间的已知距离
            'known_distances': {
                'rail_gauge': 1.435   # 标准轨距（米）
            }
        }
        
        self.current_mode = None

    def draw_all_elements(self):
        self.display_image = self.original.copy()
        
        # 绘制当前点
        for point in self.points:
            cv2.circle(self.display_image, point, 3, (0, 255, 0), -1)
        
        # 绘制已保存的参考线
        if isinstance(self.calibration_data['reference_lines'], list):
            for line_data in self.calibration_data['reference_lines']:
                if isinstance(line_data, dict):
                    # 如果是已配对的线
                    cv2.line(self.display_image, line_data['line1'][0], line_data['line1'][1], (0, 255, 0), 2)
                    cv2.line(self.display_image, line_data['line2'][0], line_data['line2'][1], (0, 255, 255), 2)
                    # 显示距离
                    mid_point = tuple(map(int, np.mean([line_data['line1'][0], line_data['line2'][0]], axis=0)))
                    cv2.putText(self.display_image, 
                              f"{line_data['distance']}m",
                              mid_point,
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (255, 255, 255), 2)
                else:
                    # 如果是单条线
                    cv2.line(self.display_image, line_data[0], line_data[1], (0, 255, 0), 2)
        
        # 绘制铁轨线
        if self.calibration_data['rail_lines']['left']:
            cv2.line(self.display_image,
                    tuple(map(int, self.calibration_data['rail_lines']['left'][0])),
                    tuple(map(int, self.calibration_data['rail_lines']['left'][1])),
                    (255, 0, 0), 2)
        if self.calibration_data['rail_lines']['right']:
            cv2.line(self.display_image,
                    tuple(map(int, self.calibration_data['rail_lines']['right'][0])),
                    tuple(map(int, self.calibration_data['rail_lines']['right'][1])),
                    (0, 0, 255), 2)
        if self.calibration_data['rail_lines']['center']:
            cv2.line(self.display_image,
                    tuple(map(int, self.calibration_data['rail_lines']['center'][0])),
                    tuple(map(int, self.calibration_data['rail_lines']['center'][1])),
                    (0, 255, 0), 2)
        
        cv2.imshow('Image', self.display_image)

    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            self.draw_all_elements()
            
            if self.current_mode in ['reference', 'rail', 'platform']:
                if len(self.points) == 2:
                    cv2.line(self.display_image, self.points[0], self.points[1], (0, 255, 0), 2)
                    cv2.imshow('Image', self.display_image)
                    
                    self.current_line = self.points.copy()
                    
                    if self.current_mode == 'reference':
                        self.handle_reference_line()
                    elif self.current_mode == 'rail':
                        self.handle_rail_line()
                    elif self.current_mode == 'platform':
                        self.handle_platform_edge()
                    
                    self.points = []
            
            elif self.current_mode == 'point_distance':
                if len(self.points) == 2:
                    cv2.line(self.display_image, self.points[0], self.points[1], (0, 255, 0), 2)
                    cv2.imshow('Image', self.display_image)
                    self.handle_point_distance()
                    self.points = []

    def handle_reference_line(self):
        if len(self.calibration_data['reference_lines']) == 0 or isinstance(self.calibration_data['reference_lines'][-1], dict):
            self.calibration_data['reference_lines'].append(self.current_line)
            print("First line marked. Please mark the second line.")
        else:
            self.calibration_data['reference_lines'].append(self.current_line)
            
            try:
                distance = float(input("Enter the real distance between these two lines (meters): "))
                description = input("Enter description for these reference lines: ")
                
                line_pair = {
                    'line1': self.calibration_data['reference_lines'][-2],
                    'line2': self.calibration_data['reference_lines'][-1],
                    'distance': distance,
                    'description': description
                }
                self.calibration_data['reference_lines'][-2:] = [line_pair]
                print("Reference line pair saved. You can mark another pair or switch mode.")
                
                self.draw_all_elements()
                
            except ValueError:
                print("Invalid input. Please enter a valid number for distance.")
                self.calibration_data['reference_lines'].pop()

    def handle_rail_line(self):
        print("\nSelect rail line type:")
        print("1 - Center line")
        print("2 - Left rail")
        print("3 - Right rail")
        choice = input("Enter choice (1-3): ")
        
        if choice == '1':
            self.calibration_data['rail_lines']['center'] = self.current_line
            print("Center line marked.")
        elif choice == '2':
            self.calibration_data['rail_lines']['left'] = self.current_line
            print("Left rail marked.")
        elif choice == '3':
            self.calibration_data['rail_lines']['right'] = self.current_line
            print("Right rail marked.")
        
        self.draw_all_elements()

    def handle_platform_edge(self):
        self.calibration_data['platform_edge'] = self.current_line
        print("Platform edge marked.")
        self.draw_all_elements()

    def handle_point_distance(self):
        try:
            distance = float(input("Enter the real distance between these points (meters): "))
            description = input("Enter description for this distance: ")
            
            self.calibration_data['point_distances'].append({
                'point1': self.points[0],
                'point2': self.points[1],
                'distance': distance,
                'description': description
            })
            print("Point distance marked.")
            self.draw_all_elements()
            
        except ValueError:
            print("Invalid input. Please enter a valid number for distance.")

    def reset_current(self):
        self.display_image = self.original.copy()
        self.points = []
        self.current_line = []
        cv2.imshow('Image', self.display_image)
        print("Current marking reset.")

    def save_calibration(self):
        try:
            with open('station_calibration.json', 'w') as f:
                json.dump(self.calibration_data, f, indent=4)
            print("Calibration data saved successfully!")
        except Exception as e:
            print(f"Error saving calibration data: {str(e)}")

    def start_calibration(self):
        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', self.click_event)
        
        print("\nCalibration Tool Controls:")
        print("1 - Mark reference line pair (with known distance)")
        print("2 - Mark rail lines")
        print("3 - Mark platform edge")
        print("4 - Mark point distances")
        print("r - Reset current marking")
        print("s - Save calibration data")
        print("q - Quit")
        
        cv2.imshow('Image', self.display_image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('1'):
                self.current_mode = 'reference'
                self.points = []
                print("Mark reference lines (click two points for each line)")
            elif key == ord('2'):
                self.current_mode = 'rail'
                self.points = []
                print("Mark rail line (click two points)")
            elif key == ord('3'):
                self.current_mode = 'platform'
                self.points = []
                print("Mark platform edge (click two points)")
            elif key == ord('4'):
                self.current_mode = 'point_distance'
                self.points = []
                print("Mark two points to measure distance")
            elif key == ord('r'):
                self.reset_current()
            elif key == ord('s'):
                self.save_calibration()
            elif key == ord('q'):
                break
        
        cv2.destroyAllWindows()

def main():
    image_path = "D:/video/validation/00500.png"  # 修改为你的图片路径
    
    # 创建标定工具并启动
    calibration_tool = StationCalibrationTool(image_path)
    calibration_tool.start_calibration()

if __name__ == "__main__":
    main()