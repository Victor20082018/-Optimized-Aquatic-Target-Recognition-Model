import sys
import cv2
import os
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from ultralytics import YOLO

class DetectionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLOv8 的水上目标检测系统")
        self.setGeometry(100, 100, 1000, 800)
        
        # 创建主窗口部件和布局
        main_widget = QWidget()
        self.main_layout = QVBoxLayout(main_widget)
        
        # 创建图像显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        
        # 创建信息显示区域
        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignLeft)
        self.info_label.setWordWrap(True)
        self.info_label.setMinimumHeight(100)
        
        # 创建按钮区域
        button_layout = QHBoxLayout()
        self.load_button = QPushButton("加载图像")
        self.detect_button = QPushButton("开始检测")
        self.detect_button.setEnabled(False)
        
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.detect_button)
        
        # 添加到主布局
        self.main_layout.addWidget(self.image_label)
        self.main_layout.addWidget(self.info_label)
        self.main_layout.addLayout(button_layout)
        
        self.setCentralWidget(main_widget)
        
        # 连接按钮信号
        self.load_button.clicked.connect(self.load_image)
        self.detect_button.clicked.connect(self.detect_objects)
        
        # 初始化模型
        self.model = YOLO('/Users/mima0000/Downloads/800_water_defect/water_surface_pro/water_surface-yolov8n/weights/best.pt')
        
        # 类别名称
        self.class_names = ['fanchuan', 'person', 'otherboat', 'bird', 'youlun', 'warship', 'yacht', 'airplane', 'Cargoship', 'fish']
        
        # 存储加载的图像
        self.current_image = None
        self.current_image_path = None

    def load_image(self):
        """加载图像"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像文件", "", 
            "图像文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*.*)", 
            options=options
        )
        
        if file_path:
            self.current_image_path = file_path
            self.current_image = cv2.imread(file_path)
            if self.current_image is not None:
                # 转换为 QImage 并显示
                q_image = self.convert_cv_to_qimage(self.current_image)
                self.image_label.setPixmap(QPixmap.fromImage(q_image))
                self.detect_button.setEnabled(True)
                self.info_label.setText(f"加载图像: {file_path}")
            else:
                print(f"无法加载图像: {file_path}")

    def detect_objects(self):
        """进行目标检测并在图像上绘制结果"""
        if self.current_image is None:
            return
        
        # 进行预测 224*224
        results = self.model.predict(self.current_image_path, imgsz=640, conf=0.25, iou=0.45)
        
        # 提取推理时间
        inference_time = results[0].speed['inference']
        
        # 绘制检测框
        image_with_boxes = self.current_image.copy()
        detection_info = []  # 用于存储检测信息
        
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = box
                label = f"{self.class_names[cls_id]} {conf:.2f}"
                cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # 保存检测信息
                detection_info.append(f"类别: {self.class_names[cls_id]}, 置信度: {conf:.2f}, 位置: ({x1}, {y1}, {x2}, {y2})")
        
        # 显示带有检测框的图像
        q_image = self.convert_cv_to_qimage(image_with_boxes)
        self.image_label.setPixmap(QPixmap.fromImage(q_image))
        
        # 更新信息标签
        info_text = f"推理时间: {inference_time:.2f}毫秒\n"
        info_text += "检测结果:\n"
        for info in detection_info:
            info_text += f"{info}\n"
        self.info_label.setText(info_text)

    def convert_cv_to_qimage(self, cv_image):
        """将 OpenCV 图像转换为 QImage"""
        height, width, channel = cv_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return q_image.rgbSwapped()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DetectionWindow()
    window.show()
    sys.exit(app.exec_())