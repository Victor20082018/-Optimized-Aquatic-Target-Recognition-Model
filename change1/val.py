import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import cv2
import os

def save_detection_results(model, image_folder, project_path, name):
    # 创建保存目录
    save_dir = os.path.join(project_path, name)
    os.makedirs(save_dir, exist_ok=True)

    # 获取文件夹中的所有图像路径
    image_paths = []
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_paths.append(os.path.join(image_folder, filename))

    # 遍历每张图像进行预测并保存结果
    for img_path in image_paths:
        # 加载图像
        image = cv2.imread(img_path)
        if image is None:
            print(f"Could not load image {img_path}")
            continue

        # 进行预测
        results = model.predict(img_path, imgsz=640, conf=0.25, iou=0.45)

        # 绘制检测框
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = box
                label = f"{model.names[cls_id]} {conf:.2f}"
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 保存图像
        save_path = os.path.join(save_dir, os.path.basename(img_path))
        cv2.imwrite(save_path, image)
        print(f"Saved detection result to {save_path}")

if __name__ == '__main__':
    model = YOLO('D:\\code\\1000_bear\\change1\\runs\\train\\exp\\weights\\best.pt')
    
    # 进行模型验证
    metrics = model.val(
        data='data.yaml',
        split='test',
        imgsz=640,
        batch=4,
        project='runs/test',
        name='exp'
    )
    
    # 设置图像文件夹路径
    image_folder = 'D:\\code\\1000_bear\\Bebidas_varias.v1i.yolov8\\test\\images'  # 替换为你的图像文件夹路径
    
    # 保存检测结果
    save_detection_results(model, image_folder, 'runs/test', 'exp')