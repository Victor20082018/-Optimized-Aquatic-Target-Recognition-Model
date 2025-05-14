from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
if __name__ == '__main__':
    # 加载模型
    model = YOLO("./yolov8n.yaml")
    # 训练模型
    results = model.train(data="./data.yaml",  
                
                          epochs=100,
                          batch=16,
                          project='water_surface_pro',
                          patience=30,
                          name='water_surface-yolov8n',
                          amp=False)
