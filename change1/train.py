import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import os
# yolov8n-MultiSEAM 0.913
os.environ['KMP_DUPLICATE_LIB_OK']='True'
if __name__ == '__main__':
    model = YOLO('D:\\code\\800_water_defect\\change1\\ultralytics\\cfg\\models\Add\\yolov8n-C2f-RFAConv.yaml')

    model.train(data='./data.yaml',
                cache=False,
          
                epochs=100,
                batch=16,
                close_mosaic=10,
                workers=0,
                device='0',
                optimizer='SGD', # using SGD
                mixup = 0.5,
      
                amp=True, 
                project='runs/water_sur',
                name='water_sur',
                )