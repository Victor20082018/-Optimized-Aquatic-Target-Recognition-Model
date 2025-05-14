from ultralytics import YOLO

# Load the best model
best_model = YOLO('../runs/train/water_surface_detection/weights/best.pt')

# Evaluate the model on the validation dataset
metrics = best_model.val(data='../datasets/water_surface/water_surface.yaml', conf=0.5, iou=0.45)
print(metrics)