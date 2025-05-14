import os
import cv2

if __name__ == '__main__':
    dataDir = r"D:\code\1000v5_v8n\yolo_data\images\test"
    saveDir = r"D:\code\1000v5_v8n\yolo_data\images\test_"

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    for pic in os.listdir(dataDir):
        old_path = os.path.join(dataDir, pic)
        new_img = cv2.imread(old_path)
        # print("new_img", new_img)
        new_path = os.path.join(saveDir, pic)
        cv2.imwrite(new_path, new_img)

    print("图片转换成功！")
