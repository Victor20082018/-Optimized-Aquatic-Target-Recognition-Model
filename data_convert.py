import os
import cv2

if __name__ == '__main__':
    dataDir = r"D:\code\800_water_defect\datasets\water_surface\val\images"
    saveDir = r"D:\code\800_water_defect\datasets\water_surface\val\images1"

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    for pic in os.listdir(dataDir):
        old_path = os.path.join(dataDir, pic)
        new_img = cv2.imread(old_path)
        # print("new_img", new_img)
        new_path = os.path.join(saveDir, pic)
        cv2.imwrite(new_path, new_img)

    print("图片转换成功！")
