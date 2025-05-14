import os
from pyecharts import Bar
import os.path
import xml.dom.minidom
import xml.etree.cElementTree as et
from scipy.ndimage import measurements
import glob
import math
import numpy as np

path='F:\YOLO V4\yolov4-pytorch-master\VOCdevkit\VOC2007\Annotations'
files=os.listdir(path)
s=[]
ratio_list=[]
little_object=0
big_object=0
medium_object=0
number=0
path_file_number=glob.glob(pathname='F:\YOLO V4\yolov4-pytorch-master\VOCdevkit\VOC2007\Annotations\*.xml') #获取当前文件夹下个数

square0_list = []
square1_list = []
# =============================================================================
# extensional filename
# =============================================================================
def file_extension(path):
    return os.path.splitext(path)[1]

for xmlFile in files:
    if not os.path.isdir(xmlFile):
        if file_extension(xmlFile) == '.xml':
            #print(os.path.join(path,xmlFile))
            tree=et.parse(os.path.join(path,xmlFile))
            root=tree.getroot()
            filename=root.find('filename').text
#            print("filename is", path + '/' + xmlFile)
            for Object in root.findall('object'):
                 name=Object.find('name').text
               #  print(len(name))
                 if name == 'other boat':
                     number=number+1
                 print(xmlFile)
        print(number)
        ''' bndbox=Object.find('bndbox')
                xmin=bndbox.find('xmin').text
                ymin=bndbox.find('ymin').text
                xmax=bndbox.find('xmax').text
                ymax=bndbox.find('ymax').text
                square0 = (int(ymax)-int(ymin)) * (int(xmax)-int(xmin))'''
