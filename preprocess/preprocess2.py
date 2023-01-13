import os
import random
import shutil
from os import listdir
import json
import sys
import cv2

# get the path of the directory that contains the images
ROOT = sys.argv[1]
if ROOT[-1] == "/":
    ROOT = ROOT[:-1]
metadata_path = sys.argv[2]

TARGET_PATH = ROOT


width = 1920
height = 1080

category_info = []
category = []
with open(metadata_path, 'r') as curr_file:
    category_info = json.load(curr_file)['categories']

for obj in category_info:
    class_name = obj['supercategory'] + '_' + obj['name']
    category.append(class_name + '-normal')
    category.append(class_name + '-abnormal')

print(category)
#category = list(set(category))
num_classes = len(category)


cat = dict()
for i in range(len(category)):
    cat[str(category[i])] = i

print(cat)

with open(os.path.join(TARGET_PATH, "data.yaml"), 'w') as f:
    f.write("path: " + TARGET_PATH)
    f.write("\n\n")
    f.write("train: train")
    f.write("\n")
    f.write("val: val")
    f.write("\n")
    f.write("test: test")
    f.write("\n\n")
    f.write("nc: " + str(num_classes))
    f.write("\n")
    f.write("names: " + str(category))

    f.close()


#write_txt file for yolov5
for mode in ["train", "val", "test"]:
    mypath = TARGET_PATH + "/" + mode + "/"
    mypath_label = TARGET_PATH + "/" + mode + "_json/"
    output_label = TARGET_PATH + "/" + mode + "/"
    os.makedirs(output_label, exist_ok=True)
    files = [f for f in listdir(mypath)]
    files_without_suffix = [f.split(".")[0] for f in files]


    labeling = {}
    for i in range(len(files_without_suffix)):
        labeling[files_without_suffix[i]] = i

    for file in files_without_suffix: 
        wrt = open(output_label + file + ".txt", "w")
        
        #width = width
        #height = height
        with open(mypath_label + file + '.json', 'r') as curr_file:
            curr = json.load(curr_file)
        width = curr['metadata']['width']
        height = curr['metadata']['height']
        if width != 1920 or height != 1080:
            print(width, height)
        list_img = curr['annotations']
        for i in range(len(list_img)):
            curr_img = list_img[i]
            st = ''
            if curr_img['polygon'] == []: 
                id = curr_img['category_id']
                status = curr_img['status']
                if status != 'normal' and status != 'abnormal':
                    continue
                
                name = ''
                for x in curr['categories']:
                    if x['id'] == id:
                        name = x['supercategory'] + '_' + x['name'] + '-' + status
                        break

                if name not in cat:
                    print('ERROR')
                    print(file)
                    print('ERROR')
                    continue
                
                category_id = cat[name]
                curr_bbox = curr_img['bbox']
                bbox = [curr_bbox[0], curr_bbox[1], curr_bbox[2], curr_bbox[3]]

                bbox[0] = max(bbox[0], 0)
                bbox[0] = min(bbox[0], width)
                bbox[1] = max(bbox[1], 0)
                bbox[1] = min(bbox[1], height)
                bbox[2] = max(bbox[2], 0)
                bbox[2] = min(bbox[2], width)
                bbox[3] = max(bbox[3], 0)
                bbox[3] = min(bbox[3], height)

                center_x = (bbox[0] + bbox[2]/2)/width
                center_y = (bbox[1] + bbox[3]/2)/height
                w = bbox[2]/width
                h = bbox[3]/height

                st = str(category_id) + ' ' + str(center_x) + ' ' + str(center_y) + ' ' + str(w) + ' ' + str(h) + '\n'
                wrt.write(st)
            
            else:
                if False:
                    id = curr_img['category_id']
                    status = curr_img['status']
                    if status != 'normal' and status != 'abnormal':
                        continue
                
                    if name not in cat:
                        print('ERROR')
                        print(file)
                        print('ERROR')
                        continue
                    
                    name = ''
                    for x in curr['categories']:
                        if x['id'] == id:
                            name = x['supercategory'] + '_' + x['name'] + '-' + status
                            break

                    category_id = cat[name]

                    curr_segm = curr_img["polygon"]
                    
                    n = len(curr_segm)
                    coor_x = []
                    coor_y = []
                    for i in range(n):
                        if i % 2 == 0:
                            coor_x.append(curr_segm[i])
                        else:
                            coor_y.append(curr_segm[i])
                    
                    x_min = max(min(coor_x), 0)
                    x_max = min(max(coor_x), width)
                    y_min = max(min(coor_y), 0)
                    y_max = min(max(coor_y), height)

                    center_x = (x_min+x_max)/(2*width)
                    center_y = (y_min+y_max)/(2*height)
                    w = (x_max-x_min)/width
                    h = (y_max-y_min)/height

                    st = str(category_id) + ' ' + str(center_x) + ' ' + str(center_y) + ' ' + str(w) + ' ' + str(h) + '\n'
                    wrt.write(st)

        
