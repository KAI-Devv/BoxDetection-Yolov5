import os
import random
import shutil
from os import listdir
import json
import sys

# get the path of the directory that contains the images
ROOT = sys.argv[1]
if ROOT[-1] == "/":
    ROOT = ROOT[:-1]

metadata_path = sys.argv[2]

TARGET_PATH = ROOT + "_data"
os.makedirs(TARGET_PATH, exist_ok=True)

filelist = []
for root, dirs, files in os.walk(ROOT):
	for file in files:
        #append the file name to the list
		filelist.append(os.path.join(root,file))


def get_file_name(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

labels = list() # json's
images = list() # images
names_without_ext = list() # names without extension

check1 = list()
check2 = list()
for file in filelist:
    if file.endswith(".json"):
        check1.append(get_file_name(file))
        labels.append(file)
    elif file.endswith(".jpg"):
        check2.append(get_file_name(file))
        images.append(file)
    else:
        continue

if set(check1) != set(check2) or len(check1) != len(check2):
    print("Something is wrong")
    os._exit(0)


os.makedirs(os.path.join(TARGET_PATH, "img"), exist_ok=True)
os.makedirs(os.path.join(TARGET_PATH, "json"), exist_ok=True)


for file in images:
    shutil.copy(file, os.path.join(TARGET_PATH, "img"))

for file in labels:
    shutil.copy(file, os.path.join(TARGET_PATH, "json"))


modes = ["train", "val", "test", "train_json", "val_json", "test_json"]
for mode in modes:
     os.makedirs(os.path.join(TARGET_PATH, mode), exist_ok=True)


# lets make folders in main directory
# os.makedirs(PATH)

files_n = [f for f in listdir(os.path.join(TARGET_PATH, "json"))]
files_without_suffix = [f.split(".")[0] for f in files_n]

splitter = {'train': [], 'val': [], 'test': []}
splitter_img = {'train': [], 'val': [], 'test': []}
splitter_json = {'train': [], 'val': [], 'test': []}


category_info = []
with open(metadata_path, 'r') as curr_file:
    category_info = json.load(curr_file)['categories']
class_num = len(category_info)


obj_normal_list = []
obj_abnormal_list = []

for i in range(0, class_num):
    obj_normal_list.append(0)
    obj_abnormal_list.append(0)

 
for root, dirs, files in os.walk(ROOT):
    for file in files:
        path = os.path.join(root,file)
        if '.json' in path:
            with open(path, 'r') as curr_file:
                curr = json.load(curr_file)
            
            normal_list = []
            abnormal_list = []
            is_valid = True

            for obj in curr['annotations']:
                temp_id = obj['category_id']
                obj_id = -1
                category_supercategory = ''
                category_name = ''
                for category in curr['categories']:
                    if category['id'] == obj['category_id']:
                        temp_id = category['id']
                        category_supercategory = category['supercategory']
                        category_name = category['name']
                        break
                for category in category_info:
                    if category_supercategory == category['supercategory'] and category_name == category['name']:
                        obj_id = category['id']
                        break
                
                if obj_id >= 0:
                    #if obj['status'] == "normal":
                    if obj['status'] == "normal" and obj_id not in normal_list:
                        normal_list.append(obj_id)
                    #if obj['status'] == "abnormal":
                    if obj['status'] == "abnormal" and obj_id not in abnormal_list:
                        abnormal_list.append(obj_id)
                    if obj['status'] != "normal" and obj['status'] != "abnormal":
                        is_valid = False
                        #print(obj)
            
            if is_valid:            
                for value in normal_list:
                    obj_normal_list[value] += 1
                for value in abnormal_list:
                    obj_abnormal_list[value] += 1

for i in range(0, class_num):
    #print(i, obj_class_list[i] + ":", obj_normal_list[i]/minimum_normal_requirement[i]['number'], obj_normal_list[i])
    #print(i, obj_class_list[i] + ":", obj_abnormal_list[i]/minimum_normal_requirement[i]['number'], obj_abnormal_list[i])
    
    normal_ratio = obj_normal_list[i]/category_info[i]['number_normal']
    abnormal_ratio = obj_abnormal_list[i]/category_info[i]['number_abnormal']

    normal_test_ratio = 0
    if obj_normal_list[i] != 0:
        normal_test_ratio = category_info[i]['number_normal'] * 0.1 / obj_normal_list[i]
        if normal_test_ratio < 0.1:
            normal_test_ratio = 0.1
        if normal_test_ratio > 0.5:
            normal_test_ratio = 0.5

    abnormal_test_ratio = 0
    if obj_abnormal_list[i] != 0:
        abnormal_test_ratio = category_info[i]['number_abnormal'] * 0.1 / obj_abnormal_list[i]
        if abnormal_test_ratio < 0.1:
            abnormal_test_ratio = 0.1
        if abnormal_test_ratio > 0.5:
            abnormal_test_ratio = 0.5
    
    print(i, category_info[i]['supercategory'] + "_" + category_info[i]['name'], obj_normal_list[i], obj_abnormal_list[i])
    category_info[i]['normal_ratio'] = normal_test_ratio
    category_info[i]['abnormalratio'] = abnormal_test_ratio



for i in range(len(files_without_suffix)):
    path = TARGET_PATH + '/json/' + files_without_suffix[i] + '.json'
    with open(path, 'r') as curr_file:
        curr = json.load(curr_file)
    list_img = curr['annotations']
    test_ratio = 0.1
    is_valid = True
    if curr['metadata']['width'] != 1920 or curr['metadata']['height'] != 1080:
        is_valid = False
    for obj in curr['annotations']:
        temp_id = obj['category_id']
        obj_id = -1
        category_supercategory = ''
        category_name = ''
        for category in curr['categories']:
            if category['id'] == obj['category_id']:
                temp_id = category['id']
                category_supercategory = category['supercategory']
                category_name = category['name']
                break
        for category in category_info:
            if category_supercategory == category['supercategory'] and category_name == category['name']:
                obj_id = category['id']
                break

        status = obj['status']
        if obj_id >= 0:
            if status == 'normal':
                if category_info[obj_id]['normal_ratio'] > test_ratio:
                    test_ratio = category_info[obj_id]['normal_ratio']
            elif status == 'abnormal':
                if category_info[obj_id]['abnormal_ratio'] > test_ratio:
                    test_ratio = category_info[obj_id]['abnormal_ratio']
            else:
                is_valid = False
        else:
            is_valid = False

    if is_valid:
        p_value = random.random()
        if p_value < test_ratio:
            splitter["test"].append(files_without_suffix[i])
            splitter_img["test"].append(files_without_suffix[i]+".jpg")
            splitter_json["test"].append(files_without_suffix[i]+".json")
        elif p_value >= test_ratio and p_value < 0.5 and p_value < test_ratio + 0.1:
            splitter["val"].append(files_without_suffix[i])
            splitter_img["val"].append(files_without_suffix[i]+".jpg")
            splitter_json["val"].append(files_without_suffix[i]+".json")
        else:
            splitter["train"].append(files_without_suffix[i])
            splitter_img["train"].append(files_without_suffix[i]+".jpg")
            splitter_json["train"].append(files_without_suffix[i]+".json")



test_obj_normal_list = []
test_obj_abnormal_list = []
for i in range(0, class_num):
    test_obj_normal_list.append(0)
    test_obj_abnormal_list.append(0)
for path in splitter_json["test"]:
    path = TARGET_PATH + '/json/' + path
    with open(path, 'r') as curr_file:
        curr = json.load(curr_file)
    list_img = curr['annotations']
    normal_list = []
    abnormal_list = []
    '''
    for obj in list_img:
        #if obj['status'] == "normal":
        if obj['status'] == "normal" and obj['category_id'] not in normal_list:
            normal_list.append(obj['category_id'])
        #if obj['status'] == "abnormal":
        if obj['status'] == "abnormal" and obj['category_id'] not in abnormal_list:
            abnormal_list.append(obj['category_id'])
        if obj['status'] != "normal" and obj['status'] != "abnormal":
            print(obj)
    for value in normal_list:
        test_obj_normal_list[value] += 1
    for value in abnormal_list:
        test_obj_abnormal_list[value] += 1
    '''

    for obj in curr['annotations']:
        temp_id = obj['category_id']
        obj_id = -1
        category_supercategory = ''
        category_name = ''
        for category in curr['categories']:
            if category['id'] == obj['category_id']:
                temp_id = category['id']
                category_supercategory = category['supercategory']
                category_name = category['name']
                break

        for category in category_info:
            if category_supercategory == category['supercategory'] and category_name == category['name']:
                obj_id = category['id']
                break
                
        if obj_id >= 0:
            if obj['status'] == "normal":
            #if obj['status'] == "normal" and obj_id not in normal_list:
                normal_list.append(obj_id)
            if obj['status'] == "abnormal":
            #if obj['status'] == "abnormal" and obj_id not in abnormal_list:
                abnormal_list.append(obj_id)
            if obj['status'] != "normal" and obj['status'] != "abnormal":
                is_valid = False
                #print(obj)
            
    if is_valid:            
        for value in normal_list:
            test_obj_normal_list[value] += 1
        for value in abnormal_list:
            test_obj_abnormal_list[value] += 1

val_obj_normal_list = []
val_obj_abnormal_list = []
for i in range(0, class_num):
    val_obj_normal_list.append(0)
    val_obj_abnormal_list.append(0)
for path in splitter_json["val"]:
    path = TARGET_PATH + '/json/' + path
    with open(path, 'r') as curr_file:
        curr = json.load(curr_file)
    list_img = curr['annotations']
    normal_list = []
    abnormal_list = []
    '''
    for obj in list_img:
        #if obj['status'] == "normal":
        if obj['status'] == "normal" and obj['category_id'] not in normal_list:
            normal_list.append(obj['category_id'])
        #if obj['status'] == "abnormal":
        if obj['status'] == "abnormal" and obj['category_id'] not in abnormal_list:
            abnormal_list.append(obj['category_id'])
        if obj['status'] != "normal" and obj['status'] != "abnormal":
            print(obj)
    for value in normal_list:
        val_obj_normal_list[value] += 1
    for value in abnormal_list:
        val_obj_abnormal_list[value] += 1
    '''
    for obj in curr['annotations']:
        temp_id = obj['category_id']
        obj_id = -1
        category_supercategory = ''
        category_name = ''
        for category in curr['categories']:
            if category['id'] == obj['category_id']:
                temp_id = category['id']
                category_supercategory = category['supercategory']
                category_name = category['name']
                break

        for category in category_info:
            if category_supercategory == category['supercategory'] and category_name == category['name']:
                obj_id = category['id']
                break
                
        if obj_id >= 0:
            if obj['status'] == "normal":
            #if obj['status'] == "normal" and obj_id not in normal_list:
                normal_list.append(obj_id)
            if obj['status'] == "abnormal":
            #if obj['status'] == "abnormal" and obj_id not in abnormal_list:
                abnormal_list.append(obj_id)
            if obj['status'] != "normal" and obj['status'] != "abnormal":
                is_valid = False
                #print(obj)
            
    if is_valid:            
        for value in normal_list:
            val_obj_normal_list[value] += 1
        for value in abnormal_list:
            val_obj_abnormal_list[value] += 1

train_obj_normal_list = []
train_obj_abnormal_list = []
for i in range(0, class_num):
    train_obj_normal_list.append(0)
    train_obj_abnormal_list.append(0)
for path in splitter_json["train"]:
    path = TARGET_PATH + '/json/' + path
    with open(path, 'r') as curr_file:
        curr = json.load(curr_file)
    list_img = curr['annotations']
    normal_list = []
    abnormal_list = []
    '''
    for obj in list_img:
        #if obj['status'] == "normal":
        if obj['status'] == "normal" and obj['category_id'] not in normal_list:
            normal_list.append(obj['category_id'])
        #if obj['status'] == "abnormal":
        if obj['status'] == "abnormal" and obj['category_id'] not in abnormal_list:
            abnormal_list.append(obj['category_id'])
        if obj['status'] != "normal" and obj['status'] != "abnormal":
            print(obj)
    for value in normal_list:
        train_obj_normal_list[value] += 1
    for value in abnormal_list:
        train_obj_abnormal_list[value] += 1
    '''
    for obj in curr['annotations']:
        temp_id = obj['category_id']
        obj_id = -1
        category_supercategory = ''
        category_name = ''
        for category in curr['categories']:
            if category['id'] == obj['category_id']:
                temp_id = category['id']
                category_supercategory = category['supercategory']
                category_name = category['name']
                break

        for category in category_info:
            if category_supercategory == category['supercategory'] and category_name == category['name']:
                obj_id = category['id']
                break
                
        if obj_id >= 0:
            if obj['status'] == "normal":
            #if obj['status'] == "normal" and obj_id not in normal_list:
                normal_list.append(obj_id)
            if obj['status'] == "abnormal":
            #if obj['status'] == "abnormal" and obj_id not in abnormal_list:
                abnormal_list.append(obj_id)
            if obj['status'] != "normal" and obj['status'] != "abnormal":
                is_valid = False
                #print(obj)
            
    if is_valid:            
        for value in normal_list:
            train_obj_normal_list[value] += 1
        for value in abnormal_list:
            train_obj_abnormal_list[value] += 1

for i in range(0, class_num):
    print(i, category_info[i]['number_normal'], category_info[i]['normal_ratio'], category_info[i]['supercategory'] + "_" + category_info[i]['name'], test_obj_normal_list[i], val_obj_normal_list[i], train_obj_normal_list[i])
    #if category_info[i]['number_normal'] * 0.08 > test_obj_normal_list[i]:
    #    print('not_enough') 
    print(i, category_info[i]['number_abnormal'], category_info[i]['abnormal_ratio'], category_info[i]['supercategory'] + "_" + category_info[i]['name'], test_obj_abnormal_list[i], val_obj_abnormal_list[i], train_obj_abnormal_list[i])
    #if category_info[i]['number_abnormal'] * 0.08 > test_obj_abnormal_list[i]:
    #    print('not_enough') 




for file in splitter["train"]:
    shutil.move(TARGET_PATH + "/img/" + file + ".jpg", TARGET_PATH + "/train/")
    shutil.move(TARGET_PATH + "/json/" + file + ".json", TARGET_PATH + "/train_json/")

for file in splitter["val"]:
    shutil.move(TARGET_PATH + "/img/" + file + ".jpg", TARGET_PATH + "/val/")
    shutil.move(TARGET_PATH + "/json/" + file + ".json", TARGET_PATH + "/val_json/")

for file in splitter["test"]:
    shutil.move(TARGET_PATH + "/img/" + file + ".jpg", TARGET_PATH + "/test/")
    shutil.move(TARGET_PATH + "/json/" + file + ".json", TARGET_PATH + "/test_json/")


shutil.rmtree(TARGET_PATH + "/img/")
shutil.rmtree(TARGET_PATH + "/json/")



'''
for i in range(len(files_without_suffix)):
    p_value = random.random()
    if p_value<0.8:
        splitter["train"].append(files_without_suffix[i])
        splitter_img["train"].append(files_without_suffix[i]+".jpg")
        splitter_json["train"].append(files_without_suffix[i]+".json")
    elif p_value<0.9:
        splitter["val"].append(files_without_suffix[i])
        splitter_img["val"].append(files_without_suffix[i]+".jpg")
        splitter_json["val"].append(files_without_suffix[i]+".json")
    else:
        splitter["test"].append(files_without_suffix[i])
        splitter_img["test"].append(files_without_suffix[i]+".jpg")
        splitter_json["test"].append(files_without_suffix[i]+".json")


for file in splitter["train"]:
    shutil.move(TARGET_PATH + "/img/" + file + ".jpg", TARGET_PATH + "/train/")
    shutil.move(TARGET_PATH + "/json/" + file + ".json", TARGET_PATH + "/train/")

for file in splitter["val"]:
    shutil.move(TARGET_PATH + "/img/" + file + ".jpg", TARGET_PATH + "/val/")
    shutil.move(TARGET_PATH + "/json/" + file + ".json", TARGET_PATH + "/val/")

for file in splitter["test"]:
    shutil.move(TARGET_PATH + "/img/" + file + ".jpg", TARGET_PATH + "/test/")
    shutil.move(TARGET_PATH + "/json/" + file + ".json", TARGET_PATH + "/test/")


shutil.rmtree(TARGET_PATH + "/img/")
shutil.rmtree(TARGET_PATH + "/json/")
'''


'''
width = 1920
height = 1080

category = list()
for file in files_for_class:
    with open(os.path.join(class_name_path, file), 'r') as class_curr_file:
        class_curr = json.load(class_curr_file)
    class_categories = class_curr['annotations']
    #print(class_categories, "\n")
    for i in range(len(class_categories)):
        class_id = class_categories[i]['category_id']
        class_status = class_categories[i]['status']
        if class_status != 'normal' and class_status != 'abnormal':
            continue
        class_name = class_curr['categories'][class_id-1]['supercategory'] + '_' + class_curr['categories'][class_id-1]['name'] + '-' + class_status
        category.append(class_name)

category = list(set(category))
num_classes = len(category)

cat = dict()
for i in range(len(category)):
    cat[str(category[i])] = i


#function for json to txt

for mode in ["train", "val", "test"]:
    mypath = TARGET_PATH + "/" + mode + "/"
    mypath_label = TARGET_PATH + "/" + mode + "_ann/"
    files = [f for f in listdir(mypath)]
    files_without_suffix = [f.split(".")[0] for f in files]


    labeling = {}
    for i in range(len(files_without_suffix)):
        labeling[files_without_suffix[i]] = i

    for file in files_without_suffix: 
        wrt = open(mypath + file + ".txt", "w")
        width = width
        height = height
        with open(mypath_label + file + '.json', 'r') as curr_file:
            curr = json.load(curr_file)
        list_img = curr['annotations']
        #print(list_img)
        for i in range(len(list_img)):
            curr_img = list_img[i]
            st = ''
            if curr_img['polygon'] == []: 
                id = curr_img['category_id']
                status = curr_img['status']
                if status != 'normal' and status != 'abnormal':
                    continue
                name = curr['categories'][id-1]['supercategory' ] + '_' + curr['categories'][id-1]['name'] + '-' + status

                if name not in cat:
                    print('ERROR')
                    print(file)
                    print('ERROR')
                
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

            else:
                if False:
                    id = curr_img['category_id']
                    status = curr_img['status']
                    if status != 'normal' and status != 'abnormal':
                        continue
                    name = curr['categories'][id-1]['supercategory'] + '_' + curr['categories'][id-1]['name'] + '-' + status
                
                    if name not in cat:
                        print('ERROR')
                        print(file)
                        print('ERROR')
                    
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
'''
        
