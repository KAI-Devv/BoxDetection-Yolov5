"""
Usage - formats:
$ for railway dataset
$ python3 validate-railway-dataset.py --data datasets/railway --data_type railway --weights railway_yolo.pt --task test
$ for catenary dataset
$ python3 validate-railway-dataset.py --data datasets/catenary --data_type catenary --weights catenary_yolo.pt --task test
"""
import pandas as pd
import argparse
import json
import os
import sys
from pathlib import Path
# import opencv
import cv2

import numpy as np
import torch
from tqdm import tqdm
from os import listdir

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, Profile, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, smart_inference_mode
import datetime


# lets make one with milliseconds as well
def timeStamped(fmt='[%Y-%m-%d_%H:%M:%S:%f]'):
    return datetime.datetime.now().strftime(fmt)


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a', encoding='utf-8') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5)})


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


@smart_inference_mode()
def run(
        data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.45,  # confidence threshold
        iou_thres=0.45,  # NMS IoU threshold
        max_det=1000,  # maximum detections per image
        task='test',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        data_type='railway'
):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

        # Data
        data = check_dataset(data)  # check

    # Configure
    model.eval()
    cuda = device.type != 'cpu'
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith(f'coco{os.sep}val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        if pt and not single_cls:  # check --weights are trained on --data
            ncm = model.model.nc
            assert ncm == nc, f'{weights} ({ncm} classes) trained on different --data than what you passed ({nc} ' \
                              f'classes). Pass correct combination of --weights and --data that are trained together.'
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        pad = 0.0 if task in ('speed', 'benchmark') else 0.5
        rect = False if task == 'benchmark' else pt  # square inference for benchmarks
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task],
                                       imgsz,
                                       batch_size,
                                       stride,
                                       single_cls,
                                       pad=pad,
                                       rect=rect,
                                       workers=workers,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc, conf = conf_thres, iou_thres = iou_thres)
    names = model.names if hasattr(model, 'names') else model.module.names  # get class names
    list_of_names = [names[int(i)] for i in range(len(names))]
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile(), Profile(), Profile()  # profiling times
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    callbacks.run('on_val_start')
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar

    result_path = 'result_' + data_type + '.txt'
    with open(result_path, 'w', encoding='utf-8') as f:
        start_str = str(timeStamped()) + ' Inference Started!\n'
        f.write(start_str)
    
    f = open(result_path, 'a', encoding='utf-8')
    nd_str = str(timeStamped())
    if data_type == 'railway':
        nd_str += ' python3 validate-railway-dataset.py --data datasets/railway --data_type railway --weights railway_yolo.pt --task test\n'
    else:
        nd_str += ' python3 validate-railway-dataset.py --data datasets/catenary --data_type catenary --weights catenary_yolo.pt --task test\n'
    f.write(nd_str)
    total_ground_truth = 0
    total_predicted = 0
    avg_fps = 0
    total = 0
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        total += 1
        main_str = ''
        line_for_gt = ''
        total_ground_truth += len(targets)
        for trg in range(len(targets)):
            line_for_gt += '(' + str(names[int(targets[trg][1])]) + ' bbox-' + str(targets[trg][2:]) + '); '

        callbacks.run('on_val_batch_start')
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        # main_str = str(timeStamped()) + ' ' + str(paths) + ': Detections --> '

        with dt[1]:
            preds_main, train_out = model(im) if compute_loss else (model(im, augment=augment), None)

        # Loss
        if compute_loss:
            loss += compute_loss(train_out, targets)[1]  # box, obj, cls

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        with dt[2]:
            preds = non_max_suppression(preds_main,
                                        conf_thres,
                                        iou_thres,
                                        labels=lb,
                                        multi_label=True,
                                        agnostic=single_cls,
                                        max_det=max_det)

        pred_like_detect = non_max_suppression(preds_main, 
                                               conf_thres=0.45, 
                                               iou_thres=0.45, 
                                               classes=None, 
                                               agnostic=False, 
                                               max_det=1000)

        absolute_path = str(Path(paths[0]))
        current_path = os.getcwd()
        relative_path = os.path.relpath(absolute_path, current_path)
        #print('batch_', batch_i)
        main_str = str(timeStamped()) + ' ' + str(relative_path) + '\n' + 'Detections --> '
        for i, detections_det in enumerate(pred_like_detect):  # per image
            im0 = cv2.imread(Path(paths[0]))
            line = ''
            total_predicted += len(detections_det)
            if len(detections_det):
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                detections_det[:, :4] = scale_coords(im.shape[2:], detections_det[:, :4], im0.shape).round()
                # print(len(detections_det))
                # Write results
                for *xyxy, conf, cls in reversed(detections_det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line += '(' + str(names[int(cls)]) + ' bbox-' + str(xywh) + '); ' 
                line += '\n'
            else:
                line = 'No Detections\n'
            main_str += line 


        # Metrics

        for si, pred in enumerate(preds):
            main_str += 'Ground Truth --> '
            main_str += line_for_gt + '\n'

            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            # print(path)
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue
    
            # Print results

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            
            scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred
            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run('on_val_image_end', pred, predn, path, names, im[si])


            
        # Plot images
        if plots and batch_i < 15:
            plot_images(im, targets, paths, save_dir / f'test_batch{batch_i}_labels.jpg', names)  # labels
            plot_images(im, output_to_target(preds), paths, save_dir / f'test_batch{batch_i}_pred.jpg', names)  # pred

        callbacks.run('on_val_batch_end', batch_i, im, targets, paths, shapes, preds)
        
        t = tuple(x.t / seen * 1E3 for x in dt)
        time_inf = t[1]
        fps = 1000 / time_inf
        avg_fps += fps

        f.write(main_str + f'Time: {time_inf} ms  FPS: {fps} fps \n\n')

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)     
        
        # lets copy ap and store without reference
        ap_copy = ap.copy()
        #print('ap:', ap_copy)
        #print('ap_class:', ap_class)
        # column_names = list_of_names
        # row_names = ["mAP@0.50", "mAP@0.55", "mAP@0.60", "mAP@0.65", "mAP@0.70", "mAP@0.75", "mAP@0.80", "mAP@0.85", "mAP@0.90", "mAP@0.95"]

        # df = pd.DataFrame(ap.T, columns = column_names, index = row_names)
        # print()
        # print()
        # with open('mycsvfile.csv', 'a') as f:  # if you want to append to the file
        #     df.to_csv(f)
        # print()
        # print()
        # print(df)  # class-wise
        # print()
        # print(df.mean(axis=1))  # mAP per IoU
        # print()
        # print()
        
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    # Print results
    pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    if nt.sum() == 0:
        LOGGER.warning(f'WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels')

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end', nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)




    category_info = []
    if data_type == 'railway':
        with open('railway_metadata.json', 'r', encoding='utf-8') as curr_file:
            category_info = json.load(curr_file)['categories']
    else:
        with open('catenary_metadata.json', 'r', encoding='utf-8') as curr_file:
            category_info = json.load(curr_file)['categories']


    TP, FP, FN, TN = confusion_matrix.get_metrics()
    # lets make those np arrays into lists
    TP = TP.tolist()
    FP = FP.tolist()
    FN = FN.tolist()
    TN = TN.tolist()

    new_time = str(timeStamped()) + '\n'

    
    
    for class_name in list_of_names:
        class_index_current = list_of_names.index(class_name)
        tp_current = TP[class_index_current]
        fp_current = FP[class_index_current]
        fn_current = FN[class_index_current]
        tn_current = TN[class_index_current]
        # precision = tp / (tp + fp)
        # recall = tp / (tp + fn)
        # f1 = 2 * ((precision * recall) / (precision + recall))
        current_class_total_predictions = tp_current + fp_current
        current_class_total_ground_truth = tp_current + fn_current

        
        
        if category_info[int(class_index_current/2)]['input_type'] == 'box':
            new_time +=  class_name + ': [TP - ' + str(tp_current) + ', FP - ' + str(fp_current) + ', FN - ' + str(fn_current) + ']\n'
            new_time += 'Total number of Ground Truth objects: ' + str(current_class_total_ground_truth) + '\n'
            new_time += 'Total number of Predicted objects: ' + str(current_class_total_predictions) + '\n'

    
    # new_time += 'Total number of Ground Truth objects: ' + str(total_ground_truth) + '\n'
    # new_time += 'Total number of Predicted objects: ' + str(total_predicted) + '\n'

    f.write(new_time + '\n')

    # lets now give mAP values per class

    new_time1 = str(timeStamped()) + '\n'

    #for class_name in list_of_names:
    #    class_index_current = list_of_names.index(class_name)
    #    ap_current_75 = ap_copy[class_index_current, 5]
    #    ap_avg = ap_copy[class_index_current, :].mean()
    #    new_time1 +=  class_name + ': [AP@.75 - ' + str(format(float(ap_current_75), '.4f')) + ', AP@.50:.95 - ' + str(format(float(ap_avg), '.4f')) + ']\n'

    map_overall = 0
    map_overall_75 = 0
    number = 0
    for i in range(0, len(ap_class)):
        
        if category_info[int(ap_class[i]/2)]['input_type'] == 'box':
            ap_current_75 = ap_copy[i, 5]
            ap_avg = ap_copy[i, :].mean()
            new_time1 +=  list_of_names[ap_class[i]] + ': [AP@.75 - ' + str(format(float(ap_current_75), '.4f')) + ', AP@.50:.95 - ' + str(format(float(ap_avg), '.4f')) + ']\n'
            map_overall_75 += ap_current_75
            map_overall += ap_avg
            number += 1
    
    map_overall /= number
    map_overall_75 /= number
    avg_fps /= total
    new_time1 += 'Total through mAP all classes: mAP@.75 - ' + str(format(float(map_overall_75), '.4f')) + '\n'
    new_time1 += 'Average FPS: ' + str(format(float(avg_fps), '.4f')) + '\n'

    f.write(new_time1 + '\n')

    f.close()





    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w', encoding='utf-8') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements('pycocotools')
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def convert_to_yolov5_format(root_folder, dataset_name):
    width=1920
    height=1080
    
    category_info = []
    category = []
    if dataset_name == 'railway':
        with open('railway_metadata.json', 'r', encoding='utf-8') as curr_file:
            category_info = json.load(curr_file)['categories']
    else:
        with open('catenary_metadata.json', 'r', encoding='utf-8') as curr_file:
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

    with open(os.path.join(root_folder, "data.yaml"), 'w', encoding='utf-8') as f:
        f.write("path: " + root_folder)
        f.write("\n\n")
        f.write("train: ")
        f.write("\n")
        f.write("val: ")
        f.write("\n")
        f.write("test: test")
        f.write("\n\n")
        f.write("nc: " + str(num_classes))
        f.write("\n")
        f.write("names: " + str(category))

        f.close()


    #write_txt file for yolov5
    for mode in ["test"]:
        mypath = root_folder + "/" + mode + "/"
        mypath_label = root_folder + "/" + mode + "_json/"
        output_label = root_folder + "/" + mode + "/"
        os.makedirs(output_label, exist_ok=True)
        files = [f for f in listdir(mypath)]
        files_without_suffix = [f.split(".")[0] for f in files]


        labeling = {}
        for i in range(len(files_without_suffix)):
            labeling[files_without_suffix[i]] = i

        for file in files_without_suffix: 
            wrt = open(output_label + file + ".txt", "w", encoding='utf-8')
            width = width
            height = height
            with open(mypath_label + file + '.json', 'r', encoding='utf-8') as curr_file:
                curr = json.load(curr_file)
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



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='confidence threshold') # 0.001 for val.py and 0.25 for detect.py
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold') # 0.6 for val.py and 0.45 for detect.py
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image') # 300 for val.py and 1000 for detect.py
    parser.add_argument('--task', default='test', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--data_type', type=str, default='railway', help='dataset.yaml path')
    opt = parser.parse_args()
    
    target_folder = opt.data
    if target_folder == "/":
        target_folder = target_folder[:-1]
    convert_to_yolov5_format(target_folder, opt.data_type)
    opt.data = check_yaml(target_folder + "/data.yaml")  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f'WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results')
        if opt.save_hybrid:
            LOGGER.info('WARNING ⚠️ --save-hybrid will return high mAP from hybrid labels, not from predictions alone')
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = True  # FP16 for fastest results
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt='%10.4g')  # save
            os.system('zip -r study.zip study_*.txt')
            plot_val_study(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
