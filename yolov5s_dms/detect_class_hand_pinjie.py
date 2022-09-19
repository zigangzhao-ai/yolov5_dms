import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import copy
import torch.backends.cudnn as cudnn
from numpy import random
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import(
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized
import torchvision.models as models
from models.resnet18_pinjie import resnet18
from models.resnet18_hand import resnet18_new
import torchvision.transforms as transforms


src_img_dir = "data_check/jiance/mark"
src_img_dir1 = "data_check/jiance/origin"
src_pinjie_dir1 = "data_check/pinjie/origin"
src_pinjie_dir = "data_check/pinjie/mark"
src_hand_dir = "data_check/hand/mark"
src_hand_dir1 = "data_check/hand/origin"

base = [src_img_dir, src_img_dir1, src_pinjie_dir1, src_pinjie_dir, src_hand_dir, src_hand_dir1]
for path in base:
    if not os.path.exists(path):
      os.makedirs(path)

label_names = ["fenxin", "normal", "smoke", "tired"]
label1_names = ["call", "normal", "smoke"]

mean, std = [0.4734, 0.4734, 0.4734], [0.2009, 0.2009, 0.2009]

def get_test_transform():
    return transforms.Compose([
        transforms.Resize([224,448]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def get_test_transform1():
    return transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def getmax(a):
    """
     input: a = [['mouse',0.9, 510, 327, 604, 386], ['eye',0.6, 456, 200, 662, 298], ['mouse',0.8, 510, 327, 604, 386]]
     return: [['mouse',0.9, 510, 327, 604, 386], ['eye',0.6, 456, 200, 662, 298]]
    """
    if len(a) == 0:
        return a
    a = sorted(a)
    key1 = []
    key2 = []
    ret = []
    for i, x in enumerate(a):
        if x[0] not in key1:
            key1.append(x[0])
            key2.append(x[1])
            ret.append(x)
        else:
            if x[1] >= key2[-1]:        
                ret.pop(key2.index(key2[-1]))
                ret.append(x)
    return ret

def detect(save_img=True):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')
    print(webcam)

    #Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):  ## can conference
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    # print(model)
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    #print(imgsz)

    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None

    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    #print(img)
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    cnt = 0

    net = resnet18()
    net.fc = torch.nn.Linear(512, 4)           
    #net.load_state_dict(torch.load(opt.weight_pinjie))
    net.load_state_dict(torch.load(opt.weight_pinjie, map_location=device))
    net.eval()
    print('..... Finished loading model! ......')

    net1 = resnet18_new()
    net1.fc = torch.nn.Linear(512, 3)           
    #net.load_state_dict(torch.load(opt.weight_pinjie))
    net1.load_state_dict(torch.load(opt.weight_hand, map_location=device))
    net1.eval()
    print('..... Finished loading model! ......')

    for path, img, im0s, vid_cap in dataset:
        #print(img.shape, im0s.shape)
        cnt += 1
        im1 = im0s
        cv2.imwrite(src_img_dir1+'/'+str(cnt) + '.jpg', im1)
        img = torch.from_numpy(img).to(device)
        #print("==========")
        #print(img.shape[0].shape[1])
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        #print(pred[0].shape[1])

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        #print(pred)
        t2 = time_synchronized()
        # print("====================")
        # print(pred, type(pred))
        # print("====================")

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            img1 = copy.deepcopy(im0)  ##深拷贝
            
            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                boxset1 = []  ##"face","mouse","eye"
                boxset2 = []  ##"hand"
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        #print(xywh)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        # print(type(label))
                        # print(xyxy)
                        # print(type(xyxy))
                        # c = xyxy.tolist()
                        # # print(c)        
                        plot_one_box(xyxy, im0, label=label, color = colors[int(cls)], line_thickness=3)                  
                        x = xyxy
                        x1, y1, x2, y2 = int(x[0]), int(x[1]), int(x[2]), int(x[3])
                        
                        ObjName = names[int(cls)]
                        conf = float(conf)
                        #print(conf)
                        BndBox = [ObjName, conf, x1, y1, x2, y2]
                        if ObjName == 'face' or ObjName == 'eye' or ObjName == 'mouse':
                            #print("+++")
                            boxset1.append(BndBox)
                        if ObjName == 'hand':
                            boxset2.append(BndBox)
                        print(boxset1)

                                              
                cnt1 = 0
                cnt2 = 0
                boxset1 = getmax(boxset1)
                for x in boxset1:
                    [name, conf, x1, y1, x2, y2] = x
                    if name == 'face':
                        img11 = img1[y1:y2, x1:x2]
                        cnt1 += 1
                    if name == 'eye':
                        img2 = img1[y1:y2, x1:x2]
                        cnt1 += 1
                    if name == 'mouth':
                        img3 = img1[y1:y2, x1:x2]
                        cnt1 += 1
 
                for x in boxset2:
                    [name, conf, x1,y1,x2,y2] = x
                    if name == 'hand':
                        img5 = img1[y1:y2, x1:x2]
                        cnt2 += 1
                        cv2.imwrite(src_hand_dir1 +'/'+str(cnt) + '_'+str(cnt2) + '.jpg', img5)
                        img5 = cv2.resize(img5, (224, 224), interpolation = cv2.INTER_CUBIC)
                        img55 = Image.fromarray(img5.astype('uint8')).convert('RGB') 
                        img55 = get_test_transform1()(img55).unsqueeze(0)
          
                        with torch.no_grad():
                            out1 = net1(img55)
                        prediction = torch.argmax(out1, dim=1).cpu().item()
                        print(label1_names[prediction])
                        cv2.putText(img5, label1_names[prediction], (60, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                        cv2.imwrite(src_hand_dir+'/' + str(cnt) + '_'+str(cnt2) + '.jpg', img5)

                if cnt1 == 3:               
                    img11 = cv2.resize(img11, (224, 224), interpolation = cv2.INTER_CUBIC) #w*h  ##cv2.resize(w,h)  img.shape h*w
                    img2 = cv2.resize(img2, (224, 112), interpolation = cv2.INTER_CUBIC)
                    img3 = cv2.resize(img3, (224, 112), interpolation = cv2.INTER_CUBIC)
                    
                    img_pin1 = np.vstack((img2, img3))
                    # print(img_pin1.shape)
                    # print(img1.shape)
                    img_pin2 = np.concatenate((img11,img_pin1),axis=1)
                    # print(img_pin2.shape)
                elif cnt1 == 2:
                    img4 = np.zeros([112, 224, 3], np.uint8) ##H*w black img
                    img11 = cv2.resize(img11, (224, 224), interpolation = cv2.INTER_CUBIC)
                    img2 = cv2.resize(img2, (224, 112), interpolation = cv2.INTER_CUBIC)
                    #img_pin1 = np.vstack((img4, img2))
                    img_pin1 = np.vstack((img2, img4))
                    img_pin2 = np.concatenate((img11, img_pin1),axis=1)

                elif cnt1 == 1:
                    img4= np.zeros([112, 224, 3], np.uint8) ##H*w black img
                    img11 = cv2.resize(img11, (224, 224), interpolation = cv2.INTER_CUBIC)
                    img_pin1 = np.vstack((img4, img4))
                    img_pin2 = np.concatenate((img11, img_pin1),axis=1)

                # cv2.imshow('image', img_pin2)
                # plt.imshow(img_pin2)
                # plt.show()

                if cnt1 > 0:
                    #print("=========")
                    cv2.imwrite(src_pinjie_dir1 + '/' + str(cnt) + '.jpg', img_pin2)
                    img22 = Image.fromarray(img_pin2.astype('uint8')).convert('RGB') 
                    img22 = get_test_transform()(img22).unsqueeze(0)
                    if torch.cuda.is_available():
                        net = net.cuda()
                        img22 = img22.cuda()
                    with torch.no_grad():
                        out0 = net(img22)

                    out0 = nn.functional.softmax(out0, dim=1)
                    #print(out0)
                    
                    prediction = torch.argmax(out0, dim=1).cpu().item()
                    score = format(torch.max(out0, dim=1)[0].cpu().item(), '.5f')
                    print("======", score)
                    print(label_names[prediction])
                    cv2.putText(img_pin2, label_names[prediction]+'_'+str(score), (224, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.imwrite(src_pinjie_dir + '/' + str(cnt) + '.jpg', img_pin2)
                    

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if True:
                    #cv2.imwrite(save_path, im0)
                    cv2.imwrite(src_img_dir+'/'+ str(cnt) + '.jpg', im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform.system() == 'Darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default="runs/exp7/weights/best.pt", help='model.pt path(s)')
    parser.add_argument('--source', type=str, default="inference/video1201/", help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output1112', help='output folder')  #output folder
    parser.add_argument('--img-size', type=int, default=320, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', default=False, help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--weight_pinjie', type=str, default="runs/model_pinjie/resnet1120_pinjie.pth",  help='the weights file you want to test')
    parser.add_argument('--weight_hand', type=str, default="runs/model_hand/20201110_hand_0.9815.pth",  help='the weights file you want to test')
    #parser.add_argument('--gpu', type=bool, default=True, help='use gpu or not')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
