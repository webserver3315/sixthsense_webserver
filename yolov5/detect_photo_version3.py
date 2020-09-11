import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from collections import deque
import argparse
from danger_zone import *
# from danger_zone_original import *
from tracker import *
from makeioutable import *
from xyxypc2ppc import *
from models.experimental import *
from utils.datasets import *
from utils.utils import *

# Namespace(agnostic_nms=False, augment=False, classes=None, conf_thres=0.4, device='', img_size=640, iou_thres=0.5, output='inference/output', save_txt=False, source='./inference/images/zidane.jpg', update=False, view_img=False, weights=['yolov5s.pt'])

'''
이 코드는 opt-독립인 코드입니다.
version2 말고 version 의 코드는 opt-독립이 아니나, 안정된 작동이 보장된 코드입니다.
version2 또한 8.15 18:24부로 안정작동이 확인되었고 Push했습니다.

Input: 사진경로 및 유지중인 tracking_object_list
실행중: tracking_object_list append 하거나 new 것을 만든다.
Output: 사진 내부에서 검출된 모든 Object 에 대한 정보 -> tracking_object_list
'''


def print_tracking_object_list_length(tracking_object_list):
    for b, tracking_object in enumerate(tracking_object_list):
        print(f"{b}th TOBJ's history length = {len(tracking_object)}")
    print(f"\n")

def initial_setting():
    original_img = imread(source)
    ORIGINAL_R, ORIGINAL_C = original_img.shape[0], original_img.shape[1]
    with torch.no_grad():
        out, weights, view_img, save_txt, imgsz = \
            'inference/output', weights, False, False, int(640)  # default: 640
        device = torch_utils.select_device('')
        if not os.path.exists(out):
            os.makedirs(out)  # make new output folder
        half = device.type != 'cpu'  # half precision only supported on CUDA
        if half:
            model.half()  # to FP16
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

        # Second-stage classifier
        classify = False
        if classify:
            modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
            modelc.to(device).eval()
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

def get_image_return_detected_object_list(source, weights):
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)
        t2 = torch_utils.time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if det is not None and len(det):
                # 이미지에 맞게 xyxy 좌표를 scaling 하는 듯.
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # print(f"det is {det}")
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                detected_object_list = []
                for *xyxy, conf, cls in det:
                    # cls 0: Person, 1: bicycle, 2: Car, 3: Motorcycle, 4: Airplane, 5: Bus, 6: Train, 7: Truck
                    if int(cls) > 7:
                        continue
                    if cls > 2:  # 버스, 트럭 등은 전부 Car 로 통일
                        cls = 2
                    current_polygon = xyxy_to_polygon(xyxy)
                    current_ppc = [current_polygon, float(conf), int(cls)]
                    detected_object_list.append(current_ppc)
    return detected_object_list


def append_detect_object_to_tracking_object_list(detected_object_list, tracking_object_list):
    if not tracking_object_list:
        for detected_ppc in detected_object_list:
            tracking_object_list.append(deque([detected_ppc]))
            # print_tracking_object_list_length(tracking_object_list)
    else:
        iou_table = make_iou_table_from_TOL_and_DOL(tracking_object_list, detected_object_list)
        iou_table = make_iou_table_to_iou_pair_table(iou_table)
        hist, done = solve(len(tracking_object_list), len(detected_object_list), iou_table)

        new_append = []
        for o, detected_ppc in enumerate(detected_object_list):
            next_append = hist[o][1]
            if next_append == -1:
                new_append.append(deque([detected_ppc]))
            else:
                # print(f"tracking_object_list[{next_append}] = {len(tracking_object_list[next_append])}")
                if len(tracking_object_list[next_append]) >= TRACKING_OBJECT_MAX_SIZE:
                    tracking_object_list[next_append].pop()
                # print_tracking_object_list_length(f"{o}th for : {tracking_object_list}")
                tracking_object_list[next_append].appendleft(detected_ppc)  # 시간복잡 O(n) 이니 차후수정
        for b, tracking_object in reversed(list(enumerate(tracking_object_list))):
            if not done[b]:
                print(f"{b}th tracking object's track has been ended")
                del tracking_object_list[b]
        for new_tracking_object in new_append:
            tracking_object_list.append(new_tracking_object)
    return tracking_object_list


def get_tracking_object_list_danger_list(ORIGINAL_R, ORIGINAL_C,
                          DANGER_ZONE_MATRIX_R, DANGER_ZONE_MATRIX_C,
                          tracking_object_list, danger_zone_matrix):
    # tt0 = time.time()
    tracking_object_list_danger_list, danger_zone_matrix = is_tracking_object_list_dangerous(ORIGINAL_R,
                                                                                             ORIGINAL_C,
                                                                                             DANGER_ZONE_MATRIX_R,
                                                                                             DANGER_ZONE_MATRIX_C,
                                                                                             tracking_object_list,
                                                                                             danger_zone_matrix)
    # print('is_tracking_object_list_dangerous: (%.3fs)' % (time.time() - tt0))
    return tracking_object_list_danger_list


def draw_box_and_lines_on_image(tracking_object_list, tracking_object_list_danger_list, im0):
    out, weights, view_img, save_txt, imgsz = \
        'inference/output', weights, False, False, int(640)  # default: 640

    for b, tracking_object in enumerate(tracking_object_list):
        xyxy = polygon_to_xyxy(tracking_object[0][0])
        conf, cls = tracking_object[0][1], tracking_object[0][2]

        speed = get_speed(tracking_object)

        if tracking_object_list_danger_list[b] > 0:
            label = f"Danger: {names[int(cls)] + str(b)} {speed}km/h {int(tracking_object_list_danger_list[b] / 500)}%"
        else:
            label = f"{names[int(cls)] + str(b)} {speed}km/h"

        if tracking_object[0][2] != 0:  # 차량은 파랑. 참고로 BGR 순임
            colors = (255, 0, 0)
        elif tracking_object_list_danger_list[b] != 0:  # 위험한 사람은 빨강
            colors = (0, 0, 255)
        else:  # 일반인은 초록
            colors = (0, 255, 0)
        draw_lines_from_tracking_object(tracking_object, im0, color=colors, line_thickness=2)
        draw_box_from_tracking_object(tracking_object, im0, label=label, color=colors,
                                      line_thickness=3)
    # Danger_zone 현황을 위험구역 색칠로써 가시화. 위험스택 쌓일수록 하얗게 변함.
    # tt0 = time.time()
    im0 = visualize_danger_zone_matrix(im0, ORIGINAL_R, ORIGINAL_C, DANGER_ZONE_MATRIX_R,
                                       DANGER_ZONE_MATRIX_C, danger_zone_matrix)
    # print('visualize_danger_zone_matrix: (%.3fs)' % (time.time() - tt0))

    # 저장하는 부분. 크게 신경쓸 것 없음.
    cv2.imwrite(save_path, im0)

    print('%sDone. (%.3fs)' % (s, t2 - t1))
    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Finally, Done. (%.3fs)' % (time.time() - t0))
    # print_tracking_object_list_length(tracking_object_list)
    return tracking_object_list