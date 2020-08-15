import argparse

from tracker.py import *
from makeioutable.py import *
from xyxypc2ppc import *
from models.experimental import *
from utils.datasets import *
from utils.utils import *

'''
Input: 사진경로
Output: 사진 내부에서 검출된 모든 Object 에 대한 정보 
'''


def detect_photo(save_img=False, tracking_object_list=[]):
    print("detect_photo function Start!")
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    save_img = True
    dataset = LoadImages(source, img_size=imgsz)
    print(f"dataset is {dataset}")

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = torch_utils.time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        """
        Tracking Object = [현좌표, 1전좌표, 2전좌표, 3전좌표, 4전좌표], 확신도, 클래스id]
        Tracking Object List = [
            [[1전폴리곤, 2전폴리곤, 3전폴리곤, 4전폴리곤], 확신도, 클래스id],
            [[1전폴리곤, 2전폴리곤, 3전폴리곤, 4전폴리곤], 확신도, 클래스id],
            [[1전폴리곤, 2전폴리곤, 3전폴리곤, 4전폴리곤], 확신도, 클래스id],
            ...
            [[현폴리곤, 1전폴리곤, 2전폴리곤, 3전폴리곤, 4전폴리곤], 확신도, 클래스id]
        ]
        Detected Object List = [
            [폴리곤, 확신도, 클래스id],
            [폴리곤, 확신도, 클래스id],
            [폴리곤, 확신도, 클래스id],
            ...
            [폴리곤, 확신도, 클래스id],
        ]
        >>> Tracking Object List 랑 Detected Object List 의 IoU Table 작성
        """
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s

            '''
            # 첫 프레임이면, 모든 DO를 신규 TO로써 TOL에 삽입해야한다.
            if i == 0:
                # 구현할 것

            '''
            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # 이미지에 맞게 xyxy 좌표를 scaling 하는 듯.
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                detected_object_list = []
                cnt = 0
                for *xyxy, conf, cls in det:
                    current_polygon = xyxy_to_polygon(xyxy)
                    current_ppc = [current_polygon, conf, cls]
                    detected_object_list.append(current_ppc)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)] + str(cnt), conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    cnt = cnt + 1

                if not tracking_object_list:
                    tracking_object_list.append(detected_object_list)
                else:
                    iou_table = make_iou_table_from_TOL_and_DOL(tracking_object_list, detected_object_list)
                    hist, done = solve(len(tracking_object_list[0]), len(detected_object_list), iou_table=[])
                    for o, detected_ppc in enumerate(detected_object_list):
                        next_append = hist[o][1]
                        new_append = []
                        if next_append == -1:
                            new_append.append(detected_ppc)
                        else
                            if len(tracking_object_list[next_append][0]) >= 5:  # 트래킹 객체의 큐사이즈가 5 초과라면 하나 버리기
                                tracking_object_list[next_append].pop()
                            tracking_object_list[next_append].insert(0, detected_ppc)  # 시간복잡 O(n) 이니 차후수정
                    for b, tracking_object in enumerate(tracking_object_list):
                        if done[b] == False:
                            del tracking_object_list[b]
                    for new_tracking_object in new_append:
                        tracking_object_list.append([[new_tracking_object], ])

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

    print('Done. (%.3fs)' % (time.time() - t0))
    return tracking_object_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    print(detect_photo())
