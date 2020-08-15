import argparse

from tracker import *
from makeioutable import *
from xyxypc2ppc import *
from models.experimental import *
from utils.datasets import *
from utils.utils import *

'''
Input: 사진경로 및 유지중인 tracking_object_list
실행중: tracking_object_list append 하거나 new 것을 만든다.
Output: 사진 내부에서 검출된 모든 Object 에 대한 정보 -> tracking_object_list
'''


def get_detected_image_from_photo(source, weights, tracking_object_list=[]):
    print("detect_photo function Start!")
    # out, source, weights, view_img, save_txt, imgsz = \
        # opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = False
    out, weights, view_img, save_txt, imgsz = \
        'inference/output', weights, 'store_true', 'store_true', int(640)

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
        let ppc = [폴리곤, 확신도, 클래스id]

        Tracking Object = [현[폴리곤, 확신도, 클래스id], 1전[폴리곤, 확신도, 클래스id], 2전[폴리곤, 확신도, 클래스id], 3전[폴리곤, 확신도, 클래스id], 4전[폴리곤, 확신도, 클래스id]]
        Tracking Object List = [
            [현[폴리곤, 확신도, 클래스id], 1전[폴리곤, 확신도, 클래스id], 2전[폴리곤, 확신도, 클래스id], 3전[폴리곤, 확신도, 클래스id], 4전[폴리곤, 확신도, 클래스id]],
            [현[폴리곤, 확신도, 클래스id], 1전[폴리곤, 확신도, 클래스id], 2전[폴리곤, 확신도, 클래스id], 3전[폴리곤, 확신도, 클래스id], 4전[폴리곤, 확신도, 클래스id]],
            [현[폴리곤, 확신도, 클래스id], 1전[폴리곤, 확신도, 클래스id], 2전[폴리곤, 확신도, 클래스id], 3전[폴리곤, 확신도, 클래스id], 4전[폴리곤, 확신도, 클래스id]],
            ...
            [현[폴리곤, 확신도, 클래스id], 1전[폴리곤, 확신도, 클래스id], 2전[폴리곤, 확신도, 클래스id], 3전[폴리곤, 확신도, 클래스id], 4전[폴리곤, 확신도, 클래스id]]
        ]
        Detected Object List = [
            [폴리곤, 확신도, 클래스id], [폴리곤, 확신도, 클래스id], [폴리곤, 확신도, 클래스id], [폴리곤, 확신도, 클래스id], [폴리곤, 확신도, 클래스id], ..., [폴리곤, 확신도, 클래스id]
        ]
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
                # print(f"det is {det}")
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

                    label = '%s %.2f' % (names[int(cls)] + str(cnt), conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                    cnt = cnt + 1

                if not tracking_object_list:
                    for detected_ppc in detected_object_list:
                        tracking_object_list.append([detected_ppc])
                else:
                    iou_table = make_iou_table_from_TOL_and_DOL(tracking_object_list, detected_object_list)
                    print(f"iou_table is {iou_table}")
                    print(f"length of tracking_object_list is {len(tracking_object_list)},"
                          f"length of detected_object_list is {len(detected_object_list)}")
                    hist, done = solve(len(tracking_object_list), len(detected_object_list), iou_table)
                    print(f"hist is {hist},"
                          f"done is {done}")

                    new_append = []
                    for o, detected_ppc in enumerate(detected_object_list):
                        next_append = hist[o][1]
                        if next_append == -1:
                            new_append.append([detected_ppc])
                        else:
                            if len(tracking_object_list[next_append][0]) >= 5:  # 트래킹 객체의 큐사이즈가 5 초과라면 하나 버리기
                                tracking_object_list[next_append].pop()
                            tracking_object_list[next_append].insert(0, detected_ppc)  # 시간복잡 O(n) 이니 차후수정
                    for b, tracking_object in enumerate(tracking_object_list):
                        if not done[b]:
                            print(f"{b}th tracking object's track has been ended")
                            # del tracking_object_list[b]
                    for new_tracking_object in new_append:
                        tracking_object_list.append(new_tracking_object)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

    print('Done. (%.3fs)' % (time.time() - t0))
    return tracking_object_list


if __name__ == '__main__':
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            print("if opt.update")
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                get_detected_image_from_photo()
                strip_optimizer(opt.weights)
        else:
            source = './inference/images/zidane.jpg'
            weights = 'yolov5l.pt'
            tracking_object_list = []
            tracking_object_list = get_detected_image_from_photo(source, weights, tracking_object_list)
            print(f"length of tracking_object_list is {len(tracking_object_list)}")
            print(tracking_object_list)
            for b, tracking_object in enumerate(tracking_object_list):
                # print(tracking_object)
                tracking_ppc = tracking_object[0]
                tracking_polygon = tracking_ppc[0]
                tracking_polygon
                print(tracking_polygon)
    # print(f"final tracking_object_list is {tracking_object_list}")