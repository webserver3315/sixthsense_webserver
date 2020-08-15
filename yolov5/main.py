import argparse

from detect_photo_version import *
from tracker import *
from makeioutable import *
from xyxypc2ppc import *
from models.experimental import *
from utils.datasets import *
from utils.utils import *

'''
용례: $ python3 main.py --source ./inference/images/zidane.jpg --weights yolov5s.pt --conf 0.4

Input: 사진경로 및 유지중인 tracking_object_list
실행중: tracking_object_list append 하거나 new 것을 만든다.
Output: 사진 내부에서 검출된 모든 Object 에 대한 정보 -> tracking_object_list
'''

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

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            print("if opt.update")
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                get_detected_image_from_photo()
                strip_optimizer(opt.weights)
        else:
            source, weights, conf = opt.source, opt.weights, opt.conf
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
