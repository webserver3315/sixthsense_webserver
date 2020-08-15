'''
Created by KYEONGTAE PARK on 2020/08/11.

로그 형식:

video 1/1 (160/187) /data/DLCV/Detection/yolov5/inference/images/street.mp4: det is :  tensor([[1.00000e+00, 5.58000e+02, 4.25000e+02, 8.82000e+02, 9.14450e-01, 2.00000e+00],
        [4.13000e+02, 2.16000e+02, 5.82000e+02, 3.58000e+02, 8.54321e-01, 2.00000e+00],
        [1.46200e+03, 3.37000e+02, 1.91800e+03, 9.95000e+02, 7.03214e-01, 2.00000e+00],
        [1.21400e+03, 6.90000e+01, 1.35600e+03, 1.92000e+02, 7.02321e-01, 2.00000e+00],
        [1.48100e+03, 2.21000e+02, 1.92000e+03, 4.64000e+02, 6.57817e-01, 2.00000e+00],
        [1.49400e+03, 3.42000e+02, 1.92000e+03, 6.01000e+02, 5.68211e-01, 2.00000e+00],
        [1.84400e+03, 2.00000e+00, 1.92000e+03, 1.01000e+02, 4.16623e-01, 5.80000e+01]], device='cuda:0')
384x640 6 cars, 1 potted plants, Done. (0.009s)

video 1/1 (167/187) /data/DLCV/Detection/yolov5/inference/images/street.mp4: det is :  tensor([[1.00000e+00, 5.56000e+02, 4.25000e+02, 8.82000e+02, 9.11287e-01, 2.00000e+00],
        [4.13000e+02, 2.15000e+02, 5.81000e+02, 3.58000e+02, 8.54835e-01, 2.00000e+00],
        [1.46500e+03, 5.15000e+02, 1.91900e+03, 9.87000e+02, 7.48497e-01, 2.00000e+00],
        [1.21000e+03, 7.70000e+01, 1.35000e+03, 1.93000e+02, 7.41220e-01, 2.00000e+00],
        [1.49000e+03, 3.35000e+02, 1.91800e+03, 6.02000e+02, 6.47647e-01, 2.00000e+00],
        [1.84300e+03, 3.00000e+00, 1.92000e+03, 1.03000e+02, 6.31918e-01, 5.80000e+01],
        [1.47900e+03, 2.11000e+02, 1.92000e+03, 5.00000e+02, 5.52295e-01, 2.00000e+00],
        [1.70600e+03, 8.50000e+01, 1.79800e+03, 1.73000e+02, 4.37537e-01, 5.80000e+01],
        [4.25000e+02, 5.08000e+02, 4.95000e+02, 6.57000e+02, 4.05220e-01, 0.00000e+00]], device='cuda:0')
384x640 1 persons, 6 cars, 2 potted plants, Done. (0.007s)

이걸, shapely 객체의 ppc 로 바꿔야 한다.
ppc = [shapely.Polygon, probability, classid]
'''
from shapely.geometry import Polygon
from models.experimental import *
from utils.datasets import *
from utils.utils import *


def get_intersection_over_union(p1: Polygon, p2: Polygon):
    intersection = p1.intersection(p2)
    union = p1.union(p2)
    return intersection.area / union.area


'''
def makeioutable(before_ppc: list, now_ppc: list):
    res = []
    for b in before_ppc[0]:
        tmp_res = []
        for o in now_ppc[0]:
            tmp_res.append(get_intersection_over_union(b, o))
        res.append(tmp_res)
    return res
'''

# tracking_object_list = 폴리곤으로만 구성된 리스트
# detected_object_list = 폴리곤으로만 구성된 리스트
def make_iou_table(tracking_object_list, detected_object_list):
    res = []
    for i, tracking_object in enumerate(tracking_object_list):
        iou_line = []
        for j, detected_object in enumerate(detected_object_list):
            current_iou = get_intersection_over_union(tracking_object, detected_object)
            iou_line.append(current_iou)
        res.append(iou_line)
    return res