from os import walk
import sys
# from detect_photo_version3 import *
from detect_photo_version_certified import *
from xyxypc2ppc import *

sys.stdout = open('/data/swmrepo/sunshine-2/yolov5/inference/output/output.txt', 'w')


def print_tracking_object_list(tracking_object_list):
    for b, tracking_object in enumerate(tracking_object_list):
        current_polygon = tracking_object[0][0]
        print(f"{current_polygon}")
        current_xyxy = polygon_to_xyxy(current_polygon)
        print(f"current frame's {b}th tracking object's current_xyxy is "
              f"{current_xyxy}")


# tracking_object_list = []
# danger_zone_matrix = [[0 for c in range(320)] for r in range(180)]
# # danger_zone_matrix = [[0 for c in range(1280)] for r in range(720)]
# for i in range(1, 51):
#     source = f'./inference/images/Accident_04_Capture/img{i:03d}.jpg'
#     tracking_object_list = get_detected_image_from_photo(source=source, weights='yolov5s.pt', tracking_object_list=tracking_object_list,
#                                                                                danger_zone_matrix=danger_zone_matrix)
#
# tracking_object_list = []
# danger_zone_matrix = [[0 for c in range(320)] for r in range(180)]
# # danger_zone_matrix = [[0 for c in range(1280)] for r in range(720)]
#
# f = []
# mypath = '/data/swmrepo/sunshine-2/yolov5/inference/eun'
# for (dirpath, dirnames, filenames) in walk(mypath):
#     f.extend(filenames)
#     break
# f.sort()
# for source in f:
#     source = mypath + '/' + source
#     tracking_object_list = get_detected_image_from_photo(source=source, weights='yolov5x.pt',
#                                                          tracking_object_list=tracking_object_list,
#                                                          danger_zone_matrix=danger_zone_matrix)



tracking_object_list = []
danger_zone_matrix = [[0 for c in range(320)] for r in range(180)]
# danger_zone_matrix = [[0 for c in range(1280)] for r in range(720)]

f = []
mypath = '/data/swmrepo/sunshine-2/yolov5/inference/eun'
for (dirpath, dirnames, filenames) in walk(mypath):
    f.extend(filenames)
    break
f.sort()
for source in f:
    source = mypath + '/' + source
    tracking_object_list = get_detected_image_from_photo(source=source, weights='yolov5s.pt',
                                                         tracking_object_list=tracking_object_list)