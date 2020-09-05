from os import walk
import sys
from detect_photo_version3 import *
# from detect_photo_version_certified import *
from xyxypc2ppc import *

sys.stdout = open('/data/swmrepo/sunshine-2/yolov5/inference/output/output.txt', 'w')


def print_tracking_object_list(tracking_object_list):
    for b, tracking_object in enumerate(tracking_object_list):
        for p, ppc in enumerate(tracking_object):
            current_polygon = ppc[0]
            current_xyxy = polygon_to_xyxy(current_polygon)
            middle_c = (current_xyxy[0] + current_xyxy[2]) / 2
            middle_r = (current_xyxy[1] + current_xyxy[3]) / 2
            if p != 0:
                print(f"    {p}Zen ppc of {b}th TO: {[middle_c, middle_r]}, {ppc[2]}")
            else:
                print(f"{p}Zen ppc of {b}th TO: {[middle_c, middle_r]}, {ppc[2]}")


#
# tracking_object_list = []
# danger_zone_matrix = [[0 for c in range(320)] for r in range(180)]
# # danger_zone_matrix = [[0 for c in range(1280)] for r in range(720)]
# for i in range(1, 51):
#     source = f'./inference/images/Accident_04_Capture/img{i:03d}.jpg'
#     tracking_object_list = get_detected_image_from_photo(source=source, weights='yolov5s.pt', tracking_object_list=tracking_object_list,
#                                                                                danger_zone_matrix=danger_zone_matrix)

t0 = time.time()

tracking_object_list = []
# danger_zone_matrix = [[0 for c in range(160)] for r in range(120)]
danger_zone_matrix = None
f = []
mypath = '/data/swmrepo/sunshine-2/yolov5/inference/images/Accident_04_Capture'
for (dirpath, dirnames, filenames) in walk(mypath):
    f.extend(filenames)
    break
f.sort()
for b, source in enumerate(f):
    source = mypath + '/' + source
    # source2 = mypath + f"/Accident_00_{b:03d}.jpg"
    # os.rename(source, source2)
    if danger_zone_matrix is None:
        original_img = imread(source)
        danger_zone_matrix = initialize_danger_zone_matrix(original_img)
    tracking_object_list = get_detected_image_from_photo(source=source, weights='yolov5s.pt',
                                                         tracking_object_list=tracking_object_list,
                                                         danger_zone_matrix=danger_zone_matrix)
    # print(f"tracking_object_list's type = {type(tracking_object_list)}")
    # print(f"tracking_object's type = {type(tracking_object_list[0])}")
    # print(f"\n\n\n*********{b:03d}th img**********")
    # print_tracking_object_list(tracking_object_list)

print('Really Finally, Done. (%.3fs)' % (time.time() - t0))


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
#     tracking_object_list = get_detected_image_from_photo(source=source, weights='yolov5s.pt',
#                                                          tracking_object_list=tracking_object_list)
