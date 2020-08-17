# import detect_photo_version2
# tracking_object_list = []
# tracking_object_list = detect_photo_version2.get_detected_image_from_photo(source = './inference/images/zidane.jpg', weights = 'yolov5s.pt', tracking_object_list = tracking_object_list)
# print(f"Before is"
#       f"{tracking_object_list}")
# tracking_object_list = detect_photo_version2.get_detected_image_from_photo(source = './inference/images/zidane.jpg', weights = 'yolov5s.pt', tracking_object_list = tracking_object_list)
# print(f"After is"
#       f"{tracking_object_list}")

# import detect_photo_version2
# tracking_object_list = []
# tracking_object_list = detect_photo_version2.get_detected_image_from_photo(source = './inference/images/Cafe_1.jpg', weights = 'yolov5s.pt', tracking_object_list = tracking_object_list)
# print(f"Cafe_1 is"
#       f"{tracking_object_list}")
# tracking_object_list = detect_photo_version2.get_detected_image_from_photo(source = './inference/images/Cafe_2.jpg', weights = 'yolov5s.pt', tracking_object_list = tracking_object_list)
# print(f"Cafe_2 is"
#       f"{tracking_object_list}")
# tracking_object_list = detect_photo_version2.get_detected_image_from_photo(source = './inference/images/Cafe_3.jpg', weights = 'yolov5s.pt', tracking_object_list = tracking_object_list)
# print(f"Cafe_3 is"
#       f"{tracking_object_list}")

def print_tracking_object_list(tracking_object_list):
      for b, tracking_object in enumerate(tracking_object_list):
            current_polygon = tracking_object[0][0]
            print(f"{current_polygon}")
            current_xyxy = polygon_to_xyxy(current_polygon)
            print(f"current frame's {b}th tracking object's current_xyxy is "
                  f"{current_xyxy}")

import detect_photo_version3
from xyxypc2ppc import *
tracking_object_list = []
tracking_object_list = detect_photo_version3.get_detected_image_from_photo(source = './inference/images/zidane.jpg', weights = 'yolov5s.pt', tracking_object_list = tracking_object_list)
print_tracking_object_list(tracking_object_list)


# tracking_object_list = []
# tracking_object_list = detect_photo_version3.get_detected_image_from_photo(source = './inference/images/Cafe_1.jpg', weights = 'yolov5s.pt', tracking_object_list = tracking_object_list)
# print(f"Cafe_1 is"
#       f"{tracking_object_list}")
# print_tracking_object_list(tracking_object_list)
# tracking_object_list = detect_photo_version3.get_detected_image_from_photo(source = './inference/images/Cafe_2.jpg', weights = 'yolov5s.pt', tracking_object_list = tracking_object_list)
#
# print(f"Cafe_2 is"
#       f"{tracking_object_list}")
# print_tracking_object_list(tracking_object_list)
#
# tracking_object_list = detect_photo_version3.get_detected_image_from_photo(source = './inference/images/Cafe_3.jpg', weights = 'yolov5s.pt', tracking_object_list = tracking_object_list)
# print(f"Cafe_3 is"
#       f"{tracking_object_list}")
# print_tracking_object_list(tracking_object_list)

#
# import detect_photo_version
# tracking_object_list = []
# res = detect_photo_version.get_detected_image_from_photo(source = './inference/images/zidane.jpg', weights = 'yolov5s.pt')
# print(res)

