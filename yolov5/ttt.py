import detect_photo_version2
tracking_object_list = []
tracking_object_list = detect_photo_version2.get_detected_image_from_photo(source = './inference/images/zidane.jpg', weights = 'yolov5s.pt', tracking_object_list = tracking_object_list)
print(f"Before is"
      f"{tracking_object_list}")
tracking_object_list = detect_photo_version2.get_detected_image_from_photo(source = './inference/images/zidane.jpg', weights = 'yolov5s.pt', tracking_object_list = tracking_object_list)
print(f"After is"
      f"{tracking_object_list}")

#
# import detect_photo_version
# tracking_object_list = []
# res = detect_photo_version.get_detected_image_from_photo(source = './inference/images/zidane.jpg', weights = 'yolov5s.pt')
# print(res)

