from yolov5.detect_photo_version2 import get_detected_image_from_photo

class Device:
    def __init__(self, id):
        self.id = id
        self.objects_list = []
    
    def new_frame(self, image_path, timestamp):
        self.objects_list = get_detected_image_from_photo(image_path, tracking_object_list=self.objects_list)
        return self.objects_list
        