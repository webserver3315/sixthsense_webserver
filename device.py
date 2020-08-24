from yolov5.detect_photo_version3 import get_detected_image_from_photo

class Device:
    def __init__(self, id):
        self.id = id
        self.objects_list = []
        self.danger_zone = [[0]*320]*180
    
    def new_frame(self, image_path, timestamp):
        self.objects_list = get_detected_image_from_photo(image_path, 'yolov5s.pt', self.objects_list, self.danger_zone)
        return self.objects_list
        