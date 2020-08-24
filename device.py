from yolov5.detect_photo_version3 import get_detected_image_from_photo
from yolov5.xyxypc2ppc import draw_box_from_tracking_object
from visualize import generate_video_demo
import cv2

class Device:
    def __init__(self, id):
        self.id = id
        self.objects_list = []
        self.danger_zone = [[0]*320]*180
        self.image = None
        self.framed_images = []


    def new_frame(self, image_path, timestamp):
        self.objects_list = get_detected_image_from_photo(image_path, 'yolov5s.pt', self.objects_list, self.danger_zone)
        self.image = cv2.imread(image_path)
        self.framed_images.append(generate_video_demo(self.image, self.objects_list))
        return self.objects_list
        