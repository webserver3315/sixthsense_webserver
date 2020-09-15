from yolov5.detect_photo_version3 import get_detected_image_from_photo
from yolov5.xyxypc2ppc import draw_box_from_tracking_object
from yolov5.danger_zone import initialize_danger_zone_matrix
from accident_cal import accident_percentage
from visualize import generate_video_demo, filter_tracking_object_list_by_class
from config import SAVE_FOLDER
import cv2
from os import path

class Device:
    def __init__(self, id):
        self.id = id
        self.objects_list = []
        self.danger_zone = None
        self.image = None
        self.framed_images = []


    def new_frame(self, image_path, timestamp):
        self.image = cv2.imread(image_path)
        if not self.danger_zone:
            self.danger_zone = initialize_danger_zone_matrix(self.image)
        self.objects_list = get_detected_image_from_photo(image_path, 'yolov5s.pt', self.objects_list, self.danger_zone)

        framed = generate_video_demo(self)
        self.framed_images.append(framed)
        filename, file_extension = path.splitext(image_path)
        cv2.imwrite(path.join(SAVE_FOLDER, str(len(self.framed_images))+file_extension), framed)
        return self.objects_list
    
    @property
    def accident_percentage_matrix(self):
        return accident_percentage(device.objects_list)
    
    @property
    def person_list(self):
        return filter_tracking_object_list_by_class(self.objects_list, 0)

    @property
    def car_list(self):
        return filter_tracking_object_list_by_class(self.objects_list, 2)