from yolov5.xyxypc2ppc import draw_box_from_tracking_object, draw_lines_from_tracking_object
from accident_cal import accident_percentage
from yolov5.danger_zone import automatic_visualize_danger_zone_matrix
import cv2

def filter_tracking_object_list_by_class(tracking_object_list, object_class):
    return list(filter(lambda x: x[0][2]==object_class, tracking_object_list))

def generate_video_demo(device):
    res = device.image.copy()
    accident_percentage_result = accident_percentage(device.objects_list)

    person_list = filter_tracking_object_list_by_class(device.objects_list, 0)
    car_list = filter_tracking_object_list_by_class(device.objects_list, 2)

    # draw shapes for person
    for i, person_object in enumerate(person_list):
        current_person_object = person_object[0]
        (polygon, confidnecy, object_class) = current_person_object
        accident_percentages_list_for_me = [x[i] for x in accident_percentage_result]
        automatic_visualize_danger_zone_matrix(res, device.danger_zone)

        max_percentage = max(accident_percentages_list_for_me)
        danger_color = (0, min(255, 510-(510*max_percentage)), min(255, 510*max_percentage))

        draw_box_from_tracking_object(person_object, res, danger_color, f'P {max_percentage}%', 2)
        draw_lines_from_tracking_object(person_object, res, danger_color, 1)

    return res
        
    
