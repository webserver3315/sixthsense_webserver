import unittest
import visualize
from accident_cal import xyxy_to_polygon, accident_percentage


class BasicObjectTestCase(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        self.car_tracking_obj_list = [[xyxy_to_polygon([10+x, 10, 60+x, 40]), 1, 2] for x in range(0, 91, 30)][::-1]
        self.person_tracking_obj_list = [[xyxy_to_polygon([300, 60-x, 320, 90-x]), 1, 0] for x in range(0, 31, 10)][::-1]
        self.person_tracking_obj_list1 = [[xyxy_to_polygon([240-x, 60-x, 260-x, 90-x]), 1, 0] for x in range(0, 31, 10)][::-1]
        self.person_tracking_obj_list2 = [[xyxy_to_polygon([140+x, 60-x, 160+x, 90-x]), 1, 0] for x in range(0, 31, 10)][::-1]
        self.tracking_object_list = [self.car_tracking_obj_list, self.person_tracking_obj_list, self.person_tracking_obj_list1, self.person_tracking_obj_list2]


    def test_filter_tracking_object_list_by_class_0(self):
        self.assertEqual(visualize.filter_tracking_object_list_by_class(self.tracking_object_list, 0), [self.person_tracking_obj_list, self.person_tracking_obj_list1, self.person_tracking_obj_list2])
    
    def test_filter_tracking_object_list_by_class_1(self):
        self.assertEqual(visualize.filter_tracking_object_list_by_class(self.tracking_object_list, 2), [self.car_tracking_obj_list])

