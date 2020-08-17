from xyxypc2ppc import *
from shapely.geometry import Polygon
from models.experimental import *
from utils.datasets import *
from utils.utils import *

def scale_xyxy_to_360_640(R, C, xyxy):  # C = 가로픽셀수, R = 세로픽셀수
    x_multiple = 640 / C
    y_multiple = 360 / R
    x1_late = int(xyxy[0] * x_multiple)
    y1_late = int(xyxy[1] * y_multiple)
    x2_late = int(xyxy[2] * x_multiple)
    y2_late = int(xyxy[3] * y_multiple)
    xyxy_late = [x1_late, y1_late, x2_late, y2_late]
    return xyxy_late


# 1. TOL 과 DZM 을 비교대조하여, 위험한 TO 여부를 반환한다.
# 2. TOL 에서 Class id 가 2 인 객체가 훑고 지나간 영역의 DZM 을 가산한다. 아니면 감산한다.
# 3. DZM 은 360(세로)*640(가로) 의 2차원 int 행렬이다.
def is_tracking_object_list_dangerous(R, C, tracking_object_list, danger_zone_matrix):
    DANGER_UPDATE_STRIDE = 200  # 훑고지나갈때마다 상승하는 위험도
    DANGER_THRESHOLD = 500  # 위험역치
    DANGER_SCORE_MAX = 1000  # 최대 저장할 수 있는 위험점수

    tracking_object_list_danger_list = []
    for b, tracking_object in tracking_object_list:
        current_polygon = tracking_object[0]
        tracking_xyxy = polygon_to_xyxy(current_polygon)
        scaled_xyxy = scale_xyxy_to_360_640(R, C, tracking_xyxy)
        danger_score = 0
        for r in range(scaled_xyxy[1], scaled_xyxy[3]):
            for c in range(scaled_xyxy[0], scaled_xyxy[2]):
                danger_score = danger_score + danger_zone_matrix[r][c]
                danger_zone_matrix[r][c] = danger_zone_matrix[r][c] + DANGER_UPDATE_STRIDE
                if danger_zone_matrix[r][c] > DANGER_SCORE_MAX:
                    danger_zone_matrix[r][c] = DANGER_SCORE_MAX
        w, h = scaled_xyxy[2] - scaled_xyxy[0], scaled_xyxy[3] - scaled_xyxy[1]
        average_danger_score = danger_score / (w * h)
        if average_danger_score >= DANGER_THRESHOLD:
            tracking_object_list_danger_list.append(True)
        else:
            tracking_object_list_danger_list.append(False)

    return tracking_object_list_danger_list, danger_zone_matrix
