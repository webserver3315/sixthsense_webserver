from cv2 import *
from numpy import *
from xyxypc2ppc import *
from shapely.geometry import Polygon
from models.experimental import *
from utils.datasets import *
from utils.utils import *


def scale_xyxy_from_left_to_right(R, C, r, c, xyxy):  # C = 가로픽셀수, R = 세로픽셀수
    x_multiple = c / C
    y_multiple = r / R
    # print(f"x_mult and y_mult is {x_multiple}, {y_multiple}")
    x1_late = int(xyxy[0] * x_multiple)
    y1_late = int(xyxy[1] * y_multiple)
    x2_late = int(xyxy[2] * x_multiple)
    y2_late = int(xyxy[3] * y_multiple)
    xyxy_late = [x1_late, y1_late, x2_late, y2_late]
    # print(f"xyxy_late is {xyxy_late}")
    return xyxy_late


def visualize_danger_zone_matrix(img, ORIGINAL_R, ORIGINAL_C, DANGER_ZONE_MATRIX_R, DANGER_ZONE_MATRIX_C,
                                 danger_zone_matrix):
    # default = 720 1280 180 320
    for rr in range(DANGER_ZONE_MATRIX_R):
        for cc in range(DANGER_ZONE_MATRIX_C):
            if danger_zone_matrix[rr][cc] == 0:
                continue
            xyxy = [cc, rr, cc + 1, rr + 1]
            original_xyxy = scale_xyxy_from_left_to_right(DANGER_ZONE_MATRIX_R, DANGER_ZONE_MATRIX_C, ORIGINAL_R,
                                                          ORIGINAL_C, xyxy)
            # print(f"original_xyxy is {original_xyxy}")

            # First we crop the sub-rect from the image _ https://stackoverflow.com/questions/56472024/how-to-change-the-opacity-of-boxes-cv2-rectangle
            x, y, w, h = original_xyxy[0], original_xyxy[1], original_xyxy[2] - original_xyxy[0], original_xyxy[3] - \
                         original_xyxy[1]
            alpha = danger_zone_matrix[rr][cc] * 0.0002
            sub_img = img[y:y + h, x:x + w]
            white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
            res = cv2.addWeighted(sub_img, 1 - alpha * 0.85, white_rect, alpha * 0.85, 0)
            # Putting the image back to its position
            img[y:y + h, x:x + w] = res
    return img


def print_danger_zone_matrix(r, c, danger_zone_matrix):
    for rr in range(r):
        print(f"{danger_zone_matrix[rr]}")


def degrade_danger_zone_matrix(DANGER_ZONE_MATRIX_R, DANGER_ZONE_MATRIX_C, danger_zone_matrix, DANGER_DEGRADE_STRIDE):
    for r, danger_zone_line in enumerate(danger_zone_matrix):
        for c, d in enumerate(danger_zone_line):
            # print(f"r, c, d = {r}, {c}, {d}")
            danger_zone_line[c] = danger_zone_line[c] - DANGER_DEGRADE_STRIDE
            if danger_zone_line[c] < 0:
                danger_zone_line[c] = 0
    return danger_zone_matrix


def is_parking(tracking_object):
    if len(tracking_object) == 1:
        return False
    else:
        d


# 1. TOL 과 DZM 을 비교대조하여, 위험한 TO 여부를 반환한다.
# 2. TOL 에서 Class id 가 2 인 객체가 훑고 지나간 영역의 DZM 을 가산한다. 아니면 감산한다.
# 3. DZM 은 r(세로)*c(가로) 의 2차원 int 행렬이다. r, c에 일단 기본값으로 360, 640 권장
def is_tracking_object_list_dangerous(ORIGINAL_R, ORIGINAL_C, DANGER_ZONE_MATRIX_R, DANGER_ZONE_MATRIX_C,
                                      tracking_object_list, danger_zone_matrix):
    DANGER_UPDATE_STRIDE = 1500  # 훑고지나갈때마다 상승하는 위험도
    DANGER_THRESHOLD = 1000  # 위험역치
    DANGER_SCORE_MAX = 5000  # 최대 저장할 수 있는 위험점수
    DANGER_DEGRADE_STRIDE = 1  # 한 프레임마다 경감되는 위험도
    SPEED_THRESHOLD = 1

    tracking_object_list_danger_list = []  # bool형 1차원 배열. 각 tracking object 의 위험구역위치여부를 의미.
    for b, tracking_object in enumerate(tracking_object_list):
        degrade_danger_zone_matrix(DANGER_ZONE_MATRIX_R, DANGER_ZONE_MATRIX_C, danger_zone_matrix,
                                   DANGER_DEGRADE_STRIDE)
        if tracking_object[0][2] == 0 or get_speed(tracking_object) <= SPEED_THRESHOLD:  # class id가 0, 즉 사람이면 스킵
            tracking_object_list_danger_list.append(False)
            continue
        current_polygon = tracking_object[0][0]
        original_xyxy = polygon_to_xyxy(current_polygon)
        scaled_xyxy = scale_xyxy_from_left_to_right(ORIGINAL_R, ORIGINAL_C, DANGER_ZONE_MATRIX_R, DANGER_ZONE_MATRIX_C,
                                                    original_xyxy)
        danger_score_of_current_tracking_object = 0

        for r in range(scaled_xyxy[1], scaled_xyxy[3]):
            for c in range(scaled_xyxy[0], scaled_xyxy[2]):
                danger_score_of_current_tracking_object += danger_zone_matrix[r][c]
                danger_zone_matrix[r][c] += DANGER_UPDATE_STRIDE
                if danger_zone_matrix[r][c] > DANGER_SCORE_MAX:
                    danger_zone_matrix[r][c] = DANGER_SCORE_MAX
        w, h = scaled_xyxy[2] - scaled_xyxy[0], scaled_xyxy[3] - scaled_xyxy[1]
        if w == 0 or h == 0:
            tracking_object_list_danger_list.append(False)
            continue
        average_danger_score = danger_score_of_current_tracking_object / (w * h)
        if average_danger_score >= DANGER_THRESHOLD:
            tracking_object_list_danger_list.append(True)
        else:
            tracking_object_list_danger_list.append(False)

    return tracking_object_list_danger_list, danger_zone_matrix
