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


def visualize_danger_zone_matrix(img, R, C, r, c, danger_zone_matrix):
    # default = 720 1280 180 320
    # print(f"R,C,r,c = {R},{C},{r},{c}")
    # print(f"x_mult and y_mult is {c / C}, {r / R}")
    for rr in range(r):
        for cc in range(c):
            if danger_zone_matrix[rr][cc] == 0:
                continue
            xyxy = [cc, rr, cc + 1, rr + 1]
            original_xyxy = scale_xyxy_from_left_to_right(r, c, R, C, xyxy)
            # print(f"original_xyxy is {original_xyxy}")

            # First we crop the sub-rect from the image _ https://stackoverflow.com/questions/56472024/how-to-change-the-opacity-of-boxes-cv2-rectangle
            x, y, w, h = original_xyxy[0], original_xyxy[1], original_xyxy[2] - original_xyxy[0], original_xyxy[3] - \
                         original_xyxy[1]
            alpha = danger_zone_matrix[rr][cc] * 0.001
            # print(f"x, y, x+w, y+h = {x}, {y}, {x + w}, {y + h}")
            sub_img = img[y:y + h, x:x + w]
            white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
            res = cv2.addWeighted(sub_img, 1 - alpha, white_rect, alpha, 0)
            # Putting the image back to its position
            img[y:y + h, x:x + w] = res
    return img


def print_danger_zone_matrix(r, c, danger_zone_matrix):
    for rr in range(r):
        print(f"{danger_zone_matrix[rr]}")


# 1. TOL 과 DZM 을 비교대조하여, 위험한 TO 여부를 반환한다.
# 2. TOL 에서 Class id 가 2 인 객체가 훑고 지나간 영역의 DZM 을 가산한다. 아니면 감산한다.
# 3. DZM 은 r(세로)*c(가로) 의 2차원 int 행렬이다. r, c에 일단 기본값으로 360, 640 권장
def is_tracking_object_list_dangerous(R, C, r, c, tracking_object_list, danger_zone_matrix):
    DANGER_UPDATE_STRIDE = 200  # 훑고지나갈때마다 상승하는 위험도
    DANGER_THRESHOLD = 500  # 위험역치
    DANGER_SCORE_MAX = 1000  # 최대 저장할 수 있는 위험점수

    tracking_object_list_danger_list = []  # bool형 1차원 배열. 각 tracking object 의 위험구역위치여부를 의미.
    # print(tracking_object_list)
    for b, tracking_object in enumerate(tracking_object_list):
        # x1_max, y1_max, x2_max, y2_max = 0, 0, 0, 0
        if tracking_object[0][2] != 2:  # class id가 0이 아니면 스킵
            tracking_object_list_danger_list.append(False)
            continue
        current_polygon = tracking_object[0][0]
        original_xyxy = polygon_to_xyxy(current_polygon)
        scaled_xyxy = scale_xyxy_from_left_to_right(R, C, r, c, original_xyxy)  # 일단 640*360으로 잡자
        # original_original_xyxy = scale_xyxy_from_left_to_right(r, c, R, C, scaled_xyxy)
        # if x1_max < original_original_xyxy[0]:
        #     x1_max = original_original_xyxy[0]
        # if y1_max < original_original_xyxy[1]:
        #     y1_max = original_original_xyxy[1]
        # if x2_max < original_original_xyxy[2]:
        #     x2_max = original_original_xyxy[2]
        # if y2_max < original_original_xyxy[3]:
        #     y2_max = original_original_xyxy[3]
        danger_score_of_current_tracking_object = 0
        # print(f"original_xyxy is {original_xyxy}")
        # print(f"scaled_xyxy is {scaled_xyxy}")
        # print(f"matrix r is {len(danger_zone_matrix)}, c is {len(danger_zone_matrix[0])}")
        for r in range(scaled_xyxy[1], scaled_xyxy[3]):
            for c in range(scaled_xyxy[0], scaled_xyxy[2]):
                # print(f"r, c = {r}, {c}")
                danger_score_of_current_tracking_object += danger_zone_matrix[r][c]
                danger_zone_matrix[r][c] += DANGER_UPDATE_STRIDE
                if danger_zone_matrix[r][c] > DANGER_SCORE_MAX:
                    danger_zone_matrix[r][c] = DANGER_SCORE_MAX
        # print_danger_zone_matrix(r, c, danger_zone_matrix)
        print(f"scaled_xyxy is {scaled_xyxy}")
        w, h = scaled_xyxy[2] - scaled_xyxy[0], scaled_xyxy[3] - scaled_xyxy[1]
        if w == 0 or h == 0:
            tracking_object_list_danger_list.append(False)
            continue
        average_danger_score = danger_score_of_current_tracking_object / (w * h)
        if average_danger_score >= DANGER_THRESHOLD:
            tracking_object_list_danger_list.append(True)
        else:
            tracking_object_list_danger_list.append(False)
        # print(f"max x1 y1 x2 y2 = {x1_max},{y1_max},{x2_max},{y2_max}")

    return tracking_object_list_danger_list, danger_zone_matrix
