import numpy as np
from cv2 import *
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


def initialize_danger_zone_matrix(original_img):
    t0 = time.time()
    ORIGINAL_R, ORIGINAL_C = int(original_img.shape[0]), int(original_img.shape[1])
    DANGER_ZONE_MATRIX_R, DANGER_ZONE_MATRIX_C = int(ORIGINAL_R / 4), int(ORIGINAL_C / 4)
    danger_zone_matrix = np.full((DANGER_ZONE_MATRIX_R, DANGER_ZONE_MATRIX_C), 0, dtype=int)
    print('initialize_danger_zone_matrix: (%.3fs)' % (time.time() - t0))
    return danger_zone_matrix


def automatic_visualize_danger_zone_matrix(original_img, danger_zone_matrix):
    ORIGINAL_R, ORIGINAL_C = int(original_img.shape[0]), int(original_img.shape[1])
    DANGER_ZONE_MATRIX_R, DANGER_ZONE_MATRIX_C = int(ORIGINAL_R / 4), int(ORIGINAL_C / 4)
    visualize_danger_zone_matrix(original_img, ORIGINAL_R, ORIGINAL_C, DANGER_ZONE_MATRIX_R, DANGER_ZONE_MATRIX_C,
                                 danger_zone_matrix)
    return original_img


def visualize_danger_zone_matrix(img, ORIGINAL_R, ORIGINAL_C, DANGER_ZONE_MATRIX_R, DANGER_ZONE_MATRIX_C,
                                 danger_zone_matrix):
    # default = 720 1280 180 320
    '''
    for rr in range(DANGER_ZONE_MATRIX_R):
        for cc in range(DANGER_ZONE_MATRIX_C):
            if danger_zone_matrix[rr][cc] == 0:
                continue
            xyxy = [cc, rr, cc + 1, rr + 1]
            original_xyxy = scale_xyxy_from_left_to_right(DANGER_ZONE_MATRIX_R, DANGER_ZONE_MATRIX_C, ORIGINAL_R,
                                                          ORIGINAL_C, xyxy)
            # print(f"original_xyxy is {original_xyxy}")

            # First we crop the sub-rect from the image _ https://stackoverflow.com/questions/56472024/how-to-change-the-opacity-of-boxes-cv2-rectangle
            x, y = original_xyxy[0], original_xyxy[1],
            w, h = original_xyxy[2] - original_xyxy[0], original_xyxy[3] - original_xyxy[1]
            alpha = danger_zone_matrix[rr][cc] * 0.0002
            sub_img = img[y:y + h, x:x + w]
            white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
            res = cv2.addWeighted(sub_img, 1 - alpha * 0.35, white_rect, alpha * 0.35, 0)
            # Putting the image back to its position
            img[y:y + h, x:x + w] = res
    '''
    return img


def print_danger_zone_matrix(r, c, danger_zone_matrix):
    for rr in range(r):
        print(f"{danger_zone_matrix[rr]}")


def degrade_danger_zone_matrix(danger_zone_matrix, DANGER_DEGRADE_STRIDE):
    # DANGER_DEGRADE_STRIDE 만큼 빼지만, 위험계수가 음수라면 0으로 대체
    t0 = time.time()
    # np.set_printoptions(threshold=np.inf)

    # 검증된 1줄 numpy 코드 41.977s
    danger_zone_matrix[danger_zone_matrix >= DANGER_DEGRADE_STRIDE] -= DANGER_DEGRADE_STRIDE

    # 검증된 3줄 numpy 코드 41.333s
    # danger_zone_matrix -= DANGER_DEGRADE_STRIDE
    # less_than_0 = danger_zone_matrix < 0
    # danger_zone_matrix[less_than_0] = 0
    # print(f"danger_zone_matrix : {danger_zone_matrix}")
    # print(f"less_than_0 : {less_than_0}")

    # 검증된 2중 for 문 코드 48.090s
    # print(f"{danger_zone_matrix.shape[0]} * {danger_zone_matrix.shape[1]}")
    # for r in range(danger_zone_matrix.shape[0]):
    #     for c in range(danger_zone_matrix.shape[1]):
    #         danger_zone_matrix[r][c] -= DANGER_DEGRADE_STRIDE
    #         if danger_zone_matrix[r][c] < 0:
    #             danger_zone_matrix[r][c] = 0
    print('degrade_danger_zone_matrix: (%.3fs)' % (time.time() - t0))
    return danger_zone_matrix


# 1. TOL 과 DZM 을 비교대조하여, 위험한 TO 여부를 반환한다.
# 2. TOL 에서 Class id 가 2 인 객체가 훑고 지나간 영역의 DZM 을 가산한다. 아니면 감산한다.
# 3. DZM 은 r(세로)*c(가로) 의 2차원 int 행렬이다. r, c에 일단 기본값으로 360, 640 권장
def is_tracking_object_list_dangerous(ORIGINAL_R, ORIGINAL_C, DANGER_ZONE_MATRIX_R, DANGER_ZONE_MATRIX_C,
                                      tracking_object_list, danger_zone_matrix):

    DANGER_UPDATE_STRIDE = 3000  # 훑고지나갈때마다 상승하는 위험도
    DANGER_THRESHOLD = 500  # 위험역치
    DANGER_SCORE_MAX = 5000  # 최대 저장할 수 있는 위험점수
    DANGER_DEGRADE_STRIDE = 5  # 한 프레임마다 경감되는 위험도
    SPEED_THRESHOLD = 5

    danger_zone_matrix = degrade_danger_zone_matrix(danger_zone_matrix, DANGER_DEGRADE_STRIDE)
    tracking_object_list_danger_list = []  # bool형 1차원 배열. 각 tracking object 의 위험구역위치여부를 의미.
    for b, tracking_object in enumerate(tracking_object_list):
        # 축소좌표 따기
        current_polygon = tracking_object[0][0]
        original_xyxy = polygon_to_xyxy(current_polygon)
        scaled_xyxy = scale_xyxy_from_left_to_right(ORIGINAL_R, ORIGINAL_C, DANGER_ZONE_MATRIX_R, DANGER_ZONE_MATRIX_C,
                                                    original_xyxy)
        danger_score_of_current_tracking_object = 0
        if tracking_object[0][2] == 0:  # 사람이면, 위험점수만 판별
            for r in range(scaled_xyxy[1], scaled_xyxy[3]):
                for c in range(scaled_xyxy[0], scaled_xyxy[2]):
                    danger_score_of_current_tracking_object += danger_zone_matrix[r][c]
        elif get_speed(tracking_object) > SPEED_THRESHOLD:  # 속도붙은 자동차라면, 따로 위험점수는 안찍는다. 행렬만 가산할 뿐
            danger_zone_matrix[scaled_xyxy[1]:scaled_xyxy[3], scaled_xyxy[0]:scaled_xyxy[2]] += DANGER_UPDATE_STRIDE
            danger_zone_matrix[danger_zone_matrix > DANGER_SCORE_MAX] = DANGER_SCORE_MAX
            # danger_zone_matrix[scaled_xyxy[1]:scaled_xyxy[3]+1][scaled_xyxy[0]:scaled_xyxy[2]+1] += DANGER_UPDATE_STRIDE
            # danger_zone_matrix = np.where(danger_zone_matrix > DANGER_SCORE_MAX, danger_zone_matrix, DANGER_SCORE_MAX)
        w, h = scaled_xyxy[2] - scaled_xyxy[0], scaled_xyxy[3] - scaled_xyxy[1]
        if w == 0 or h == 0:
            tracking_object_list_danger_list.append(0)
            continue
        average_danger_score = danger_score_of_current_tracking_object / (w * h)
        if average_danger_score >= DANGER_THRESHOLD:
            tracking_object_list_danger_list.append(int(average_danger_score))
        else:
            tracking_object_list_danger_list.append(0)
    return tracking_object_list_danger_list, danger_zone_matrix
