import numpy as np

DANGER_UPDATE_STRIDE = 3000  # 훑고지나갈때마다 상승하는 위험도
DANGER_THRESHOLD = 500  # 위험역치
DANGER_SCORE_MAX = 5000  # 최대 저장할 수 있는 위험점수
DANGER_DEGRADE_STRIDE = 5  # 한 프레임마다 경감되는 위험도
SPEED_THRESHOLD = 5

np.set_printoptions(threshold=np.inf)
danger_zone_matrix = np.full((10, 10), 0, dtype=int)
# for r in range(1, 3):
#     for c in range(2, 4):
#         danger_zone_matrix[r][c] += DANGER_UPDATE_STRIDE
#         if danger_zone_matrix[r][c] > DANGER_SCORE_MAX:
#             danger_zone_matrix[r][c] = DANGER_SCORE_MAX

print(f"danger_zone_matrix = {danger_zone_matrix[1:3]}")
danger_zone_matrix[1:3, 2:4] += DANGER_UPDATE_STRIDE
danger_zone_matrix[2:4, 3:5] += DANGER_UPDATE_STRIDE
danger_zone_matrix[danger_zone_matrix > DANGER_SCORE_MAX] = DANGER_SCORE_MAX
print(f"{danger_zone_matrix}\n")

danger_zone_matrix = danger_zone_matrix - DANGER_DEGRADE_STRIDE
less_than_0 = danger_zone_matrix < 0

print(f"less_than_0 : {less_than_0}")
# danger_zone_matrix[less_than_0] = 0

print(f"{danger_zone_matrix}\n")
