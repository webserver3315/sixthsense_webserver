# [[polygon(p), p, c], [polygon(pre1:latest), p, c], [polygon(pre2), p, c], [polygon(pre3), p, c], [polygon(pre4), p, c], ...]

person = 0
car = 2

from shapely.geometry import Polygon
import matplotlib.pyplot as plt

# 데이터를 받아옴
def data_in(data):
    object_data = []
    for item in data:
        object_data.append(data_preprocess(item))
    return object_data
    

# 받아온 데이터를 자료구조에 전처리함
def data_preprocess(item):
    res = {}
    res["now_pos"] = item[0][0]
    res["per"] = item[0][1]
    res["id"] = item[0][2]
    item_size = len(item)
    res["pre_pos_list"] = []
    for i in range(1, item_size):
        res["pre_pos_list"].append(item[i][0])
    res["pre_pos_data_size"] = len(res["pre_pos_list"])
    res["speed"] = speed_check(res)
    res["field_size"] = field_size_check(res)
    return res


# 오브젝트의 상대속도 구하기
def speed_check(item):
    item_size = min(5, item["pre_pos_data_size"])
    if item_size == 0:
        return (0, 0)
    oldest_pre_pos = item["pre_pos_list"][item_size-1]
    oldest_pre_pos = list(zip(*oldest_pre_pos.exterior.coords.xy))
    now_pos = item["now_pos"]
    now_pos = list(zip(*now_pos.exterior.coords.xy))
    center_point_of_oldest_pre_pos = (
        (oldest_pre_pos[0][0]+oldest_pre_pos[1][0])/2,
        (oldest_pre_pos[1][1]+oldest_pre_pos[2][1])/2
    )
    center_point_of_now_pos = (
        (now_pos[0][0]+now_pos[1][0])/2,
        (now_pos[1][1]+now_pos[2][1])/2
    )
    if (abs(center_point_of_now_pos[0]-center_point_of_oldest_pre_pos[0])+abs(center_point_of_now_pos[1]-center_point_of_oldest_pre_pos[1]))/item_size < 10:
        return(0, 0)
    return (
        (center_point_of_now_pos[0]-center_point_of_oldest_pre_pos[0])/item_size,
        (center_point_of_now_pos[1]-center_point_of_oldest_pre_pos[1])/item_size
    )


def field_size_check(item):
    now_pos = list(zip(*item["now_pos"].exterior.coords.xy))
    width = now_pos[1][0]-now_pos[0][0]
    height = now_pos[2][1]-now_pos[1][1]
    return width*height

# 각 오브젝트별로 계산을 수행 (n/4)
def calculation(object_data):
    object_data_size = len(object_data)
    car_list = []
    person_list = []
    res_list = []

    for i in range(object_data_size):
        if object_data[i]["id"] == car:
            car_list.append(object_data[i])
        elif object_data[i]["id"] == person:
            person_list.append(object_data[i])
    for i in car_list:
        for j in person_list:
            res_list.append(accident_simulation(i, j)/j["field_size"])
    return res_list


# 각 오브젝트별로 계산을 수행 matrix를 리턴(n/4)
def calculation_matrix(object_data):
    object_data_size = len(object_data)
    car_list = []
    person_list = []

    for i in range(object_data_size):
        if object_data[i]["id"] == car:
            car_list.append(object_data[i])
        elif object_data[i]["id"] == person:
            person_list.append(object_data[i])

    car_list_size = len(car_list)
    person_list_size = len(person_list)
    res_list = [[-1 for i in range(person_list_size)] for i in range(car_list_size)]

    for i in range(car_list_size):
        for j in range(person_list_size):
            car_object = car_list[i]
            person_object = person_list[j]
            res_list[i][j] = accident_simulation(car_object, person_object)/person_object["field_size"]

    return res_list


# 각 오브젝트별로 계산을 수행 matrix를 리턴(n/4)
def calculation_matrix_debug(object_data):
    object_data_size = len(object_data)
    car_list = []
    person_list = []
    res_car_list = []

    for i in range(object_data_size):
        if object_data[i]["id"] == car:
            car_list.append(object_data[i])
            res_car_list.append(i)
        elif object_data[i]["id"] == person:
            person_list.append(object_data[i])

    car_list_size = len(car_list)
    person_list_size = len(person_list)
    res_list = [[-1 for i in range(person_list_size)] for i in range(car_list_size)]

    for i in range(car_list_size):
        for j in range(person_list_size):
            car_object = car_list[i]
            person_object = person_list[j]
            res_list[i][j] = accident_simulation(car_object, person_object)/person_object["field_size"]

    return res_car_list, res_list


# 예상 최대 intersection percentage 계산
def accident_simulation(object_car, object_person):
    object_car_now_pos = list(zip(*object_car["now_pos"].exterior.coords.xy))
    object_person_now_pos = list(zip(*object_person["now_pos"].exterior.coords.xy))
    
    # w
    object_car_x_speed = object_car["speed"][0]
    object_person_x_speed = object_person["speed"][0]
    object_car_now_pos_x_left = object_car_now_pos[0][0]
    object_car_now_pos_x_right = object_car_now_pos[1][0]
    object_person_now_pos_x_left = object_person_now_pos[0][0]
    object_person_now_pos_x_right = object_person_now_pos[1][0]
    x_crash_time_list = []
    
    # 차의 w 왼쪽점과 사람의 w 왼쪽점 충돌시간 계산
    crash_time_left_left = cross_line(
        object_car_now_pos_x_left, object_person_now_pos_x_left,
        object_car_x_speed, object_person_x_speed
    )
    if crash_time_left_left == "INF":
        return 0
    x_crash_time_list.append([crash_time_left_left, "left_left"])

    # 차의 w 왼쪽점과 사람의 w 오른쪽점 충돌시간 계산
    crash_time_left_right = cross_line(
        object_car_now_pos_x_left, object_person_now_pos_x_right,
        object_car_x_speed, object_person_x_speed
    )
    if crash_time_left_right == "INF":
        return 0
    x_crash_time_list.append([crash_time_left_right, "left_right"])

    # 차의 w 오른쪽점과 사람의 w 왼쪽점 충돌시간 계산
    crash_time_right_left = cross_line(
        object_car_now_pos_x_right, object_person_now_pos_x_left,
        object_car_x_speed, object_person_x_speed
    )
    if crash_time_right_left == "INF":
        return 0
    x_crash_time_list.append([crash_time_right_left, "right_left"])

    # 차의 w 오른쪽점과 사람의 w 오른쪽점 충돌시간 계산
    crash_time_right_right = cross_line(
        object_car_now_pos_x_right, object_person_now_pos_x_right,
        object_car_x_speed, object_person_x_speed
    )
    if crash_time_right_right == "INF":
        return 0
    x_crash_time_list.append([crash_time_right_right, "right_right"])

    # 차를 고정시킨 상태의 상대속력
    x_res = x_function(
        crash_time_left_left,
        crash_time_left_right,
        crash_time_right_left,
        crash_time_right_right,
        abs(object_person_x_speed-object_car_x_speed),
        min(
            object_car_now_pos_x_right-object_car_now_pos_x_left,
            object_person_now_pos_x_right-object_person_now_pos_x_left
        )
    )

    # h
    object_car_y_speed = object_car["speed"][1]
    object_person_y_speed = object_person["speed"][1]
    object_car_now_pos_y_down = object_car_now_pos[2][1]
    object_car_now_pos_y_up = object_car_now_pos[1][1]
    object_person_now_pos_y_down = object_person_now_pos[2][1]
    object_person_now_pos_y_up = object_person_now_pos[1][1]
    y_crash_time_list = []
    
    # 차의 h 아래쪽점과 사람의 h 아래쪽점 충돌시간 계산
    crash_time_down_down = cross_line(
        object_car_now_pos_y_down, object_person_now_pos_y_down,
        object_car_y_speed, object_person_y_speed
    )
    if crash_time_down_down == "INF":
        return 0
    y_crash_time_list.append([crash_time_down_down, "down_down"])

    # 차의 h 아래쪽점과 사람의 h 위쪽점 충돌시간 계산
    crash_time_down_up = cross_line(
        object_car_now_pos_y_down, object_person_now_pos_y_up,
        object_car_y_speed, object_person_y_speed
    )
    if crash_time_down_up == "INF":
        return 0
    y_crash_time_list.append([crash_time_down_up, "down_up"])

    # 차의 h 위쪽점과 사람의 h 아래쪽점 충돌시간 계산
    crash_time_up_down = cross_line(
        object_car_now_pos_y_up, object_person_now_pos_y_down,
        object_car_y_speed, object_person_y_speed
    )
    if crash_time_up_down == "INF":
        return 0
    y_crash_time_list.append([crash_time_up_down, "up_down"])

    # 차의 h 위쪽점과 사람의 h 위쪽점 충돌시간 계산
    crash_time_up_up = cross_line(
        object_car_now_pos_y_up, object_person_now_pos_y_up,
        object_car_y_speed, object_person_y_speed
    )
    if crash_time_up_up == "INF":
        return 0
    y_crash_time_list.append([crash_time_up_up, "up_up"])

    # 차가 정지 상태이면 확률 0 리턴
    if object_car_x_speed == 0 and object_car_y_speed == 0:
        return 0

    # 차를 고정시킨 상태의 상대속도
    y_res = y_function(
        crash_time_down_down,
        crash_time_down_up,
        crash_time_up_down,
        crash_time_up_up,
        abs(object_person_y_speed-object_car_y_speed),
        min(
            object_car_now_pos_y_down-object_car_now_pos_y_up,
            object_person_now_pos_y_down-object_person_now_pos_y_up
        )
    )

    x_crash_time_list.sort()
    y_crash_time_list.sort()
    total_crash_time_list = x_crash_time_list+y_crash_time_list
    total_crash_time_list.sort()

    point_of_total = 0
    point_of_x = 0
    point_of_y = 0
    res_func = []

    # x, y 양 쪽 리스트에서 시간 순으로 먼저인 사건부터 처리
    while point_of_x < 4 and point_of_y < 4:
        res_func.append([merge_function(x_res[point_of_x][0], y_res[point_of_y][0])])
        # x 리스트에 있는 값일 경우
        if total_crash_time_list[point_of_total][1] == x_crash_time_list[point_of_x][1]:
            point_of_x += 1
        # y 리스트에 있는 값일 경우
        else:
            point_of_y += 1
        point_of_total += 1
    
    # x 리스트 남은 값 처리
    while point_of_x < 4:
        res_func.append([merge_function(x_res[point_of_x][0], y_res[point_of_y][0])])
        point_of_x += 1

    # y 리스트 남은 값 처리
    while point_of_y < 4:
        res_func.append([merge_function(x_res[point_of_x][0], y_res[point_of_y][0])])
        point_of_y += 1
    res_func.append([merge_function(x_res[point_of_x][0], y_res[point_of_y][0])])
    

    # res_func에 범위 추가
    res_func_size = len(res_func)
    res_func[0].append("-INF")
    res_func[0].append(total_crash_time_list[0][0])
    for i in range(1, res_func_size-1):
        res_func[i].append(total_crash_time_list[i-1][0])
        res_func[i].append(total_crash_time_list[i][0])
    res_func[res_func_size-1].append(total_crash_time_list[res_func_size-2][0])
    res_func[res_func_size-1].append("INF")

    # 9개의 구간별 최대 2차 1변수 식을 미분해서 최댓 값 구하기
    res_max = 0
    cal_limit = 15 # 사고 확률 계산 유효 범위

    for poly in res_func:
        # 계산 범위에 아예 포함되지 않는 경우 (0 <= t <= car_limit)
        if (poly[1] != "-INF" and poly[1] > cal_limit) or (poly[2] != "INF" and poly[2] < 0):
            res_max = max(res_max, 0)
            continue
        # 왼쪽 끝 범위가 음수이면(일부 범위에서 이미 충돌했을 때) 0으로 옮기기
        if poly[1] == "-INF" or poly[1] < 0:
            poly[1] = 0
        # 오른쪽 끝 범위가 너무 멀 때(충돌 예상까지 너무 긴 시간일 때) car_limit로 옮기기
        if poly[2] == "INF" or poly[2] > cal_limit:
            poly[2] = cal_limit
        
        res_max = max(res_max, diff(poly))

    return res_max


# 한 점과 방향벡터가 두 쌍이 주어질 때, 두 직선의 교점까지 걸리는 시간을 리턴
def cross_line(car_point, person_point, car_speed, person_speed):
    # 영원히 만나지 않을 때
    if car_speed == person_speed:
        return "INF"
    # 이미 만났거나 만날 예정일 때의 시간
    crash_time = (person_point-car_point)/(car_speed-person_speed)
    return crash_time


def x_function(
    left_left,
    left_right,
    right_left,
    right_right,
    x_speed,
    width
):
    timestamp_list = [
        [left_left, "left_left"],
        [left_right, "left_right"],
        [right_left, "right_left"],
        [right_right, "right_right"]
    ]
    timestamp_list.sort()

    x_function_with_timestamp = []
    
    # t <= timestamp1
    x_function_with_timestamp.append([[0, 0, 0], "-INF", timestamp_list[0][1]])

    # timestamp1 <= t <= timestamp2
    x_function_with_timestamp.append([[0, x_speed, -timestamp_list[0][0]*x_speed], timestamp_list[0][1], timestamp_list[1][1]])

    # timestamp2 <= t <= timestamp3
    x_function_with_timestamp.append([[0, 0, width], timestamp_list[1][1], timestamp_list[2][1]])

    # timestamp3 <= t <= timestamp4
    x_function_with_timestamp.append([[0, -x_speed, x_speed*timestamp_list[3][0]], timestamp_list[2][1], timestamp_list[3][1]])

    # timestamp4 <= t
    x_function_with_timestamp.append([[0, 0, 0], timestamp_list[3][1], "INF"])

    return x_function_with_timestamp


def y_function(
    down_down,
    down_up,
    up_down,
    up_up,
    y_speed,
    height
):
    timestamp_list = [
        [down_down, "down_down"],
        [down_up, "down_up"],
        [up_down, "up_down"],
        [up_up, "up_up"]
    ]
    timestamp_list.sort()

    y_function_with_timestamp = []
    
    # t <= timestamp1
    y_function_with_timestamp.append([[0, 0, 0], "-INF", timestamp_list[0][1]])

    # timestamp1 <= t <= timestamp2
    y_function_with_timestamp.append([[0, y_speed, -timestamp_list[0][0]*y_speed], timestamp_list[0][1], timestamp_list[1][1]])

    # timestamp2 <= t <= timestamp3
    y_function_with_timestamp.append([[0, 0, height], timestamp_list[1][1], timestamp_list[2][1]])

    # timestamp3 <= t <= timestamp4
    y_function_with_timestamp.append([[0, -y_speed, y_speed*timestamp_list[3][0]], timestamp_list[2][1], timestamp_list[3][1]])

    # timestamp4 <= t
    y_function_with_timestamp.append([[0, 0, 0], timestamp_list[3][1], "INF"])

    return y_function_with_timestamp


# 두 개의 일차방정식을 곱해서 리턴
def merge_function(poly1, poly2):
    return [
        poly1[1]*poly2[1],
        poly1[1]*poly2[2]+poly1[2]*poly2[1],
        poly1[2]*poly2[2]
    ]


# 이차식을 미분해서 최댓 값 리턴
def diff(poly):
    num = poly[0]
    if num[0] == 0:
        if poly[1] == "-INF" or poly[2] == "INF":
            return 0
        else:
            return max(
                num[1]*poly[1]+num[2],
                num[1]*poly[2]+num[2]
            )
    target_time = -num[1]/(2*num[0])
    if poly[1] == "-INF":
        if target_time <= poly[2]:
            return num[0]*target_time*target_time+num[1]*target_time+num[2]
        else:
            return num[0]*poly[2]*poly[2]+num[1]*poly[2]+num[2]
    if poly[2] == "INF":
        if poly[1] <= target_time:
            return num[0]*target_time*target_time+num[1]*target_time+num[2]
        else:
            return num[0]*poly[1]*poly[1]+num[1]*poly[1]+num[2]
    if poly[1] <= target_time and target_time <= poly[2]:
        return max(
            num[0]*target_time*target_time+num[1]*target_time+num[2],
            num[0]*poly[1]*poly[1]+num[1]*poly[1]+num[2],
            num[0]*poly[2]*poly[2]+num[1]*poly[2]+num[2]
        )
    else:
        return max(
            num[0]*poly[1]*poly[1]+num[1]*poly[1]+num[2],
            num[0]*poly[2]*poly[2]+num[1]*poly[2]+num[2]
        )


# 모든 과정을 수행 후 리턴
def accident_percentage(data):
    object_data = data_in(data)
    result = calculation_matrix(object_data)
    return result


# 모든 과정을 수행 후 리턴(디버그용)
def accident_percentage_debug(data):
    object_data = data_in(data)
    result_car, result = calculation_matrix_debug(object_data)
    return object_data, result_car, result


def xyxy_to_polygon(xyxy: list):
#     print(f"xyxy is {xyxy}")
    x1=int(xyxy[0])
    y1=int(xyxy[1])
    x2=int(xyxy[2])
    y2=int(xyxy[3])
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


if __name__ == "__main__":
    car_tracking_obj_list = [[xyxy_to_polygon([10+x, 10, 60+x, 40]), 1, 2] for x in range(0, 91, 30)][::-1]
    person_tracking_obj_list = [[xyxy_to_polygon([300, 60-x, 320, 90-x]), 1, 0] for x in range(0, 31, 10)][::-1]
    person_tracking_obj_list1 = [[xyxy_to_polygon([240-x, 60-x, 260-x, 90-x]), 1, 0] for x in range(0, 31, 10)][::-1]
    person_tracking_obj_list2 = [[xyxy_to_polygon([140+x, 60-x, 160+x, 90-x]), 1, 0] for x in range(0, 31, 10)][::-1]
    res = accident_percentage([car_tracking_obj_list, person_tracking_obj_list, person_tracking_obj_list1, person_tracking_obj_list2])
    print(res)
    #for i in res:
    #    print(f"{i*100}%", end=" ")