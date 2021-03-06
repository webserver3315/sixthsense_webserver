'''
Created by KYEONGTAE PARK on 2020/08/10.

3 5
74 86 66 90 94
90 82 8 13 74
73 53 27 52 71

4 5
79 20 55 90 44
0 39 64 16 95
80 0 24 45 41
45 90 4 3 20


'''

import heapq


# import collections


# iou_table = collections.deque(collections.deque(input() for _ in range(0, int(O)) for _ in range(0, int(B))))

def verbose_answer(B, O, hist, done):
    for o in range(O):
        if hist[o][1] == -1:
            # print(f"신규탐지객체 {o} 는 일치하는 이전탐지객체가 없어서 신규 트래킹 객체가 됩니다")
            print(f"New DO {o} does not have matching TO, so it become NEW TO")
        else:
            # print(f"신규탐지객체 {o} 는 이전탐지객체 {hist[o][1]} 에 {-hist[o][0]} 의 확신도로 append 됩니다.")
            print(f"NEW DO {o} APPENDS at {hist[o][1]}th TO with  {-hist[o][0]} IOU value")
    for b in range(B):
        if done[b] == False:
            # print(f"이전탐지객체 {b} 는 일치하는 현재탐지객체가 없어서 트래킹 영구중단합니다")
            print(f"EXISTING TO {b} STOPPED Tracking as having no child")


def is_all_blue_assigned(B, done, iou_table=[]):
    for b in range(B):
        if not done[b] and not len(iou_table[b]) == 0:
            return False
    return True


def print_hist(hist):
    print("hist is : ")
    print(hist)
    return


def print_done(done):
    # print("done is : ")
    print([int(d) for d in done])
    return


def solve(B, O, iou_table=[]):
    IOU_THRESHOLD = -40
    hist = [[1, -1] for _ in range(O)]
    done = [False for _ in range(B)]
    while not is_all_blue_assigned(B, done, iou_table):
        for b in range(B):
            if done[b] == True or len(iou_table[b]) == 0:
                continue
            heap = iou_table[b]
            front = heapq.heappop(heap)
            heapq.heapify(heap)
            o = front[1]
            p = front[0]
            if o == -1 or p > IOU_THRESHOLD:  # python 의 heap 은 최소힙만 지원되므로 반전
                continue
            if hist[o][0] > p:
                if hist[o][1] != -1:
                    done[hist[o][1]] = False
                hist[o][1] = b
                done[hist[o][1]] = True
                hist[o][0] = p
    verbose_answer(B, O, hist, done)
    return hist, done


def initial_input_1():
    iou_table = []
    B, O = list(map(int, input().split()))
    # print("\n")
    # print(f"{B},{O}")

    for o in range(B):
        # print(o)
        i = list(map(int, input().split()))
        # print(i)
        heap = []
        for oo, p in enumerate(i):
            heapq.heappush(heap, (int(-p), int(oo)))
            # print(heap)
        iou_table.append(heap)

    # print(iou_table)
    return B, O, iou_table

# # 테스트 코드
# # B, O, iou_table = initial_input()
# B, O = 4, 4
# iou_table = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.1807372175980975, 0.07689830129553214],
#              [0.0, 0.1807372175980975, 1.0, 0.4151120929471445], [0.0, 0.07689830129553214, 0.4151120929471445, 1.0]]
# B, O, iou_pair_table = make_iou_table_to_iou_pair_table(B, O, iou_table)
# print(iou_pair_table)
# # hist, done = solve(B, O, iou_table)
# # verbose_answer(B, O, hist, done)
