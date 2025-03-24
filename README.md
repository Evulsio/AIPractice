from copy import deepcopy
import numpy as np
import time
import sys

# 현재 상태들의 정보를 받아 Goal State로 가는 최적 경로를 반환하는 함수
def bestsolution(state):
    bestsol = np.array([], int).reshape(-1, 9)
    count = len(state) - 1
    while count != -1:
        bestsol = np.insert(bestsol, 0, state[count]['puzzle'], 0)
        count = (state[count]['parent'])
    return bestsol.reshape(-1, 3, 3)


# 특정 상태가 이전에 탐색된 적이 있는지(고유한지) 확인하는 함수
def all(checkarray):
    set=[]
    for it in set:
        for checkarray in it:
            return 1
        else:
            return 0


# 퍼즐(현재 상태)와 목표 상태 사이의 맨해튼 거리(Manhattan distance)를 계산하는 함수
def manhattan(puzzle, goal):
    a = abs(puzzle // 3 - goal // 3)
    b = abs(puzzle % 3 - goal % 3)
    mhcost = a + b
    return sum(mhcost[1:])


# 퍼즐(현재 상태)와 목표 상태 사이에서 위치가 다른 타일 수를 계산하는 함수
def misplaced_tiles(puzzle, goal):
    mscost = np.sum(puzzle != goal) - 1
    return mscost if mscost > 0 else 0


# 각 숫자(0~8)가 퍼즐에서 어디 위치하는지를 나타내는 좌표 정보를 반환하는 함수
def coordinates(puzzle):
    pos = np.array(range(9))
    for p, q in enumerate(puzzle):
        pos[q] = p
    return pos


# 8퍼즐을 맨해튼 거리 휴리스틱으로 탐색하여 해를 찾는 함수
def evaluvate(puzzle, goal):
    # 이동 가능한 방향(위, 아래, 왼쪽, 오른쪽) 정의
    steps = np.array([
        ('up', [0, 1, 2], -3),
        ('down', [6, 7, 8], 3),
        ('left', [0, 3, 6], -1),
        ('right', [2, 5, 8], 1)
    ], dtype=[('move', str, 1), ('position', list), ('head', int)])

    # 상태 정보를 저장할 때의 데이터 구조 (퍼즐 상태, 부모 인덱스, g(n), h(n))
    dtstate = [('puzzle', list), ('parent', int), ('gn', int), ('hn', int)]
    
    # 초기 상태 설정 (부모 = -1, g(n)=0, h(n)은 맨해튼 거리로 계산)
    costg = coordinates(goal)
    parent = -1
    gn = 0
    hn = manhattan(coordinates(puzzle), costg)
    state = np.array([(puzzle, parent, gn, hn)], dtstate)

    # 우선순위 큐: (상태의 위치, f(n)) 형태
    dtpriority = [('position', int), ('fn', int)]
    priority = np.array([(0, hn)], dtpriority)

    while True:
        # 우선순위 큐를 f(n), position 순으로 정렬하여 가장 작은 f(n)을 가진 상태 선택
        priority = np.sort(priority, kind='mergesort', order=['fn', 'position'])
        position, fn = priority[0]
        # 선택된 상태(맨 앞)를 큐에서 제거
        priority = np.delete(priority, 0, 0)

        puzzle, parent, gn, hn = state[position]
        puzzle = np.array(puzzle)
        # 현재 퍼즐에서 빈 칸(0)의 위치 확인
        blank = int(np.where(puzzle == 0)[0][0])
        # g(n)을 1 증가 (한 단계 이동)
        gn = gn + 1
        c = 1
        start_time = time.time()

        # 가능한 방향들(위, 아래, 왼쪽, 오른쪽)에 대해 새 상태를 생성
        for s in steps:
            c = c + 1
            if blank not in s['position']:
                # 현재 퍼즐 상태를 복사
                openstates = deepcopy(puzzle)
                # 빈 칸과 이동할 칸을 교환
                openstates[blank], openstates[blank + s['head']] = openstates[blank + s['head']], openstates[blank]
                # 이전에 탐색된 적이 없는 상태인지 확인
                if ~(np.all(list(state['puzzle']) == openstates, 1)).any():
                    end_time = time.time()
                    # 시간 제한(예: 2초)을 두어 무한 탐색 방지
                    if (end_time - start_time) > 2:
                        print(" The 8 puzzle is unsolvable ! \n")
                        sys.exit()

                    # 맨해튼 거리 휴리스틱 비용 계산
                    hn = manhattan(coordinates(openstates), costg)
                    # 새로운 상태를 state에 추가
                    q = np.array([(openstates, position, gn, hn)], dtstate)
                    state = np.append(state, q, 0)
                    # f(n) = g(n) + h(n)
                    fn = gn + hn
                    # 우선순위 큐에도 추가
                    q = np.array([(len(state) - 1, fn)], dtpriority)
                    priority = np.append(priority, q, 0)
                    # 목표 상태와 동일한지 확인
                    if np.array_equal(openstates, goal):
                        print(' The 8 puzzle is solvable ! \n')
                        return state, len(priority)

    return state, len(priority)


# 8퍼즐을 Misplaced Tiles 휴리스틱으로 탐색하여 해를 찾는 함수
def evaluvate_misplaced(puzzle, goal):
    # 이동 가능한 방향(위, 아래, 왼쪽, 오른쪽) 정의
    steps = np.array([
        ('up', [0, 1, 2], -3),
        ('down', [6, 7, 8], 3),
        ('left', [0, 3, 6], -1),
        ('right', [2, 5, 8], 1)
    ], dtype=[('move', str, 1), ('position', list), ('head', int)])

    # 상태 정보를 저장할 때의 데이터 구조 (퍼즐 상태, 부모 인덱스, g(n), h(n))
    dtstate = [('puzzle', list), ('parent', int), ('gn', int), ('hn', int)]

    # 초기 상태 설정 (부모 = -1, g(n)=0, h(n)은 misplaced_tiles로 계산)
    costg = coordinates(goal)
    parent = -1
    gn = 0
    hn = misplaced_tiles(coordinates(puzzle), costg)
    state = np.array([(puzzle, parent, gn, hn)], dtstate)

    # 우선순위 큐: (상태의 위치, f(n)) 형태
    dtpriority = [('position', int), ('fn', int)]
    priority = np.array([(0, hn)], dtpriority)

    while True:
        # 우선순위 큐를 f(n), position 순으로 정렬하여 가장 작은 f(n)을 가진 상태 선택
        priority = np.sort(priority, kind='mergesort', order=['fn', 'position'])
        position, fn = priority[0]
        # 선택된 상태(맨 앞)를 큐에서 제거
        priority = np.delete(priority, 0, 0)

        puzzle, parent, gn, hn = state[position]
        puzzle = np.array(puzzle)
        # 현재 퍼즐에서 빈 칸(0)의 위치 확인
        blank = int(np.where(puzzle == 0)[0][0])
        # g(n)을 1 증가
        gn = gn + 1
        c = 1
        start_time = time.time()

        # 가능한 방향들(위, 아래, 왼쪽, 오른쪽)에 대해 새 상태를 생성
        for s in steps:
            c = c + 1
            if blank not in s['position']:
                # 현재 퍼즐 상태를 복사
                openstates = deepcopy(puzzle)
                # 빈 칸과 이동할 칸을 교환
                openstates[blank], openstates[blank + s['head']] = openstates[blank + s['head']], openstates[blank]
                # 이전에 탐색된 적이 없는 상태인지 확인
                if ~(np.all(list(state['puzzle']) == openstates, 1)).any():
                    end_time = time.time()
                    # 시간 제한(예: 2초)을 두어 무한 탐색 방지
                    if (end_time - start_time) > 2:
                        print(" The 8 puzzle is unsolvable \n")
                        break

                    # Misplaced Tiles 휴리스틱 비용 계산
                    hn = misplaced_tiles(coordinates(openstates), costg)
                    # 새로운 상태를 state에 추가
                    q = np.array([(openstates, parent, gn, hn)], dtstate)
                    state = np.append(state, q, 0)
                    # f(n) = g(n) + h(n)
                    fn = gn + hn
                    # 우선순위 큐에도 추가
                    q = np.array([(len(state) - 1, fn)], dtpriority)
                    priority = np.append(priority, q, 0)
                    # 목표 상태와 동일한지 확인
                    if np.array_equal(openstates, goal):
                        print(' The 8 puzzle is solvable \n')
                        return state, len(priority)

    return state, len(priority)


# ---------- 메인(실행) 부분 ----------

# 시작 상태 입력
puzzle = []
print(" 0~8 사이의 숫자를 입력하세요 (시작 상태): ")
for i in range(0, 9):
    x = int(input("값 입력: "))
    puzzle.append(x)

# 목표 상태 입력
goal = []
print(" 0~8 사이의 숫자를 입력하세요 (목표 상태): ")
for i in range(0, 9):
    x = int(input("값 입력: "))
    goal.append(x)

n = int(input("1. 맨해튼 거리 휴리스틱 \n2. Misplaced Tiles 휴리스틱\n번호 선택: "))

if n == 1:
    state, visited = evaluvate(puzzle, goal)
    bestpath = bestsolution(state)
    print(str(bestpath).replace('[', ' ').replace(']', ''))
    totalmoves = len(bestpath) - 1
    print('Goal까지의 이동 횟수:', totalmoves)
    visit = len(state) - visited
    print('방문한 노드 수: ', visit, "\n")
    print('생성된 노드 총 수:', len(state))

elif n == 2:
    state, visited = evaluvate_misplaced(puzzle, goal)
    bestpath = bestsolution(state)
    print(str(bestpath).replace('[', ' ').replace(']', ''))
    totalmoves = len(bestpath) - 1
    print('Goal까지의 이동 횟수:', totalmoves)
    visit = len(state) - visited
    print('방문한 노드 수: ', visit, "\n")
    print('생성된 노드 총 수:', len(state))

    print("Puzzle 상태:", puzzle)
    print("np.where 결과:", np.where(puzzle == 0))
    print("np.where[0] 결과:", np.where(puzzle == 0)[0])

# 퍼즐이 해결 가능한지 확인하는 함수를 가정 (is_solvable)
# 실제로 구현되지 않았으므로 사용 시 에러 발생 가능
# if not is_solvable(puzzle):
#     sys.exit("이 퍼즐은 해결할 수 없습니다.")

# 추가적으로 필요하다면 아래와 같이 사용:
# state, visited = evaluvate(puzzle, goal)
