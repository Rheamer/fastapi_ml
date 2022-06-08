from typing import *


def isPrintable(tg: List[List[int]]) -> bool:
    cubes = {}

    if len(tg) == 0 or len(tg[0]) != len(tg):
        return False

    for i in range(len(tg)):
        for j in range(len(tg[0])):
            if tg[i][j] not in cubes:
                cubes[tg[i][j]] = [[i, j], [i, j]]
            elif j < cubes[tg[i][j]][0][1]:
                cubes[tg[i][j]][0][1] = j
            elif i > cubes[tg[i][j]][1][0]:
                cubes[tg[i][j]][1][0] = i
            elif j > cubes[tg[i][j]][1][1]:
                cubes[tg[i][j]][1][1] = j

    containers = {}
    for color, _ in cubes.items():
        containers[color] = set()

    for color, cube in cubes.items():
        for i in range(cube[0][0], cube[1][0] + 1):
            for j in range(cube[0][1], cube[1][1] + 1):
                if tg[i][j] != color:
                    containers[color].add(tg[i][j])

    for color, container in containers.items():
        for containedColor in container:
            if color in containers[containedColor]:
                return False
    count = 0
    for color, container in containers.items():
        if len(container) == 0:
            count += 1
    if count == 0:
        return False

    return True