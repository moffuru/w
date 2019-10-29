import cv2
import os
import glob
import numpy as np
import pickle
import wa_utils
import argparse
from estimator import Estimator
import wa_constants

parser = argparse.ArgumentParser()
parser.add_argument('movie')
parser.add_argument('output_puyofu_file')
parser.add_argument('model_for_empty')
parser.add_argument('model_for_color')
parser.add_argument('-s', '--start_pos', type=int, default=0)
args = parser.parse_args()

PUYO_IMAGE_LIST = {}
PUYO_FEATURE_LIST = {}

estimator = Estimator(
    pickle.load(open(args.model_for_empty, 'rb')),
    pickle.load(open(args.model_for_color, 'rb'))
)


def recognize_next(image):
    return estimator.estimate_puyo(image)


def recognize_next_pairs(image):
    res = ''
    res += recognize_next(wa_utils.filter_for_next1_puyo1(image))
    res += recognize_next(wa_utils.filter_for_next1_puyo2(image))
    res += recognize_next(wa_utils.filter_for_next2_puyo1(image))
    res += recognize_next(wa_utils.filter_for_next2_puyo2(image))
    return res


def recognize_field(image):
    puyo_field = []
    for y in range(wa_constants.FIELD_HEIGHT):
        puyo_col = []
        for x in range(wa_constants.FIELD_WIDTH):
            rx = wa_constants.OFFSET_X + wa_constants.REAL_PUYO_WIDTH * x
            ry = wa_constants.OFFSET_Y + wa_constants.REAL_PUYO_HEIGHT * y
            cropped = wa_utils.filter_for_puyo(image[ry: ry + wa_constants.REAL_PUYO_HEIGHT, rx: rx + wa_constants.REAL_PUYO_WIDTH])
            puyo_col.append(estimator.estimate_puyo(cropped))
        puyo_field.append(puyo_col)

    puyo_field_str = []
    for col in puyo_field:
        puyo_field_str.append(''.join(col))

    return puyo_field_str


cap = cv2.VideoCapture(args.movie)
go = wa_utils.filter_for_go(cv2.imread(f'../resources/rec/8488.png'))
next_move = wa_utils.filter_for_next_move(cv2.imread(f'../resources/rec/9000.png'))
next_move2 = wa_utils.filter_for_next_move(cv2.imread(f'../resources/rec/3926.png'))
next_move3 = wa_utils.filter_for_next_move2(cv2.imread(f'../resources/rec/9000.png'))
next_move4 = wa_utils.filter_for_next_move2(cv2.imread(f'../resources/rec/3926.png'))

cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_pos)

prev_next_move = False
prev_go = False

game_times = 0

with open(args.output_puyofu_file, 'w') as file:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        cv2.bitwise_and(wa_utils.filter_for_go(frame), go)
        diff_go = wa_utils.diff_binary(wa_utils.filter_for_go(frame), go)

        if diff_go < 500:
            if not prev_go:
                game_times += 1
                f = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                print(f'start={f}')
                print(f'start={f}', file=file)
            prev_go = True
        else:
            prev_go = False

        cropped_next_move = wa_utils.filter_for_next_move(frame)
        cropped_next_move2 = wa_utils.filter_for_next_move2(frame)
        diff_next_move = wa_utils.diff_gray(cropped_next_move, next_move)
        diff_next_move2 = wa_utils.diff_gray(cropped_next_move, next_move2)
        diff_next_move3 = wa_utils.diff_gray(cropped_next_move2, next_move3)
        diff_next_move4 = wa_utils.diff_gray(cropped_next_move2, next_move4)
        # print('next move', diff_next_move, diff_next_move2)

        if min(diff_next_move, diff_next_move2, diff_next_move3, diff_next_move4) < 100:
            if not prev_next_move:
                f = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                print(f'move={f}')
                print(f'move={f}', file=file)
                cap.set(cv2.CAP_PROP_POS_FRAMES, f - 3)
                _, frame_for_next = cap.read()

                next_pairs = recognize_next_pairs(frame_for_next)
                print(f'next={next_pairs}')
                print(f'next={next_pairs}', file=file)

                cap.set(cv2.CAP_PROP_POS_FRAMES, f)

                field = recognize_field(frame)
                print(f'field={"".join(field)}')
                print(f'field={"".join(field)}', file=file)
            prev_next_move = True
        else:
            prev_next_move = False
