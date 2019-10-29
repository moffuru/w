import argparse
import os

import cv2

import wa_utils

parser = argparse.ArgumentParser()
parser.add_argument('movie')
parser.add_argument('-s', '--start_pos', type=int, default=0)
parser.add_argument('-d', '--dir', default=None)
parser.add_argument('--log', default=None)
parser.add_argument('-g', '--games', type=int, default=10)
args = parser.parse_args()

cap = cv2.VideoCapture(args.movie)
go = wa_utils.filter_for_go(cv2.imread(f'../resources/rec/8488.png'))
next_move = wa_utils.filter_for_next_move(cv2.imread(f'../resources/rec/9000.png'))

# print(go.shape)

cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_pos)
c = 0

prev_next_move = False
prev_go = False

game_times = 0

log_file = open(args.log, 'w') if args.log else None

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.bitwise_and(wa_utils.filter_for_go(frame), go)
    diff_go = wa_utils.diff_binary(wa_utils.filter_for_go(frame), go)
    # print('go', diff_go)

    if diff_go < 500:
        if not prev_go:
            game_times += 1
            if game_times > args.games:
                break
            f = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            print(f'start={f}')
            if log_file:
                print(f'start={f}', file=log_file)
        prev_go = True
    else:
        prev_go = False

    diff_next_move = wa_utils.diff_gray(wa_utils.filter_for_next_move(frame), next_move)
    # print('next move', diff_next_move)

    if diff_next_move < 100:
        if not prev_next_move:
            f = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            print(f'move={f}')
            if log_file:
                print(f'move={f}', file=log_file)
            if args.dir is not None:
                path = f'{args.dir}/{f}.png'
                if not os.path.exists(path):
                    cv2.imwrite(path, frame)
                    pass
        prev_next_move = True
    else:
        prev_next_move = False

