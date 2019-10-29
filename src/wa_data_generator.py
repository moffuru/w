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
next_move2 = wa_utils.filter_for_next_move(cv2.imread(f'../resources/rec/3926.png'))
next_move3 = wa_utils.filter_for_next_move2(cv2.imread(f'../resources/rec/9000.png'))
next_move4 = wa_utils.filter_for_next_move2(cv2.imread(f'../resources/rec/3926.png'))

# print(go.shape)

cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_pos)
c = 0

prev_next_move = False
prev_go = False

game_times = 0

log_file = open(args.log, 'w') if args.log else None
movie_name, _ = os.path.split(os.path.basename(args.movie))

while True:
    ret, frame = cap.read()

    if not ret:
        break

    diff_go = wa_utils.diff_binary(wa_utils.filter_for_go(frame), go)
    # print('go', diff_go, diff_go2)

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

    cropped_next_move = wa_utils.filter_for_next_move(frame)
    cropped_next_move2 = wa_utils.filter_for_next_move2(frame)
    diff_next_move = wa_utils.diff_gray(cropped_next_move, next_move)
    diff_next_move2 = wa_utils.diff_gray(cropped_next_move, next_move2)
    diff_next_move3 = wa_utils.diff_gray(cropped_next_move2, next_move3)
    diff_next_move4 = wa_utils.diff_gray(cropped_next_move2, next_move4)
    # print('next move', diff_next_move, diff_next_move2)

    if min(diff_next_move, diff_next_move2, diff_next_move3, diff_next_move4) < 500:
        if not prev_next_move:
            f = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            print(f'move={f}')
            if log_file:
                print(f'move={f}', file=log_file)
            if args.dir is not None:
                path = f'{args.dir}/{movie_name}_{f}.png'
                if not os.path.exists(path):
                    cv2.imwrite(path, frame)
                    pass
        prev_next_move = True
    else:
        prev_next_move = False

