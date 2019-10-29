import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('movie')
args = parser.parse_args()

cap = cv2.VideoCapture(args.movie)

cur = 0
pos = 8440

while True:
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

    ret, frame = cap.read()
    if ret:
        cv2.imshow('frame', frame)
    else:
        break

    k = cv2.waitKey(1) & 0xff

    if k == ord('s'):
        cv2.imwrite(f'{pos}.png', frame)

    if k == ord('a'):
        pos -= 1

    if k == ord('d'):
        pos += 1

    if k == ord('q'):
        break
