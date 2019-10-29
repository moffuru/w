import wa_constants
import cv2


def filter_for_go(frame):
    _, go = cv2.threshold(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[wa_constants.GO_Y1: wa_constants.GO_Y2, wa_constants.GO_X1: wa_constants.GO_X2],
        200,
        255,
        cv2.THRESH_BINARY
    )

    return go


def filter_for_next_move(frame):
    next_move = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[wa_constants.NEXT_MOVE_Y1: wa_constants.NEXT_MOVE_Y2, wa_constants.NEXT_MOVE_X1: wa_constants.NEXT_MOVE_X2]

    # return frame
    return next_move


def diff_binary(a, b):
    h, w = a.shape
    d = 0
    for x in range(w):
        for y in range(h):
            d += 1 if a.item(y, x) != b.item(y, x) else 0

    return d


def diff_gray(a, b):
    # h, w = a.shape
    # d = 0
    # for y in range(h):
    #     for x in range(w):
    #         d += abs(a.item(y, x) - b.item(y, x))

    d = 0
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    fgbg.apply(a)
    fgmask = fgbg.apply(b)

    h, w = fgmask.shape
    for x in range(w):
        for y in range(h):
            d += fgmask.item(y, x)

    return d
    #return fgmask


def flatten_image(image):
    s = image.shape[0] * image.shape[1] * image.shape[2]
    a = image.reshape(1, s) / 255
    return a[0]


def transform_for_recognition(image):
    return flatten_image(cv2.resize(image, (16, 16)))
