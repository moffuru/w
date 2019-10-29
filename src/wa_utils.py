import wa_constants
import cv2


def filter_for_go(frame):
    _, go = cv2.threshold(
        cv2.cvtColor(frame[wa_constants.GO_Y1: wa_constants.GO_Y2, wa_constants.GO_X1: wa_constants.GO_X2], cv2.COLOR_BGR2GRAY),
        200,
        255,
        cv2.THRESH_BINARY
    )

    return go


def filter_for_next_move(frame):
    next_move = cv2.cvtColor(frame[wa_constants.NEXT_MOVE_Y1: wa_constants.NEXT_MOVE_Y2, wa_constants.NEXT_MOVE_X1: wa_constants.NEXT_MOVE_X2], cv2.COLOR_BGR2GRAY)

    # return cv2.equalizeHist(next_move)
    return next_move


def filter_for_next_move2(frame):
    next_move = cv2.cvtColor(frame[153: 153+27, 477: 477+13], cv2.COLOR_BGR2GRAY)

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


def filter_for_puyo(image):
    # image = cv2.resize(image, (200, 200))
    # return image
    return image  # cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#[8: 8 + 23, 11: 11 + 23]


def filter_for_next1_puyo1(image):
    image = image[109: 109 + wa_constants.REAL_PUYO_HEIGHT, 480: 480 + wa_constants.REAL_PUYO_WIDTH]
    return filter_for_puyo(image)


def filter_for_next1_puyo2(image):
    image = image[109 + wa_constants.REAL_PUYO_HEIGHT: 109 + wa_constants.REAL_PUYO_HEIGHT * 2, 480: 480 + wa_constants.REAL_PUYO_WIDTH]
    return filter_for_puyo(image)


def filter_for_next2_puyo1(image):
    image = image[196: 196 + wa_constants.REAL_NEXT2_PUYO_HEIGHT, 513: 513 + wa_constants.REAL_NEXT2_PUYO_WIDTH]
    return filter_for_puyo(image)


def filter_for_next2_puyo2(image):
    image = image[196 + wa_constants.REAL_NEXT2_PUYO_HEIGHT: 196 + wa_constants.REAL_NEXT2_PUYO_HEIGHT * 2, 513: 513 + wa_constants.REAL_NEXT2_PUYO_WIDTH]
    return filter_for_puyo(image)

