import copy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('puyofu')
args = parser.parse_args()

# https://github.com/puyoai/puyoai/blob/master/src/solver/puyop.cc

URL_PREFIX = 'http://www.puyop.com/s/'

ENCODER = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ[]'


def tsumo_color_id(c):
    if c == 'R':
        return 0
    if c == 'G':
        return 1
    if c == 'B':
        return 2
    if c == 'Y':
        return 3
    if c == 'P':
        return 4
    return 0


def encode_control(seq, decisions):
    res = ''
    for i in range(len(decisions)):
        # print(res)
        a = tsumo_color_id(seq[i * 2])
        b = tsumo_color_id(seq[i * 2 + 1])
        d = a * 5 + b
        h = ((decisions[i][0] + 1) << 2) + decisions[i][1]
        d |= h << 7
        res += ENCODER[d & 0x3f] + ENCODER[(d >> 6) & 0x3f]
    return res


def encode_field(field):
    return ''


def make_puyop_url(field, seq, decisions):
    return URL_PREFIX + encode_field(field) + '_' + encode_control(seq, decisions)


class Field:
    def __init__(self, field_str=None):
        self.for_vanish = None
        if field_str is None:
            self.field = None
            return

        field_str += ' ' * (12 * 6)
        field_str = field_str[0: 12 * 6]
        # print(f'{field_str}$')
        self.field = []
        self.field = [[' '] * 6]
        for y in range(12):
            col = []
            for x in range(6):
                i = y * 6 + x
                col.append(field_str[i])
            self.field.append(col)

        # print(self.field)

    def rec(self, x, y, c, vid):
        if x < 0 or x >= 6 or y < 0 or y >= 13 or self.for_vanish[y][x] >= 0 or self.field[y][x] != c:
            return 0

        self.for_vanish[y][x] = vid
        res = 1

        for i in range(4):
            nx = x + [0, 1, 0, -1][i]
            ny = y + [1, 0, -1, 0][i]
            res += self.rec(nx, ny, c, vid)

        return res

    def make_falled_row(self, x):
        map_row = []
        for y in range(12, -1, -1):
            if self.field[y][x] != ' ':
                map_row.append(self.field[y][x])

        for y in range(13):
            if y < len(map_row):
                self.field[12 - y][x] = map_row[y]
            else:
                self.field[12 - y][x] = ' '

    def make_falled(self):
        for x in range(6):
            self.make_falled_row(x)

    def vanish_ojama(self, x, y):
        if x < 0 or x >= 6 or y < 0 or y >= 12 or self.field[y][x] != 'O':
            return

        self.field[y][x] = ' '

    def vanish(self):
        chains = 0
        while True:
            self.for_vanish = [[-1] * 6 for _ in range(13)]
            vanished = False
            vid = 0
            a = []
            for y in range(1, 13):
                for x in range(6):
                    if self.field[y][x] != ' ' and self.field[y][x] != 'O' and self.for_vanish[y][x] < 0:
                        cnt = self.rec(x, y, self.field[y][x], vid)
                        # print('cnt=', cnt)
                        if cnt >= 4:
                            vanished = True
                            a.append(vid)
                        vid += 1

            # print('vanished', vanished)
            if not vanished:
                break

            chains += 1

            for y in range(1, 13):
                for x in range(6):
                    if self.for_vanish[y][x] in a:
                        self.field[y][x] = ' '
                        for i in range(4):
                            nx = x + [0, 1, 0, -1][i]
                            ny = y + [1, 0, -1, 0][i]
                            self.vanish_ojama(nx, ny)

            self.make_falled()

        return chains

    def fall_puyo(self, x, puyo):
        if x < 0 or x >= 6 or self.field[0][x] != ' ':
            return 0

        for y in range(12, -1, -1):
            if self.field[y][x] == ' ':
                self.field[y][x] = puyo
                return 1
        return 0

    def fall_kumipuyo(self, x, r, kumipuyo):
        f = Field()
        f.field = copy.deepcopy(self.field)

        res = 0

        if r == 0:
            res += f.fall_puyo(x, kumipuyo[0])
            res += f.fall_puyo(x, kumipuyo[1])
        if r == 1:
            res += f.fall_puyo(x, kumipuyo[0])
            res += f.fall_puyo(x + 1, kumipuyo[1])
        if r == 2:
            res += f.fall_puyo(x, kumipuyo[1])
            res += f.fall_puyo(x, kumipuyo[0])
        if r == 3:
            res += f.fall_puyo(x, kumipuyo[0])
            res += f.fall_puyo(x - 1, kumipuyo[1])

        # print(f.to_string_h())

        chains = f.vanish()

        # print(x, r)
        # print(kumipuyo)
        # print('-------------------------')
        # print(f.to_string_h())
        # print('-------------------------')
        if res == 2:
            return f, chains
        else:
            return None, None

    def to_string(self):
        res = ''
        for y in range(13):
            for x in range(6):
                res += self.field[y][x]
        return res

    def to_string_h(self):
        res = ''
        for y in range(13):
            if y < 10:
                res += f' {y}'
            else:
                res += f'{y}'
            res += ': '
            for x in range(6):
                res += self.field[y][x]
            res += '\n'
        return res

    def estimate_next_decision(self, next_field, kumipuyo):
        next_field_str = next_field.to_string()
        ax = -1
        ar = -1
        ac = -1
        for x in range(6):
            for r in range(4):
                res, chains = self.fall_kumipuyo(x, r, kumipuyo)
                if res is not None and res.to_string() == next_field_str and chains > ac:
                    ax = x
                    ar = r
                    ac = chains
        return ax, ar, ac


with open(args.puyofu, 'r') as f:
    prev_field = None
    field = None
    kumipuyo_seq = ''
    continue_until_start = True
    decisions = []
    g = 0
    mc = 0

    for l in f:
        # print(l)
        # if g == 6:
        #     print(l)
        if l.startswith('start='):
            g += 1
            mc = 0
            continue_until_start = False
            decisions = []
            f.readline()
            kumipuyo_seq = f.readline().split('=')[1].strip()[0:2]
            field = Field(f.readline().split('=')[1].strip())

        # elif g != 6:
        #     continue

        elif continue_until_start:
            continue

        elif l.startswith('next='):
            kumipuyo_seq += l.split('=')[1].strip()[0:2]
            # print(kumipuyo_seq)

        elif l.startswith('field='):
            prev_field = field
            field = Field(l.split('=')[1][:-1])
            # print(kumipuyo_seq[-4:-2])
            # print(field.to_string_h())
            # break
            if len(kumipuyo_seq) < 4:
                continue_until_start = True
                # print('field',)
                continue
            x, r, c = prev_field.estimate_next_decision(field, kumipuyo_seq[-4:-2])
            if c > mc:
                mc = c
            if x < 0:
                # print('dame', field.to_string(), prev_field.to_string())
                # print(field.to_string_h())
                continue_until_start = True
                print(f'|{g}|{make_puyop_url(field, kumipuyo_seq, decisions)} |{mc}||')
            else:
                # print(x, r)
                decisions.append((x, r))
            # break


        # print(field, kumipuyo_seq)
