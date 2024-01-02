from sklearn.ensemble import RandomForestClassifier
import json
from itertools import islice
from typing import Iterable, Union, Tuple
import numpy as np
from numpy import logical_and as land
from numpy import logical_not as lnot
from skimage.feature import canny
from skimage.transform import rescale, ProjectiveTransform, warp
from skimage.morphology import dilation, disk
import cv2
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(20, 40, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        
        self.linear_block = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1960, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 10)
        )

    def forward(self, x):

        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.linear_block(x)
        
        return F.log_softmax(x)

SCALE = 0.33

def plot_img(img, cmap='gray'):
    plt.figure(figsize=(12,6))
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.show()

def get_table_only(img, kernel_size=20, size=1):
    kernel1 = np.ones((size, kernel_size), dtype='uint8')
    kernel2 = np.ones((kernel_size, size), dtype='uint8')

    a1 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel1)
    a2 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel2)
    
    a3 = np.min([a1, a2], axis=0)
    
    return a3

def inverse_img(img):
    return 255 - img

def remove_table(img):
    a4 = inverse_img(img)
    img_table = get_table_only(img)
    a5 = inverse_img(img_table)
    a6 = np.clip(a4 - a5, 0, None)
    return a6, img_table

def read_data(num, num2='0'):
    f = open('train/train_' + num + '_digits_'+ num2 + '.json')

    data = json.load(f)

    f.close()
    data = np.asarray(data)
  
    return data

def create_dataset():
  res = np.array([])
  target = np.array([[]])
  for num in range(9):
    img = cv2.imread('train/train_' + str(num) + '.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    src, _ = detect(img)
    for i,v in enumerate(src):
        data = read_data(str(num), str(i))
        target = np.concatenate((target, data.reshape(-1, 81)), axis=None)

        w = masked(img, v)
        tmp = np.asarray(list(gen(w)))
        if len(res) == 0:
            res = tmp.reshape(len(tmp), -1)
        else:
            res = np.vstack((res, tmp.reshape(len(tmp), -1)))
    
    
  return res, target

def gen(img):
    l = img.shape[0] // 9
    w = img.shape[1] // 9
    for i in range(0, img.shape[0]-l+1, l):
        for j in range(0, img.shape[1]-w+1, w):
            yield img[i:i+l-2, j:j+w-2]

def del_cnt(cnts, size=500000):
    res = []
    for cnt in cnts:
        if cv2.contourArea(cnt) > size:
            res.append(cnt)
    return res

def contours(img):
    th = cv2.adaptiveThreshold(img,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
    contour, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = del_cnt(contour)
    return cnt

def detect(img):
    cnts = contours(img)
    res = np.zeros_like(img)
    srcs = []

    for cnt in cnts:
        hull = cv2.convexHull(cnt)
        maskh = cv2.drawContours(np.zeros_like(img), [hull], -1, 1, -1)
        approx = cv2.approxPolyDP(hull, 0.05*cv2.arcLength(hull, True), True)

        if len(approx) == 4:
            area = cv2.contourArea(approx)
            perimeter = cv2.arcLength(approx, True) 
            ar = area * 16 / perimeter**2
            if ar >= 0.985 and ar <= 1.15:
                res = np.ma.mask_or(res.astype(np.uint8), maskh)

                src = approx[::-1].reshape(4, 2)
                if abs(src[0][0] - src[1][0]) > abs(src[0][0] - src[3][0]):
                    src = np.vstack((src[3], src[:3]))

                srcs.append(src)
                
    return srcs, res

def masked(img, src):

    dst = np.array([[0, 0],[0, 1350],[1350, 1350],[1350, 0]])
    tform = ProjectiveTransform()
    tform.estimate(dst, src)

    warped = warp(img, tform, output_shape=(1350,1350))

    res = rescale(warped, 0.2, anti_aliasing=True)
    res, table = remove_table(res)
    return (res - res.min()) / (res.max() - res.min()), table

def solve(field: Union[str, np.ndarray, list, tuple], cell_shape: Tuple[int, int] = None) -> Iterable[np.ndarray]:
    """
    Solves a sudoku puzzle and yields all possible solutions.

    Parameters
    ----------
    field
        if array - nan values denote unfilled positions, if str - '?' denotes unfilled positions
    cell_shape
        the shape of a single cell (e.g. simple sudoku = (3, 3) ), supports non-square cells.
        if None - the shape is considered square and is inferred from field's shape
    """

    if isinstance(field, str):
        field, initial, cell_shape = parse_text(field, cell_shape)
    else:
        field, initial, cell_shape = parse_array(field, cell_shape)

    for v in initial:
        propagate_constraints(field, *v, cell_shape)
    return make_suggestions(field, cell_shape)


class NotSolvable(ValueError):
    pass

def parse_array(array: np.ndarray, cell_shape):
    array = np.asarray(array, float)
    field_side = len(array)
    if array.ndim != 2 or field_side != array.shape[1]:
        raise ValueError('The field is not square.')

    cell_shape, cell_size = get_cell_shape(field_side, cell_shape)
    field = np.ones([field_side] * 3, dtype=bool)
    not_empty = ~np.isnan(array)

    initial = []
    for i, j in zip(*not_empty.nonzero()):
        value = int(array[i, j])
        assert 0 < value <= cell_size

        value -= 1
        field[i, j] = 0
        field[i, j, value] = 1
        initial.append((i, j, value))

    return field, initial, cell_shape

def parse_text(text: str, cell_shape):
    values = text.split()
    field_side = int(np.sqrt(len(values)))
    if len(values) != field_side ** 2:
        raise ValueError('The field is not square.')

    cell_shape, cell_size = get_cell_shape(field_side, cell_shape)
    field = np.ones([field_side] * 3, dtype=bool)

    initial = []
    for i, value in enumerate(values):
        if value != '?':
            value = int(value)
            assert 0 < value <= cell_size, value

            i, j = np.unravel_index(i, field.shape[:2])
            value -= 1
            field[i, j] = 0
            field[i, j, value] = 1
            initial.append((i, j, value))

    return field, initial, cell_shape

def propagate_constraints(field, i, j, value, cell_shape):
    mask = field[..., value]
    update_at = np.zeros_like(mask)

    # rows, cols
    update_at[i] = 1
    update_at[:, j] = 1

    # cell
    start = (np.array((i, j)) // cell_shape) * cell_shape
    stop = start + cell_shape
    update_at[tuple(map(slice, start, stop))] = 1

    # except
    update_at[i, j] = 0

    # only where should remove
    update_at = update_at & mask
    mask[update_at] = 0

    if not field.any(-1).all():
        # no possible values
        raise NotSolvable

    # propagate
    update_at = update_at & (field.sum(-1) == 1)
    xs, ys = np.where(update_at)
    for i, j in zip(xs, ys):
        unique_value, = np.where(field[i, j])[0]
        propagate_constraints(field, i, j, unique_value, cell_shape)


def is_solved(field):
    return (field.sum(-1) == 1).all()


def make_suggestions(field, cell_shape):
    if is_solved(field):
        values = np.where(field)[2] + 1
        yield values.reshape(*field.shape[:2])
        return

    variants = field.sum(-1).astype(float)
    variants[variants == 1] = np.inf
    idx = np.argmin(variants)
    i, j = np.unravel_index(idx, field.shape[:2])

    for value in range(len(field)):
        if field[i, j, value]:
            suggestion = field.copy()
            # suppose that the real value at (i,j) is `value`
            suggestion[i, j] = 0
            suggestion[i, j, value] = 1

            try:
                propagate_constraints(suggestion, i, j, value, cell_shape)
                yield from make_suggestions(suggestion, cell_shape)
            except NotSolvable:
                pass


def get_cell_shape(field_side, cell_shape):
    if cell_shape is None:
        cell_side = int(np.sqrt(field_side))
        assert cell_side ** 2 == field_side
        cell_shape = np.array((cell_side, cell_side), int)

    cell_size = np.prod(cell_shape)
    assert field_side == cell_size
    return cell_shape, cell_size


def predict_image(image: np.ndarray) -> (np.ndarray, list):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sudoku_digits = [
        np.int16([[-1, -1, -1, -1, -1, -1, -1, -1, -1],

                  [-1, -1, -1,  8,  9,  4, -1, -1, -1],

                  [-1, -1, -1,  6, -1,  1, -1, -1, -1],

                  [-1,  6,  5,  1, -1,  9,  7,  8, -1],

                  [-1,  1, -1, -1, -1, -1, -1,  3, -1],

                  [-1,  3,  9,  4, -1,  5,  6,  1, -1],

                  [-1, -1, -1,  8, -1,  2, -1, -1, -1],

                  [-1, -1, -1,  9,  1,  3, -1, -1, -1],

                  [-1, -1, -1, -1, -1, -1, -1, -1, -1]]),
    ]

    img3 = cv2.imread('/autograder/source/train/train_3.jpg')
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    src, _ = detect(img3)
    for s in src:
        w, _ = masked(img3, s)

    n1 = w[30*4:30*(4+1),30*3:30*(3+1)]
    n2 = w[30*0:30*(0+1),30*0:30*(0+1)]
    n3 = w[30*3:30*(3+1),30*2:30*(2+1)] 
    n4 = w[30*5:30*(5+1),30*0:30*(0+1)]
    n5 = w[30*2:30*(2+1),30*3:30*(3+1)]
    n6 = w[30*2:30*(2+1),30*5:30*(5+1)]
    n7 = w[30*1:30*(1+1),30*1:30*(1+1)]
    n8 = w[30*7:30*(7+1),30*1:30*(1+1)]
    n9 = w[30*3:30*(3+1),30*0:30*(0+1)]
    d = {1:n1,2:n2,3:n3,4:n4,5:n5,6:n6,7:n7,8:n8,9:n9}

    src, mask = detect(image)
    #clf = joblib.load('/autograder/source/rf1.joblib')
    network = Net()
    network.load_state_dict(torch.load('/autograder/submission/model.pth'))
    network.eval()

    for s in src:
        w, table = masked(image, s)
        tmp = np.asarray(list(gen(w)))

        tmp = torch.from_numpy(tmp).type(torch.float).reshape(len(tmp),1,28,28)
        output = network(tmp)
        pred = output.data.max(1, keepdim=True)[1]
        pred = pred.flatten()
        pred[pred == 0] = -1

        for i, v in enumerate(tmp):
            sudoku_digits[0][i // 9][i % 9] = pred[i]
            
    #res = sudoku_digits[0].copy()
    #res = res.astype(float)
    #res[res==-1] = np.nan

    #try:
    #    res = next(solve(res))
    #except:
    #    res = np.array([])
    #if len(res):
    #    OOO = np.zeros((270,270))

    #    for i in range(9):
    #        for j in range(9):
    #            OOO[30*i:30*(i+1), 30*j:30*(j+1)] = d[res[i][j]]
    
    #    table = (255 * (table - table.min()) / (table.max() - table.min())).astype(np.int16)
    #    res = rescale((table-OOO), 5, anti_aliasing=True)
    #    dst = np.array([[0, 0],[0, 1350],[1350, 1350],[1350, 0]])
    #    tform = ProjectiveTransform()
    #    tform.estimate(src[0], dst)
    #    warped = warp(res, tform, output_shape=image.shape)
    #    orig = image*(1-mask)+warped
    #    orig = cv2.cvtColor(orig.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    #plot_img(orig)

    return mask, sudoku_digits

