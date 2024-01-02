import time
import json
from pathlib import Path
from typing import Union, Callable
from gzip import GzipFile
from itertools import permutations
from functools import wraps

import cv2
import numpy as np
from skimage.measure import label

# from dpipe.im.metrics import iou

from sudoku import predict_image


# ################################################## Utils ############################################################


PathLike = Union[Path, str]


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def load_numpy(path: PathLike, *, allow_pickle: bool = True, fix_imports: bool = True, decompress: bool = False):
    """A wrapper around ``np.load`` with ``allow_pickle`` set to True by default."""
    if decompress:
        with GzipFile(path, 'rb') as file:
            return load_numpy(file, allow_pickle=allow_pickle, fix_imports=fix_imports)

    return np.load(path, allow_pickle=allow_pickle, fix_imports=fix_imports)


def join(values):
    return ", ".join(map(str, values))


def check_bool(*arrays):
    for i, a in enumerate(arrays):
        assert a.dtype == bool, f'{i}: {a.dtype}'


def check_shapes(*arrays):
    shapes = [array.shape for array in arrays]
    if any(shape != shapes[0] for shape in shapes):
        raise ValueError(f'Arrays of equal shape are required: {join(shapes)}')


def add_check_function(check_function: Callable):
    """Decorator that checks the function's arguments via ``check_function`` before calling it."""

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            check_function(*args, *kwargs.values())
            return func(*args, **kwargs)

        return wrapper

    name = getattr(check_function, '__name__', '`func`')
    decorator.__doc__ = f"Check the function's arguments via `{name}` before calling it."
    return decorator


add_check_bool, add_check_shapes = map(add_check_function, [check_bool, check_shapes])


def fraction(numerator, denominator, empty_val: float = 1):
    assert numerator <= denominator, f'{numerator}, {denominator}'
    return numerator / denominator if denominator != 0 else empty_val


@add_check_bool
@add_check_shapes
def iou(x: np.ndarray, y: np.ndarray) -> float:
    return fraction(np.sum(x & y), np.sum(x | y))


# ############################################# Evaluation code #######################################################


def evaluate_case(i):
    time_eval_start = time.perf_counter()

    predict_fn = predict_image

    img = cv2.imread(f'train/train_{i}.jpg')
    mask = load_numpy(f'train/train_{i}_mask.npy.gz', decompress=True)

    digits = []
    lbl, n = label(mask, connectivity=2, return_num=True)
    for j in range(1, n + 1):
        digits.append(load_json(f'train/train_{i}_digits_{j - 1}.json'))

    time_predict_start = time.perf_counter()
    mask_pred, digits_pred = predict_fn(img)
    time_predict_end = time.perf_counter()

    score_iou = score_mask(lbl, n, mask_pred)
    score_accuracy = score_digits(digits, digits_pred)

    time_eval_end = time.perf_counter()

    return score_iou, score_accuracy, time_predict_end - time_predict_start, time_eval_end - time_eval_start


def score_mask(lbl_true, n_true, mask_pred, iou_max_th=0.9):
    lbl_pred, n_pred = label(mask_pred, connectivity=2, return_num=True)
    masks_true = [lbl_true == i for i in range(1, n_true + 1)]
    masks_pred = [lbl_pred == i for i in range(1, n_pred + 1)]

    def _iou(x, y, th=iou_max_th):
        s = iou(x, y)
        return s if s < th else 1.

    return score_permutations(obj1=masks_true, obj2=masks_pred, score_fn=_iou)


def score_digits(digits_true, digits_pred):

    def _score_digits(d_true, d_pred):
        _empty_symb = -1
        d_true = np.array(d_true)
        d_pred = np.array(d_pred)
        n_nonempty = np.sum(d_true != _empty_symb)
        n_error = np.sum(d_true != d_pred)
        return 1 - np.clip(n_error / n_nonempty, 0, 1)

    return score_permutations(obj1=digits_true, obj2=digits_pred, score_fn=_score_digits)


def score_permutations(obj1, obj2, score_fn):
    if (len(obj1) == 0) or (len(obj2) == 0):
        return 0.

    if len(obj1) >= len(obj2):
        permut_obj = obj1
        static_obj = obj2
        preserved = True
    else:
        permut_obj = obj2
        static_obj = obj1
        preserved = False

    scores = []
    permut_idx = np.arange(len(permut_obj))
    for p in permutations(permut_idx):
        _p = p[:len(static_obj)]
        if preserved:
            scores.append(np.mean([score_fn(permut_obj[_p[i]], static_obj[i]) for i in range(len(_p))]))
        else:  # if `score_fn` is sensitive (not symmetric) to permutation of true and pred
            scores.append(np.mean([score_fn(static_obj[i], permut_obj[_p[i]]) for i in range(len(_p))]))

    return np.max(scores)


def main():
    train_ids = list(range(9))

    leaderboard_score = 90.
    one_img_score = leaderboard_score / len(train_ids)

    final_score_masks, final_score_digits = 0, 0
    total_eval_time = 0
    for i in train_ids:

        print(f'### train case `{i}` ###', flush=True)
        try:
            score_iou, score_accuracy, t_pred, t_eval = evaluate_case(i)

            print(f'Raw IoU score = {score_iou:.2f}', flush=True)
            print(f'Raw digit recognition score = {score_accuracy:.2f}', flush=True)
            print(f'Prediction time = {t_pred:.2f}s. Evaluation overhead = {t_eval - t_pred:.2f}s.', flush=True)
            print()
        except Exception as e:
            score_iou, score_accuracy, t_pred, t_eval = 0, 0, 0, 0
            print(f'Exception `{e}` in train case {i}. The case scores are set to 0.', flush=True)
            print()

        final_score_masks += score_iou * one_img_score * 0.5
        final_score_digits += score_accuracy * one_img_score * 0.5
        total_eval_time += t_eval

    print(f'### Summary ###', flush=True)
    print(f'Final train score (finding tables) = {final_score_masks:.2f} out of {leaderboard_score * 0.5:.2f}',
          flush=True)
    print(f'Final train score (recognizing digits) = {final_score_digits:.2f} out of {leaderboard_score * 0.5:.2f}',
          flush=True)
    print(f'Total evaluation time = {total_eval_time:.2f}s for 9 images.', flush=True)
    print(f'The platform limit is {40*60}s for 30 images.')


if __name__ == '__main__':
    main()
