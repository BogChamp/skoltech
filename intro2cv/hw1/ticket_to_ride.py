from typing import Union
import json
from collections import defaultdict
from itertools import combinations

import numpy as np
import cv2
from skimage.transform import rescale, hough_circle, hough_circle_peaks
from skimage.measure import label, find_contours
from skimage.filters import gaussian
from scipy.spatial.distance import cdist
import scipy.stats as st


COLORS = ('blue', 'green', 'black', 'yellow', 'red')
TRAINS2SCORE = {1: 1, 2: 2, 3: 4, 4: 7, 6: 15, 8: 21}

def all_t(img_g):
    img_gb = cv2.GaussianBlur(img_g, (7, 7), 0)
    grad = cv2.adaptiveThreshold(img_gb, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 2)
    h_rads = np.arange(25, 28, 1)
    h_circ = hough_circle(grad, h_rads)

    _, x, y, _ = hough_circle_peaks(h_circ, h_rads,
                                               min_xdistance=150, min_ydistance=150, 
                                               total_num_peaks=47)
    ccc = np.vstack((y, x))
    return ccc.T


def clip_c(loc):
    tmp = loc.copy()
    for i,v in enumerate(tmp):
        tmp2 = tmp[i+1:]
        for j, k in enumerate(tmp2):
            if abs(v-k).sum() < 150:
                tmp[j+i+1] = tmp[i]
    return np.unique(tmp, axis=0)

def del_cnt(cnts, size=300):
    res = []
    for cnt in cnts:
        if cv2.contourArea(cnt) > size:
            res.append(cnt)
    return res

def find_trains(ttt, img, size=300):
    contours, hierarchy = cv2.findContours(ttt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = del_cnt(contours, size)
    return contours

def dilate(img,size=5, it=1):
    kernel = np.ones((size,size), np.int8)
    return cv2.dilate(img.astype(np.uint8), kernel, iterations=it)

def erdi(img, size=5):
    kernel = np.ones((size,size), np.int8)
    return cv2.morphologyEx(img.astype(np.uint8),cv2.MORPH_OPEN, kernel)

def dier(img, size=5):
    kernel = np.ones((size,size), np.int8)
    return cv2.morphologyEx(img.astype(np.uint8),cv2.MORPH_CLOSE, kernel)

def erode(img,size=5, it=1):
    kernel = np.ones((size,size), np.int8)
    return cv2.erode(img.astype(np.uint8), kernel, iterations=it)

def contour(img_g, template, th):
    res = cv2.matchTemplate(img_g, template, cv2.TM_CCOEFF_NORMED)
 
    loc = np.where(res >= th)
    return np.asarray(loc)

def del_slot(img, cnts, size=40):
    tmp = img.copy()
    for cnt in cnts:
        tmp[cnt[0]-size:cnt[0]+size, cnt[1]-size:cnt[1]+size] = 0
    
    return tmp

def plot_col(img, bgr):
    brightHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    
    hls = cv2.cvtColor( np.uint8([[bgr]] ), cv2.COLOR_BGR2HLS)[0][0]
    thresh = 50
    minHLS = np.array([hls[0] - thresh, hls[1] - thresh, hls[2] - thresh])
    maxHLS = np.array([hls[0] + thresh, hls[1] + thresh, hls[2] + thresh])
 
    maskHLS = cv2.inRange(brightHLS, minHLS, maxHLS)
    resultHLS = cv2.bitwise_and(brightHLS, brightHLS, mask = maskHLS)
   
    return resultHLS

def ret_YCB(img, bgr):
    bright = img
    brightYCB = cv2.cvtColor(bright, cv2.COLOR_BGR2YCrCb)
    
    thresh = 20
    ycb = cv2.cvtColor( np.uint8([[bgr]] ), cv2.COLOR_BGR2YCrCb)[0][0]
 
    minYCB = np.array([ycb[0] - thresh, ycb[1] - thresh, ycb[2] - thresh])
    maxYCB = np.array([ycb[0] + thresh, ycb[1] + thresh, ycb[2] + thresh])
 
    maskYCB = cv2.inRange(brightYCB, minYCB, maxYCB)
    resultYCB = cv2.bitwise_and(brightYCB, brightYCB, mask = maskYCB)
    return resultYCB

def LAB_plot(img, bgr):
    bright = img
    brightLAB = cv2.cvtColor(bright, cv2.COLOR_BGR2LAB)
    
    lab = cv2.cvtColor( np.uint8([[bgr]] ), cv2.COLOR_BGR2LAB)[0][0]
    thresh = 20
    minLAB = np.array([lab[0] - thresh, lab[1] - thresh, lab[2] - thresh])
    maxLAB = np.array([lab[0] + thresh, lab[1] + thresh, lab[2] + thresh])
 
    maskLAB = cv2.inRange(brightLAB, minLAB, maxLAB)
    resultLAB = cv2.bitwise_and(brightLAB, brightLAB, mask = maskLAB)

    return resultLAB

def red_trains(img, img_g, temp, rgb):
    HLS1 = plot_col(img, rgb[::-1])
    cnt = contour(img_g, temp, 0.5)
    qqq = HLS1[:,:,0]
    qqq = erode(qqq,6)
    qqq = del_slot(qqq, cnt.T)
    ccc = find_trains(qqq, img, 330)
    return len(ccc)

def blue_trains(img, img_g, temp, rgb):
    YCB = ret_YCB(img, rgb[::-1])
    cnt = contour(img_g, temp, 0.5)
    cnt = clip_c(cnt.T)
    qqq = YCB[:,:,0]
    qqq = erode(qqq,2)
    qqq = dilate(qqq,6)
    ccc = find_trains(qqq, img, 500)
    return len(ccc) - len(cnt)

def green_trains(img, img_g, template, rgb):
    YCB = ret_YCB(img, rgb[::-1])
    cnt = contour(img_g, template, 0.5)
    qqq = YCB[:,:,0]
    qqq = del_slot(qqq, cnt.T)
    qqq = dier(qqq,5)
    ccc = find_trains(qqq, img, 500)
    return len(ccc)

def yellow_trains(img, img_g, template, rgb):
    LAB = LAB_plot(img, rgb[::-1])
    cnt = contour(img_g, template, 0.5)
    qqq = LAB[:,:,0]
    qqq = del_slot(qqq, cnt.T, 50)
    qqq = erdi(qqq,5)
    ccc = find_trains(qqq, img, 1000)
    return len(ccc)

def black_trains(img):
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = (HSV[:, :, 2] < 29)
    mask[:100] = 0
    mask[:,:120] = 0
    mask[2500:] = 0
    mask[:,3700:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (13, 13))
    q = dier(mask)
    q = cv2.morphologyEx(np.uint8(q), cv2.MORPH_OPEN, kernel)
    q = mask * q
    q = erode(q,5)
    r = find_trains(q, img, 400)
    return len(r)

def predict_image(img: np.ndarray) -> (Union[np.ndarray, list], dict, dict):
    img_all = cv2.imread('./train/all.jpg')
    img_bbg = cv2.imread('./train/black_blue_green.jpg')
    img_bry = cv2.imread('./train/black_red_yellow.jpg')
    img_rgb = cv2.imread('./train/red_green_blue.jpg')
    img_inac = cv2.imread('./train/red_green_blue_inaccurate.jpg')

    img_all_g = cv2.cvtColor(img_all, cv2.COLOR_BGR2GRAY)
    img_bbg_g = cv2.cvtColor(img_bbg, cv2.COLOR_BGR2GRAY)
    img_bry_g = cv2.cvtColor(img_bry, cv2.COLOR_BGR2GRAY)
    img_rgb_g = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    img_inac_g = cv2.cvtColor(img_inac, cv2.COLOR_BGR2GRAY)

    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cities = all_t(img_g)

    red = img_all[750:790,2445:2460, ::-1]
    r = red[:,:,0].mean(), red[:,:,1].mean(), red[:,:,2].mean()
    blue = img_all[1990:2020,1920:1945, ::-1]
    b = blue[:,:,0].mean(), blue[:,:,1].mean(), blue[:,:,2].mean()
    green = img_all[950:975,1640:1665, ::-1]
    g = green[:,:,0].mean(), green[:,:,1].mean(), green[:,:,2].mean()
    yel = img_all[800:900,810:840, ::-1]
    y = yel[:,:,0].mean(), yel[:,:,1].mean(), yel[:,:,2].mean()

    pptr = img_all_g[1710:1740,1200:1230]
    pptb = img_all_g[790:820,1645:1675]
    pptg = img_rgb_g[1105:1135,1990:2020]
    ppty = img_all_g[673:703,1232:1262]
    red_tr = red_trains(img, img_g, pptr, r)
    blue_tr = blue_trains(img, img_g, pptb, b)
    green_tr = green_trains(img, img_g, pptg, g)
    yellow_tr = yellow_trains(img, img_g, ppty, y)
    black_tr = black_trains(img)
    r_score = red_tr * 1.5
    b_score = blue_tr * 1.5
    g_score = green_tr * 1.4
    y_score = yellow_tr * 1.5
    bl_score = black_tr * 1.4

    city_centers = np.int64(cities)
    n_trains = {'blue': blue_tr, 'green': green_tr, 'black': black_tr, 'yellow': yellow_tr, 'red': red_tr}
    scores = {'blue': b_score, 'green': g_score, 'black': bl_score, 'yellow': y_score, 'red': r_score}
    return city_centers, n_trains, scores
