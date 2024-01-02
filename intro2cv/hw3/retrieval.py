import cv2
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth, DBSCAN

def get_points(kp):
    res = np.array([kp[0].pt])
    for i in range(1, len(kp)):
        res = np.append(res, [kp[i].pt], axis=0)

    return res

def clasterize(kp2, ddd):
    dst_pts = [kp2[m.trainIdx].pt for m in ddd]
    clusters = DBSCAN(eps=80, min_samples = 2).fit_predict(dst_pts)
    uniq_clusters = np.unique(clusters)
    sizes = {}
    for clus in uniq_clusters:
        d, = np.where(clusters == clus)
        sizes[clus] = len(clusters[d])
    
    ind, = np.where(max(sizes, key=sizes.get) == clusters)
    
    maxclus = [ddd[i] for i in ind]
    return maxclus

def divide(num_of_clusters, labels, kp):
    kluster = [None] * num_of_clusters
    for i in range(num_of_clusters):
        d, = np.where(labels == i)
        kluster[i] = list(kp[j] for j in d)
    return kluster

def detection(kp1, kp2, ddd, query, train):
    
    maxclus = clasterize(kp2, ddd)
    
    if len(maxclus) <= 3 :
        return []

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in maxclus ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in maxclus ]).reshape(-1,1,2)
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2)

    if M is None:
        return []
    
    h, w = query.shape
    corners = np.float32([ [0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0] ]).reshape(-1, 1, 2)
    transformedCorners = cv2.perspectiveTransform(corners, M)
    pts = np.int32(transformedCorners).reshape(8).tolist()
    x1, x2 = min(pts[1], pts[3], pts[5], pts[7]), max(pts[1], pts[3], pts[5], pts[7])
    y1, y2 = min(pts[0], pts[2], pts[4], pts[6]), max(pts[0], pts[2], pts[4], pts[6])
    if not np.isclose((x2 - x1) / h, (y2 - y1) / w, 0.3):
        return []
    
    train = cv2.fillPoly(train, [np.int32(transformedCorners)], 255)
    return [x1, y1, y2 - y1, x2 - x1]


def find_bboxes(kp1, kp2, des1, des2, flann):

    if(len(kp2)<2 or len(kp1)<2):
        return []

    matches = flann.knnMatch(des1, des2, 2)

    ddd = []
    for m,n in matches:
        if m.distance < 0.88*n.distance:
            ddd.append(m)

    if len(ddd) <= 3:
        return []

    return ddd
    

def predict_image(img: np.ndarray, query: np.ndarray) -> list:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    query = cv2.cvtColor(query, cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(query, None)
    
    des1 = np.float32(des1)

    index_params = dict(algorithm = 1,  trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    list_of_bboxes = []
    while True:
        kp2, des2 = sift.detectAndCompute(img, None)
        des2 = np.float32(des2)
        bbox = find_bboxes(kp1, kp2, des1, des2, flann)
        if len(bbox):
            res = detection(kp1, kp2, bbox, query, img)
            if len(res):
                res[0] /= img.shape[0]
                res[1] /= img.shape[1]
                res[2] /= img.shape[1]
                res[3] /= img.shape[0]
                list_of_bboxes.append(res)
            else:
                break
        else:
            break

    return list_of_bboxes