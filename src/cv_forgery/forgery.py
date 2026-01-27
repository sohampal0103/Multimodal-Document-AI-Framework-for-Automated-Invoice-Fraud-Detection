import cv2, numpy as np

def forgery_score(image):
    edges = cv2.Canny(image,100,200)
    return float(np.mean(edges)/255)
