import numpy as np
import cv2 as cv

def extractColorMask(imgs, n_cell = 2, method = "median"): #imgs : (, w, h, chanels)
    w, h = imgs.shape[-3], imgs.shape[-2]
    assert(w % n_cell == 0 and h % n_cell == 0)

    mask_shape = list(imgs.shape)
    mask_shape[-3] = mask_shape[-2] = n_cell
    masks = np.zeros(mask_shape)
    
    for i in range(n_cell):
        for j in range(n_cell):
            l = int(w / n_cell * i)
            r = int(w / n_cell * (i+1))
            u = int(h / n_cell * j)
            d = int(h / n_cell * (j+1))
            if method == "mean":
                masks[...,i,j,:] = np.median(imgs[...,l:r,u:d,:], axis=(-3,-2))
            elif method == "median":
                masks[...,i,j,:] = np.mean(imgs[...,l:r,u:d,:], axis=(-3,-2))
            else: raise ValueError("Method must be \"mean\" or \"median\"")
                
    return masks

def autoCrop(img, size = 96): 
    w, h = img.shape[0], img.shape[1]
    rat = size / min(w,h)
    
    img = cv.resize(img, None, fx=rat, fy=rat, interpolation=cv.INTER_CUBIC)  
    nw, nh = img.shape[0], img.shape[1]

    # Center croppingnh
    img = img[int((nw-size)/2):int((nw+size)/2), int((nh-size)/2):int((+size)/2),:]

    return img

def toGrayscale(imgs): #imgs: np.ndarray[,w,h,ch]
    return np.dot(imgs[...,:3], [0.299, 0.587, 0.114])

def adjustHSV(img, hp, sp):
    hsv = cv.cvtColor(img, code=cv.COLOR_RGB2HSV)
    if hp < 0:
        hsv[:,:,0] = np.clip(hsv[:,:,0] - np.abs(hp), 0, 167)
    else: hsv[:,:,0] = np.clip(hsv[:,:,0] + hp, 0, 167)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * sp, 0, 255)
    return cv.cvtColor(hsv, code=cv.COLOR_HSV2RGB)

def enhanceBrightness(img, alpha, beta):
    return cv.convertScaleAbs(img, alpha=alpha, beta=beta) 

def normalizeLAB(imgs):
    norm = imgs.astype(np.float32)
    norm = norm / 255
    return norm

def denormalizeLAB(imgs):
    denorm = imgs.copy()
    denorm = denorm * 255
    return denorm.astype("uint8")