import cv2
import numpy as np

#Basic image processing methods

#Method for filling holes in connected components of the mask
#Brings all the connected components to simply connected

def fill_holes (img):
    (h, w) = img.shape
    
    before_area = img.sum ()
    
    img_enlarged = np.zeros ((h + 2, w + 2), np.uint8)
    img_enlarged [1:h+1, 1:w+1] = img

    img_enl_not = cv2.bitwise_not (img_enlarged)
    th, im_th = cv2.threshold (img_enl_not, 220, 255, cv2.THRESH_BINARY_INV);

    im_floodfill = im_th.copy()

    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = im_th | im_floodfill_inv
    
    result = im_out [1:h-1, 1:w-1]
    
    return result

#Method for bringing 3-channel image to RG-chromaticity color space
#Essentially it is a kind of normalization such that the absolute
#values of the intensity are neglected; the components ratios are not

def to_RG_chromaticity (img):
    (h, w, d) = img.shape
    
    norm = np.zeros ((h, w), np.float)
    norm = img [:, :, 0].astype ('float') +\
           img [:, :, 1].astype ('float') +\
           img [:, :, 2].astype ('float')
    
    norm [norm == 0] = 5
    
    turned = np.zeros (img.shape, np.uint8)
    turned [:, :, 0] = ((img [:, :, 0].astype ('float')) / norm * 255).astype ('uint8')
    turned [:, :, 1] = ((img [:, :, 1].astype ('float')) / norm * 255).astype ('uint8')
    turned [:, :, 2] = ((img [:, :, 2].astype ('float')) / norm * 255).astype ('uint8')
    
    return turned

# Single-channel image to three-channel
# Forgive me Father for I have sinned
def to_three (img):
    sh = img.shape

    result = np.zeros((sh[0], sh[1], 3), img.dtype)

    for i in range(0, 3):
        result[:, :, i] = img.copy()

    return result

#Method for finding the bounding box of the connected component having the biggest value
#of the given criterion, i.e. height, width, area.
#Not implemented yet.

#def find_bounding_box (mask, criterion):
#    result = np.array (mask)
#    output = cv2.connectedComponentsWithStats (mask, 8, cv2.CV_32S)
#    labels_num = output      [0]
#    labels     = output      [1]
#    stats      = output      [2]
#    sz         = stats.shape [0]
#    
#    max_crit_val = 0
#    max_label    = 0
#    
#    for label_num in range (1, sz - 1):
#        if (stats [label_num, cv2.CC_STAT_AREA] > max_w):
#            max_w = stats [label_num, cv2.CC_STAT_AREA]
#            max_label = label_num
#    
#    top    = stats [max_label, cv2.CC_STAT_TOP]
#    left   = stats [max_label, cv2.CC_STAT_LEFT]
#    width  = stats [max_label, cv2.CC_STAT_WIDTH]
#    height = stats [max_label, cv2.CC_STAT_HEIGHT]
#    
#    return (left, top), (left + width, top + height)

def find_max_bounding_box (mask, bbox_num):
    result = np.array (mask)
    output = cv2.connectedComponentsWithStats (mask, 8, cv2.CV_32S)
    labels_num = output      [0]
    labels     = output      [1]
    stats      = output      [2]
    sz         = stats.shape [0]
    
    max_area  = 0
    max_label = 0
    
    success = True

    #if (sz == 1):
    #    success = False

    sorted_components = stats [np.argsort (stats[:, cv2.CC_STAT_AREA])]
    sorted_components = sorted_components [: -1]
    sorted_components = sorted_components [ - min (bbox_num, len (sorted_components)) :]

    #for label_num in range (1, sz):
    #    if (stats [label_num, cv2.CC_STAT_AREA] > max_area):
    #        max_area = stats [label_num, cv2.CC_STAT_AREA]
    #        max_label = label_num
    
    result = []

    for i in range (len (sorted_components)):
        top    = sorted_components [i, cv2.CC_STAT_TOP]
        left   = sorted_components [i, cv2.CC_STAT_LEFT]
        width  = sorted_components [i, cv2.CC_STAT_WIDTH]
        height = sorted_components [i, cv2.CC_STAT_HEIGHT]
    
        result.append (((left, top), (left + width, top + height)))

    #print ("result", result)

    return result, success

def leave_max_connected_component (mask):
    result = np.zeros_like (mask)
    output = cv2.connectedComponentsWithStats (mask, 8, cv2.CV_32S)
    labels_num = output      [0]
    labels     = output      [1]
    stats      = output      [2]
    sz         = stats.shape [0]
    
    max_area  = 0
    max_label = 0
    
    success = True

    if (sz == 1):
        success = False

    for label_num in range (1, sz):
        if (stats [label_num, cv2.CC_STAT_AREA] > max_area):
            max_area = stats [label_num, cv2.CC_STAT_AREA]
            max_label = label_num

    result [np.where (labels == max_label)] = 255

    return result, success

#Connected components filtering
#Supports basic conditions, height/width, area, density (area by w * h ratio)

def _in_range (value, low, high):
    if ((value < low  and low  != -1) or
        (value > high and high != -1)):
        return False

    return True

def filter_connected_components (mask, area_low = -1, area_high = -1, hei_low = -1, hei_high = -1,
                wid_low = -1, wid_high = -1, den_low = -1, den_high = -1):
    result = np.array (mask)
    output = cv2.connectedComponentsWithStats (mask, 8, cv2.CV_32S)

    labels_count = output      [0]
    labels       = output      [1]
    stats        = output      [2]
    sz           = stats.shape [0]

    #print (sz)

    for label_num in range (0, sz):
        #print (label_num)

        area   = stats [label_num, cv2.CC_STAT_AREA]
        height = stats [label_num, cv2.CC_STAT_HEIGHT]
        width  = stats [label_num, cv2.CC_STAT_WIDTH]
        density = float (area) / (height * width)
        #print(area)
        if (_in_range (area, area_low, area_high)  == False or
            _in_range (height, hei_low, hei_high)  == False or
            _in_range (width, wid_low, wid_high)   == False or
            _in_range (density, den_low, den_high) == False):
            result [labels == label_num] = 0

        #else:
        #    print ("llalala")
    
    return result
