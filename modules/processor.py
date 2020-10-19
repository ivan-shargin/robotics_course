#!/usr/bin/env python

basketball = False
obstacles = True
import image_processing
import cv2
import json
import numpy as np
import collections
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

#from matplotlib.figure import Figure
plt.switch_backend('Agg')

#TODO: Implement simultaneous stages displaying in single window
#TODO: Document the logics behind the project architecture, filters creation
#TODO: Fix issues arised because of the folder reorganization
#TODO: In HSV H represents circular matter. There should be an option to
#      set [240 -- 5] range, for instance
#Integrate ilastic - tool for segmentation in the detection process
#Filter returning lowest point of the connected component

#TODO/REFACTOR
#Move parameters parsing into the filters constructors from Detector constructor
#Refactor parameters extraction in find_obstacles_distances creation, automate
#      types number obtainment
#Move code to standard Python style
#Move filters to a separate file
#Move image processing (if any) from detectors.py to image_processing.py
#Adopt tests to new Detector usage (with multiple objects)
#Refactor Detector, particularly multi-object part
#Move particular cases of get_stages_steps into the filters
#    for the generalization

#TODO/FUTURE
#Make up a way to plug filters in another filters.
#             Closest obstacle finder uses inrange, morphology, connected components filtering,
#             iterating
#Filter can store its parameters in a dictionary
#Implement IO library with picture, video, camera, ROS input handling
#Online vizualizing tool for filters
#Metaconfig with a list of configs (?)
#Tuning of Inrange (ranges.py) to .json
#Different verbosity levels logging system

#Filter is an img-to-img transformation; generally from any shape to any shape
#Previous comment was written in the very beginning of the development
#Filter is an anything-to-anything transformation

class Filter:
    def __init__(self, name_):
        self.name = name_
        self.success = []
    
    def apply (self, img):
        return img

class custom_operation (Filter):
    def __init__ (self, operation_, operation_name_):
        Filter.__init__ (self, operation_name_)

        self.operation = operation_

    def apply (self, img):
        return self.operation (img)

class morphology (Filter):
    operations = {}

    def __init__ (self, operation_, ker_sz_ = 3):
        Filter.__init__ (self, "morphology")

        self.ker_sz    = ker_sz_
        self.operation = operation_

        self.operations.update ({"erode"  : cv2.MORPH_ERODE})
        self.operations.update ({"dilate" : cv2.MORPH_DILATE})
        self.operations.update ({"open"   : cv2.MORPH_OPEN})
        self.operations.update ({"close"  : cv2.MORPH_CLOSE})
        
    def apply (self, img):
        kernel = np.ones ((self.ker_sz, self.ker_sz), np.uint8)
        return cv2.morphologyEx (img, self.operations [self.operation], kernel)

class gaussian_blur (Filter):
    def __init__ (self, ker_sz_ = 3):
        Filter.__init__ (self, "Gaussian Blur")

        self.ker_sz = ker_sz_
        
    def apply (self, img):
        return cv2.GaussianBlur (img, (self.ker_sz, self.ker_sz), 0)

class find_object (Filter):
    def __init__ (self, img_):
        Filter.__init__ (self, "find object")

        self.img = img_

        self.compute_keypoints ()

    def compute_keypoints (self):
        return 5
        
    def apply (self, img):
        return cv2.GaussianBlur (img, (ker_sz, kre_sz), 0)

class crop (Filter):
    def __init__ (self, x1 = 0, y1 = 0, x2 = 100, y2 = 100):
        Filter.__init__ (self, "crop")

        self.parameters = {"x1" : x1,
                           "y1" : y1,
                           "x2" : x2,
                           "y2" : y2,
                           }
        
    def set_parameter (self, parameter, new_value):
        if (parameter not in self.parameters.keys ()):
            print ("Warning: no such parameter ", parameter, "adding")

        self.parameters.update ({parameter, new_value})

    def apply (self, img):
        return img [self.parameters ["y1"] : self.parameters ["y2"],
                    self.parameters ["x1"] : self.parameters ["x2"], :].copy ()

class gamma_correction (Filter):
    def __init__ (self, gamma_ = 1):
        Filter.__init__ (self, "gamma correction")

        self.gamma = gamma_
        
    def set_gamma (self, new_gamma):
        if (new_gamma == 0):
            return

        self.gamma = new_gamma

    def apply (self, img):
        #print ("kek", img)
        inv_gamma = 1.0 / self.gamma

        table = np.array([((i / 255.0) ** inv_gamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")

        #print ("img shape", img)

        return cv2.LUT (img, table)

class resize (Filter):
    def __init__ (self, downscale_factor_ = 2, new_x_ = 100, new_y_ = 100):
        Filter.__init__ (self, "downscale img by factor")

        self.downscale_factor = downscale_factor_
        self.new_x = new_x_
        self.new_y = new_y_
        
    def apply (self, img):
        #TODO: implement fitting into the given shape

        sh = img.shape

        return cv2.resize (img, (int (sh [1] / self.downscale_factor), int (sh [0] / self.downscale_factor)))

class colorspace_to_colorspace (Filter):
    transforms = {}

    def __init__ (self, from_, to_):
        Filter.__init__ (self, "colorspace2colorspace")

        self.source_colorspace = from_
        self.target_colorspace = to_

        self.transforms.update ({"RGB2BGR"  : cv2.COLOR_RGB2BGR})
        
        self.transforms.update ({"RGB2GRAY" : cv2.COLOR_RGB2GRAY})
        self.transforms.update ({"GRAY2RGB" : cv2.COLOR_GRAY2RGB})
        
        self.transforms.update ({"RGB2HSV"  : cv2.COLOR_RGB2HSV})
        self.transforms.update ({"HSV2RGB"  : cv2.COLOR_HSV2RGB})
        
        self.transforms.update ({"RGB2YCrCb"  : cv2.COLOR_RGB2YCrCb})
        self.transforms.update ({"YCrCb2RGB"  : cv2.COLOR_YCrCb2RGB})
        
        #self.transforms.update ({"HSV2YCrCb"  : cv2.COLOR_HSV2YCrCb})
        #self.transforms.update ({"YCrCb2HSV"  : cv2.COLOR_YCrCb2HSV})
        
    def apply (self, img):
        return cv2.cvtColor (img, self.transforms [self.source_colorspace +
                             "2" + self.target_colorspace])

class inrange (Filter):
    def __init__ (self, low_th_, high_th_):
        Filter.__init__ (self, "inrange")

        self.set_ths (low_th_, high_th_)
        
    def set_ths (self, low_th_, high_th_):
        self.low_th  = low_th_
        self.high_th = high_th_

    def apply (self, img):
        return cv2.inRange (img, self.low_th, self.high_th)

#find bbox of the connected component with maximal area
class max_area_cc_bbox (Filter):
    def __init__ (self, bbox_num_ = 1):
        Filter.__init__ (self, "max_area_cc_bbox")
        self.bbox_num = bbox_num_

    def apply (self, img):
        result, success_curr = image_processing.find_max_bounding_box (img, self.bbox_num)

        self.success.append (success_curr)

        #print ("max area cc bbox", result, success_curr)

        return result

#leave maximal area connected component
class leave_max_area_cc (Filter):
    def __init__ (self):
        Filter.__init__ (self, "leave_max_area_cc")

    def apply (self, img):
        result, success_curr = image_processing.leave_max_connected_component (img)

        self.success.append (success_curr)

        return result

#returns bottom point of the bbox, middle by x axis
class bottom_bbox_point (Filter):
    def __init__ (self):
        Filter.__init__ (self, "bottom_bbox_point")

    def apply (self, img):
        tl, br = img

        x = int ((tl [0] + br [0]) / 2)
        y = br [1]

        return (x, y)

#returns bottom point of the cc
class bottom_cc_point (Filter):
    def __init__ (self):
        Filter.__init__ (self, "bottom_cc_point")

    def apply (self, img):
        contours, hierarchy = cv2.findContours (img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cont = []

        for c in contours [0]:
            cont.append (c[0])

        x = 5
        y = 5

        if (len (cont) != 0):
            sor = sorted (cont, key=lambda con: con [0])

            x = sor [-1] [0]
            y = sor [-1] [1]

        return (x, y)

#incapsulates basic processing function
class filter_connected_components (Filter):
    def __init__ (self, area_low_ = -1, area_high_ = -1, hei_low_ = -1, hei_high_ = -1,
               wid_low_ = -1, wid_high_ = -1, den_low_ = -1, den_high_ = -1):
        Filter.__init__ (self, "filter_connected_components")

        self.area_low  = area_low_
        self.area_high = area_high_
        self.hei_low   = hei_low_
        self.hei_high  = hei_high_
        self.wid_low   = wid_low_
        self.wid_high  = wid_high_
        self.den_low   = den_low_
        self.den_high  = den_high_

    def apply (self, img): #, area_low = -1, area_high = -1, hei_low = -1, hei_high = -1,
               #wid_low = -1, wid_high = -1, den_low = -1, den_high = -1):
        return image_processing.filter_connected_components (img, self.area_low,
               self.area_high, self.hei_low, self.hei_high, self.wid_low,
               self.wid_high, self.den_low, self.den_high)

#finds pixel distance (by y axis) from the bottom of the frame to the closest obstacle
#returns list of points (x, y, obstacle type)
class find_obstacles_distances (Filter):
    def __init__ (self, ranges_):
        Filter.__init__ (self, "find_obstacles_distances")
        self.ranges = ranges_
        self.inrange_filter = inrange ((0, 0, 0), (255, 255, 255))
        #self.cc_filter = filter_connected_components ()

    def _get_obstacles_dists (self, obstacles):
        obstacles_flipped = cv2.flip (obstacles, 0)
        distances = np.argmax (obstacles_flipped, axis=0)
	
        for i in range (len (distances)):
            if (distances [i] == 0 and obstacles_flipped [distances [i]] [i] == 0):
                distances [i] = -2

        #print ("fuck")
        #print (distances)

        return distances

    def apply (self, img):
        result = []
        labels = []

        self.obstacles_stages = {}

        sh = img.shape

        for i in range (sh [1]):
            labels.append (0)

        filled = False

        #save all masks, not last
        for range_num in range (len (self.ranges)):
            range_ = self.ranges [range_num]

            self.inrange_filter.set_ths (range_ [0], range_ [1])
            mask = self.inrange_filter.apply (img)
            self.obstacles_stages.update ({"0" : image_processing.to_three (mask)})

            mask = self.cc_filter.apply (mask)
            self.obstacles_stages.update ({"1" : image_processing.to_three (mask)})

            op_ker = 3
            cl_ker = 3

            morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((op_ker, op_ker),np.uint8))
            self.obstacles_stages.update ({"2" : image_processing.to_three (mask)})

            morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, np.ones((cl_ker,cl_ker),np.uint8))
            self.obstacles_stages.update ({"3" : image_processing.to_three (mask)})

            temp_result = self._get_obstacles_dists (morph)

            if (filled == False):
                filled = True
                result = temp_result.copy ()

            for i in range (len (temp_result)):
                if (temp_result [i] <= result [i] and temp_result [i] != -2) or\
                   (result [i] == -2 and temp_result [i] > 0):
                    result [i] = temp_result [i]
                    labels [i] = range_num + 1

        for i in range (len (result)):
            if (result [i] != -2):
                result [i] = sh [0] - result [i]
                #temp_result [i] = temp_result [i]


        #for i in range (sh [1]):
        #    result.append ((i, 200, i))

        #print ("ll")
        #print (labels)

        return result, labels


class calc_distribution(Filter):
    def __init__(self):
        Filter.__init__ (self, "calc_distribution")

    def apply(self, img):
        first_channel = cv2.calcHist([img[:,:,0]],[0],None,[256],[0,256])
        second_channel = cv2.calcHist([img[:,:,1]],[0],None,[256],[0,256])
        third_channel = cv2.calcHist([img[:,:,2]],[0],None,[256],[0,256])
        return (first_channel, second_channel, third_channel)

class background_subtraction(Filter):
    def __init__(self, frame = None, background_frame_skip_ = 10):
        Filter.__init__(self, "background_subtraction")
        self.background_frame_set = background_frame_skip_
        
        if (frame is not None):
            self.update_background_frame(frame)

    def update_background_frame(self, frame):
        self.background_frame = frame
        #print (self.background_frame_set)
        self.background_frame_set -= 1

    def get_background_frame (self):
        return self.background_frame

    def apply(self, img):
        if (self.background_frame_set != 0):
            self.update_background_frame (img)
            return img

        #print ("a")

        subtract = np.absolute(self.background_frame - img)

        #cv2.imshow ("mlem", subtract)

        #single_channel = cv2.cvtColor (subtract, cv2.COLOR_RGB2GRAY)

        return subtract

class contour_approximation(Filter):
    def __init__(self):
        Filter.__init__(self, "contour_approximation")

    def apply(self, img):
        contours, hierarchy = cv2.findContours (img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours [0]

        #epsilon = 0.1 * cv2.arcLength (cnt, True)
        #approx = cv2.approxPolyDP (cnt, epsilon, True)

        approx = cv2.convexHull (cnt)

        epsilon = 0.05 * cv2.arcLength (approx, True)
        approx = cv2.approxPolyDP (approx, epsilon, True)

        #print ("appr", approx)

        return approx

class warp_perspective(Filter):
    def __init__(self, x1 = 463, y1 = 442, x2 = 691, y2 = 310, x3 = 761, y3 = 452, x4 = 478, y4 = 586):
        Filter.__init__(self, "warp_prespective")

        parameters = {"x1": x1,
                      "y1": y1,
                      "x2": x2,
                      "y2": y2,
                      "x3": x3,
                      "y3": y3,
                      "x4": x4,
                      "y4": y4,
                      }

        self.set_parameters (parameters)

    def set_parameters (self, new_parameters):
        self.parameters = new_parameters

    def set_parameters_as_contour (self, new_parameters):
        parameters = {"x1": new_parameters [0] [0] [0],
                      "y1": new_parameters [0] [0] [1],
                      "x2": new_parameters [1] [0] [0],
                      "y2": new_parameters [1] [0] [1],
                      "x3": new_parameters [2] [0] [0],
                      "y3": new_parameters [2] [0] [1],
                      "x4": new_parameters [3] [0] [0],
                      "y4": new_parameters [3] [0] [1],
                      }

        self.set_parameters (parameters)

    def apply(self, img):
        def order_points(pts):
            # initialzie a list of coordinates that will be ordered
            # such that the first entry in the list is the top-left,
            # the second entry is the top-right, the third is the
            # bottom-right, and the fourth is the bottom-left
            rect = np.zeros((4, 2), dtype="float32")
            # the top-left point will have the smallest sum, whereas
            # the bottom-right point will have the largest sum
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            # now, compute the difference between the points, the
            # top-right point will have the smallest difference,
            # whereas the bottom-left will have the largest difference
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            # return the ordered coordinates
            return rect

        def four_point_transform(image, pts):
            # obtain a consistent order of the points and unpack them
            # individually
            rect = order_points(pts)
            (tl, tr, br, bl) = rect
            # compute the width of the new image, which will be the
            # maximum distance between bottom-right and bottom-left
            # x-coordiates or the top-right and top-left x-coordinates
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))
            # compute the height of the new image, which will be the
            # maximum distance between the top-right and bottom-right
            # y-coordinates or the top-left and bottom-left y-coordinates
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))
            # now that we have the dimensions of the new image, construct
            # the set of destination points to obtain a "birds eye view",
            # (i.e. top-down view) of the image, again specifying points
            # in the top-left, top-right, bottom-right, and bottom-left
            # order
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype="float32")
            # compute the perspective transform matrix and then apply it
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
            # return the warped image
            return warped

        #image = cv2.imread("../data/images/warped.png")

        # pts = np.array(eval(args["coords"]), dtype = "float32")
        p = self.parameters
        pts = np.array([(p["x1"], p["y1"]), (p["x2"], p["y2"]), (p["x3"], p["y3"]), (p["x4"], p["y4"])], dtype="float32")

        # apply the four point tranform to obtain a "birds eye view" of
        # the image
        warped = four_point_transform (img, pts)

        return warped

#------------------------------------------------------

#Detector incapsulates the whole detection process, which practically means image processing
#to certain stage and consequent extraction of the required object

#Any detector (color-based, NN, template-based) is supposed to
#be set as a sequence of filters. The idea is partially stolen from NNs

class Processors:
    #filters = []
    processors = {}

    #processing stages (for debugging purposes)
    stages  = {}

    def __init__(self):
        pass

    def _init_cc_filter (self, filter):
        area_low  = int (filter ["area_low"])
        area_high = int (filter ["area_high"])
        hei_low   = int (filter ["hei_low"])
        hei_high  = int (filter ["hei_high"])
        wid_low   = int (filter ["wid_low"])
        wid_high  = int (filter ["wid_high"])
        den_low   = int (filter ["den_low"])
        den_high  = int (filter ["den_high"])

        cc_filter = filter_connected_components (area_low, area_high,
                    hei_low, hei_high, wid_low, wid_high, den_low, den_high)

        return cc_filter

    def __init__(self, detector_filename = "-111"):
        if (detector_filename == "-111"):
            self.processors.update ({"a" : []})
            return

        with open (detector_filename) as f:
            data = json.load(f)
        competition = data["competition"]
        
        with open (detector_filename) as f:
            data = json.load(f)
        
        for processor in data ["processors"]:
            processor_name = processor ["name"]

            self.add_processor (processor_name)

            for filter in processor ["filters"]:
                filter_name = filter ["name"]
                print(filter_name)

                if (filter_name == "inrange"):
                    low_th   = (int (filter ["l1"]), int (filter ["l2"]), int (filter ["l3"]))
                    high_th  = (int (filter ["h1"]), int (filter ["h2"]), int (filter ["h3"]))
                    new_filter = inrange (low_th, high_th)

                if (filter_name == "max_area_cc_bbox"):
                    new_filter = max_area_cc_bbox ()

                if (filter_name == "bottom_bbox_point"):
                    new_filter = bottom_bbox_point ()

                if (filter_name == "colorspace2colorspace"):
                    source = filter ["from"]
                    target = filter ["to"]

                    new_filter = colorspace_to_colorspace (source, target)

                if (filter_name == "filter_connected_components"):
                    new_filter = self._init_cc_filter (filter)

                if (filter_name == "find_obstacles_distances"):
                    types_num = int (filter ["types_num"])
                
                    ranges = []

                    for i in range (types_num):
                        type_num = str (i + 1)

                        low_th   = (int (filter [type_num + "l1"]),
                                    int (filter [type_num + "l2"]),
                                    int (filter [type_num + "l3"]))

                        high_th  = (int (filter [type_num + "h1"]),
                                    int (filter [type_num + "h2"]),
                                    int (filter [type_num + "h3"]))

                        ranges.append ((low_th, high_th))

                    new_cc_filter = self._init_cc_filter (filter)

                    new_filter = find_obstacles_distances (ranges)

                    new_filter.cc_filter = new_cc_filter

                self.add_filter (new_filter, detector_name, filter_name)
        
    def add_processor (self, processor_name):
        self.processors.update ({processor_name : collections.OrderedDict ()})

    def add_filter (self, new_filter, processor_name, filter_name):
        self.processors [processor_name] [filter_name] = new_filter
    
    def get_stages (self, detector_name):
        return self.stages [detector_name]

    def get_stages_picts (self, processor_name, filters_list = []):
        stages_picts = []

        #print ("stages: ", self.stages [processor_name])

        for i in range (len (self.stages [processor_name])):
            if (i == 0 and ("initial" in filters_list or len (filters_list) == 0)):
                stages_picts.append (self.stages [processor_name] [i])
                continue

            filter_usr_name = list (self.processors [processor_name].keys ()) [i - 1]

            if (len (filters_list) != 0 and filter_usr_name not in filters_list):
                #print ("skip", filter_usr_name)
                continue

            filter = self.processors [processor_name] [filter_usr_name]
            filter_type = filter.name

            #print ("filter type", filter_type)

            stage = self.stages [processor_name] [i]

            #print ("filter_usr_name", filter_usr_name)
            #print ("stage", stage)

            if (filter_type == "max_area_cc_bbox"):
                first_img = self.stages [processor_name] [0].copy ()

                #for j in range (0, i):
                #    if (self.processors [processor_name] [j].name () == "crop"):
                #        params = self.processors [processor_name] [j].parameters
                #        x_shift += params ["x1"]
                #        y_shift += params ["y1"]

                rect_marked = first_img.copy ()

                stage = self.stages [processor_name] [-1]

                for s in stage:
                    #print (s)
                    #print (s [0])
                    rect_marked = cv2.rectangle (rect_marked, s [0], s [1], (100, 200, 10), 5)

                stages_picts.append (rect_marked)

            elif (filter_type == "crop"):
                prev_img = self.stages [processor_name] [i - 1].copy ()

                x1 = self.processors [processor_name] [filter_usr_name].parameters ["x1"]
                y1 = self.processors [processor_name] [filter_usr_name].parameters ["y1"]
                x2 = self.processors [processor_name] [filter_usr_name].parameters ["x2"]
                y2 = self.processors [processor_name] [filter_usr_name].parameters ["y2"]

                #rect_marked = cv2.rectangle (prev_img, (x1, y1), (x2, y2), (10, 200, 100), 5)

                rect_marked = prev_img [y1:y2, x1:x2, :]

                stages_picts.append (rect_marked)

            elif (filter_type == "find_obstacles_distances"):
                prev_img = self.stages [processor_name] [0].copy ()

                obstacles_stages = self.processors [processor_name] [i - 1] [0].obstacles_stages

                #print ("lalalaaa")
                #print (self.processors [processor_name] [i-1])

                for i in range (len (obstacles_stages)):
                    stages_picts.append (obstacles_stages [str (i)])

                obstacle_pixels, labels = stage

                for i in range (len (obstacle_pixels)):
                    x = i
                    y = obstacle_pixels [i]

                    type = labels [i]

                    rect_marked = cv2.circle (prev_img, (x, y), 5, (10 + type * 50, 150 - type * 90, 190 + type * 140),
                                              thickness = -1)

                stages_picts.append (rect_marked)

            elif (filter_type == "warp_prespective"):
                prev_img = self.stages[processor_name][i - 1].copy()

                x1 = self.processors[processor_name][filter_usr_name].parameters["x1"]
                y1 = self.processors[processor_name][filter_usr_name].parameters["y1"]
                x2 = self.processors[processor_name][filter_usr_name].parameters["x2"]
                y2 = self.processors[processor_name][filter_usr_name].parameters["y2"]
                x3 = self.processors[processor_name][filter_usr_name].parameters["x3"]
                y3 = self.processors[processor_name][filter_usr_name].parameters["y3"]
                x4 = self.processors[processor_name][filter_usr_name].parameters["x4"]
                y4 = self.processors[processor_name][filter_usr_name].parameters["y4"]

                pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
                pts = pts.reshape((-1, 1, 2))
                isClosed = True
                color = (255, 0, 0)
                thickness = 2

                polygon_marked = cv2.polylines (prev_img, [pts], isClosed, color, thickness)

                stages_picts.append (polygon_marked)

            elif (filter_type == "contour_approximation"):
                first_img = self.stages [processor_name] [0].copy ()

                #x1 = self.processors[processor_name][filter_usr_name].parameters["x1"]
                #pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
                #pts = pts.reshape((-1, 1, 2))
                #isClosed = True
                #color = (255, 0, 0)
                #thickness = 20

                #polygon_marked = cv2.polylines (first_img, stage, isClosed, color, thickness)

                polygon_marked = cv2.drawContours (first_img, stage, -1, (0, 255, 0), 3)

                stages_picts.append (polygon_marked)

            elif (filter_type == "calc_distribution"):
                #prev_img = self.stages[processor_name][0].copy()
                hists = self.stages[processor_name][i]
                fig, axs = plt.subplots(len(hists), 1)
                for j, (hist, ax) in enumerate(zip(hists, axs)):
                    ax.plot(hist)
                    ax.set_title(str(j+1) + " channel")
                #fig.canvas.draw()
                plt.savefig("temp.jpg")
                X = plt.imread ("temp.jpg")
                #print ("fig canvas", fig.canvas.renderer.buffer_rgba().shape)
                #X = np.frombuffer(fig.canvas.renderer.buffer_rgba(), dtype=np.uint8)
                #X = np.reshape (X, (800, 512, 3))
                #print ("arr shape", X.shape)
                plt.close('all')
                stages_picts.append(X)

            #elif (filter_type == "background_subtraction"):
            #    stages_picts.append (filter.get_background_frame ())

            else:
                stages_picts.append (stage)

        return stages_picts

    def process(self, image, processor_name):
        self.stages [processor_name] = []
        self.stages [processor_name].append (image)

        success = True

        for name, filter in self.processors [processor_name].items ():
            previous_step = self.stages [processor_name] [-1]

            curr_state = filter.apply (previous_step)

            #print ("name, filter, curr_state", name, filter, curr_state)

            self.stages [processor_name].append (curr_state)

            if (len (filter.success) != 0 and filter.success [-1] == False):
                success = False

        return self.stages [processor_name] [-1], success
        
if __name__ == "__main__":
    rospy.init_node('detectors')
    conf_file = rospy.get_param('~conf_file')
    print(conf_file)
    detector = Detector(conf_file)
    rospy.spin()
