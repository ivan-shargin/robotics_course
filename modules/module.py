import numpy as np
import json
import cv2
from input_output import Source as Source_
from processor import Processors as Processor_
import math

class Module:
    def __init__(self, logger=None, type_name="module"):
        self.logger = logger
        self.type_name = type_name

    def apply(self, input):
        pass

    def draw(self, img):
        return img

class Vision(Module):
    def __init__(self, params):
        Module.__init__(self, "vision")
        self.processor = Processor_(params["config_path"])

    def apply(self, input):
        #print("Vision: ", input["camera"].shape)

        _, frame = input["camera"]

        result, _ = self.processor.process(frame, "basket detector")

        self.resultant_frame = self.processor.get_stages_picts ("basket detector") [-1]

        return result

    def draw (self, img):
        return self.resultant_frame

class Source(Module):
    def __init__(self, type_name):
        Module.__init__(self, type_name)

    def apply(self, input):
        print("Source.apply() called")

class Camera(Source):
    def __init__(self, path):
        Source.__init__ (self, "camera")
        self.source = Source_ (path)

    def apply(self, input):
        self.frame = self.source.get_frame ()
        return self.frame

    def draw(self, img):
        return self.frame [1]

class Keyboard(Source):
    def __init__(self, params):
        Module.__init__(self, "keyboard")

    def apply(self, input):
        return cv2.waitKey(50) & 0xFF

class Container(Module):
    def __init__(self, config, logger=None):
        Module.__init__(self, logger, type_name="container")

        self.config = config

        self.modules = {}
        #self.load_configuration()

        self.naming = {"camera" : Camera,
                       "image processor" : Vision,
                       "keyboard" : Keyboard}

    def load_configuration(self):
        with open (self.config) as f:
            data = json.load(f)

        for module in data["modules"]:
            module_name = module["name"]

            print (module_name)

            req = []
            if ("requires" in module.keys()):
                req = [x for x in module["requires"]]

            #print ("req", module_name, req)

            module_embedded = {"module" : self.naming [module_name] (module),
                               "name" : module_name,
                               "last_result" : "",
                               "requires" : req,
                               "updated" : False}

            self.modules.update ({module_name : module_embedded})

    def close(self):
        cv2.destroyAllWindows()

    def draw (self, img):
        h, w, _ = img.shape
        r = int (min (h, w) / 3)

        def calc_xy (i, r, w, h):
            x = int(w / 2 + math.cos(i * 2 * math.pi / len(self.modules)) * r)
            y = int(h / 2 + math.sin(i * 2 * math.pi / len(self.modules)) * r)

            return x, y

        for i, (module_name, module) in enumerate(self.modules.items()):
            #print ("moduuu", i, module_name, module)

            x, y = calc_xy (i, r, w, h)

            cv2.circle (img, (x, y), 20, (20, 40, 199), -1)
            cv2.putText(img, module["name"], (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 220, 10), 1, cv2.LINE_AA)

            img [y : y + 100, x : x + 100] = cv2.resize (module["module"].draw(img), (100, 100))

            for req in module["requires"]:
                ind = list(self.modules.keys()).index(req)
                x1, y1 = calc_xy (ind, r, w, h)

                cv2.arrowedLine(img, (x1, y1), (x, y), (100, 120, 22), 1, cv2.LINE_AA)

        for _, module in self.modules.items():
            module ["module"].draw (img)

        return img

    def update_modules(self):
        updated_num = 0

        while (updated_num < len (self.modules)):
            for i, (module_name, module) in enumerate(self.modules.items()):
                if (self.modules [module_name] ["updated"] == True):
                    continue

                req_updated = True
                req_values = {}

                for req in self.modules [module_name] ["requires"]:
                    if (self.modules [req] ["updated"] == False):
                        req_updated = False
                        break

                    req_values.update({req : self.modules [req] ["last_result"]})

                if (req_updated == False):
                    continue

                new_result = self.modules[module_name]["module"].apply(req_values)
                self.modules[module_name].update({"last_result": new_result})

                updated_num += 1
                module ["updated"] = True

                #print ("updated ", module_name, i)

        for _, module in self.modules.items():
            module ["updated"] = False

    def tick(self):
        return self.apply({})

    def apply(self, input):
        #print("tick")

        self.update_modules()

        if ("keyboard" in self.modules.keys()):
            if (self.modules["keyboard"] ["last_result"] == ord('q')):
                return True

        canvas = np.ones ((600, 600, 3), np.uint8) * 50

        self.canvas = self.draw (canvas)
        cv2.imshow ("frame", self.canvas)

        return False
