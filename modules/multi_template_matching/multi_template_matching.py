import MTM
from MTM import matchTemplates, drawBoxesOnRGB
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
class MultiTemplateMatching(object):
    def __init__(self, input_page, template_folder, templates, N_object, method, maxOverlap, split_number):
        super(MultiTemplateMatching, self).__init__()
        self.input_page = input_page
        self.template_folder = template_folder
        self.templates = templates
        self.N_object = N_object
        self.method = method
        self.maxOverlap = maxOverlap
        self.split_number = split_number
        self.listTemplate = self.load_templates()

    def load_templates(self):
        listTemplate = []
        for template in self.templates[self.input_page]:
            tem_path = Path(self.template_folder, self.input_page, template['label'], template['name'])
            print(tem_path)
            image = cv2.imread(str(tem_path))
            template = [(template['label'], image)]
            listTemplate.append(template)
        return listTemplate

    def matching_splited_template(self, template, image, k=3):
        width = template.shape[0]//k
        height = template.shape[1]//k
        tiles = [template[x:x+width,y:y+height] for x in range(0,template.shape[0],width) for y in range(0,template.shape[1],height)]

        score = 0.0
        for tile in tiles:
            Hit = matchTemplates(tile, image, N_object=self.N_object,
                                 method=self.method, maxOverlap=self.maxOverlap)
            score += Hit['Score'][0]

        return score/float(len(tiles))

    def __call__(self, image):
        Hits = pd.DataFrame()
        for template in self.listTemplate:
            Hit = matchTemplates(template, image, N_object=self.N_object,
                                 method=self.method, maxOverlap=self.maxOverlap)
            print(Hit['BBox'])
            x, y, h, w = Hit['BBox'][0]
            score = self.matching_splited_template(template, image[y:y+w, x:x+h], self.split_number)
            print(score)
            Hits = Hits.append(Hit)
            
        return Hits