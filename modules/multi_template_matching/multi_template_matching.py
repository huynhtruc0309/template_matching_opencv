import MTM
from MTM import matchTemplates, drawBoxesOnRGB
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
class MultiTemplateMatching(object):
    def __init__(self, input_page, template_folder, templates, N_object, method, maxOverlap):
        super(MultiTemplateMatching, self).__init__()
        self.input_page = input_page
        self.template_folder = template_folder
        self.templates = templates
        self.N_object = N_object
        self.method = method
        self.maxOverlap = maxOverlap
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

    def __call__(self, image):
        Hits = pd.DataFrame()
        for template in self.listTemplate:
            Hit = matchTemplates(template, image, N_object=self.N_object,
                                 method=self.method, maxOverlap=self.maxOverlap)
            Hits = Hits.append(Hit)
            
        return Hits