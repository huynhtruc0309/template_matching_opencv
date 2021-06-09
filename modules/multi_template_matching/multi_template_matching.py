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
        # self.split_height = split_height
        # self.split_width = split_width
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
    
    def template_matching(self, listTemplate, image):
        Hits = pd.DataFrame()
        for template in listTemplate:
            Hit = matchTemplates(template, image, N_object=self.N_object,
                                 method=self.method, maxOverlap=self.maxOverlap)
            Hits = Hits.append(Hit, ignore_index=True)
        
        return Hits

    def __call__(self, image, split_height, split_width, thres_score):
        Hits = self.template_matching(self.listTemplate, image)
        
        # print(Hits)
        # Overlay = drawBoxesOnRGB(image, Hits, showLabel=True)
        # cv2.imshow('BBox', cv2.resize(Overlay, (0,0), fx=0.5, fy=0.5))
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        
        for index, row in Hits.iterrows():
            TemplateName, BBox = row['TemplateName'], row['BBox']
            
            # Find the template to split
            for template in self.listTemplate:
                if template[0][0] == TemplateName:
                    image_tem = template[0][1] # Get the image

                    width = image_tem.shape[0]  // split_height
                    height = image_tem.shape[1] // split_width
                    splited_template = [[(TemplateName, image_tem[x : x+width,y : y+height])]
                                        for x in range(0, image_tem.shape[0], width)
                                        for y in range(0, image_tem.shape[1], height)]

                    x, y, h, w = BBox
                    splitedHits = self.template_matching(splited_template, image[y:y+w, x:x+h])
                    
                    # print(splitedHits)
                    score = splitedHits['Score'].mean()
                    if score < thres_score:
                        Hits.drop(index, inplace=True)
                    else:
                        Hits.loc[index, 'Score'] = score
                    
                    # Overlay = drawBoxesOnRGB(image[y:y+w, x:x+h], splitedHits, showLabel=True, boxColor=(255, 0, 0))
                    # cv2.imshow('splitedHits', Overlay)
                    # cv2.waitKey()
        
        cv2.destroyAllWindows()

        return Hits