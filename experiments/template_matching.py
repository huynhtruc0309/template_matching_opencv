import cv2
import itertools
import numpy as np
from pathlib import Path
from preprocess import get_preprocess

def ioa(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    
    return interArea / float(boxAArea)

class TemplateMatching(object):
    def __init__(self, template_folder, method=cv2.TM_CCOEFF_NORMED, n_object=0, score_threshold=0.8, overlap_threshold=0.5):
        '''
        Args:
            template_folder: path to folder containing template images.
            method: template matching method. Only supports cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED. Default: cv2.TM_CCOEFF_NORMED.
            n_object: number of boxes would be returned. If n_object is 0, all boxes found would be returned. Default: 0.
            score_threshold: boxes with scores lower than score_threshold would be ignored. Default: 0.8.
            overlap_threshold: if boxes haves iou greater than overlap_threshold, only one box would be kept. Default: 0.5.
        '''
        super(TemplateMatching, self).__init__()
        assert method in [cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED], 'Only supports cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED.'
        self._load_templates(template_folder)
        self.method = method
        self.n_object = n_object
        self.score_threshold = score_threshold
        self.overlap_threshold = overlap_threshold

    def _match_template(self, image, template):
        '''
        Template matching
        Matching the template in image
        return list of (score, box) 
        '''
        assert len(image.shape) == 2 and len(template.shape) == 2, 'Image and template must be 2D-matrices.'
        result = cv2.matchTemplate(image, template, self.method).astype(float)

        res_w, res_h = result.shape[::-1]
        w, h = template.shape[::-1]

        bboxes = list(itertools.product(range(res_w), range(res_h), [w], [h]))
        scores = result.flatten(order='F')

        bbox_indices = cv2.dnn.NMSBoxes(bboxes, scores, self.score_threshold, self.overlap_threshold, top_k=self.n_object)
        return [(scores[idx], bboxes[idx]) for idx in bbox_indices.flatten()] if len(bbox_indices) else []

    def _load_templates(self, template_folder):
        self.templates = {}
        template_folder = Path(template_folder)
        for template_path in template_folder.glob('**/*.*'):
            template = cv2.imread(str(template_path))
            template = get_preprocess(template)
            if template is not None:
                self.templates[str(template_path)] = template

    def __call__(self, image):
        assert len(image.shape) == 2, 'Image must be a 2D-matrix.'
        result = []

        for _, template in self.templates.items():
            if image.shape[0] > template.shape[0] and image.shape[1] > template.shape[1]:
                bboxes = self._match_template(image, template)
                result.extend(bboxes)
        return result

class MultiScaleTemplateMatching(TemplateMatching):
    def __init__(self, min_scale, max_scale, steps, *args, **kwargs):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.steps = steps
        super(MultiScaleTemplateMatching, self).__init__(*args, **kwargs)

    def _multi_scale_match_template(self, image, template):
        matched_bboxes = []
        for s in np.linspace(self.min_scale, self.max_scale, self.steps):
            resized_template = cv2.resize(template, dsize=(0, 0), fx=s, fy=s)
            if image.shape[0] > resized_template.shape[0] and image.shape[1] > resized_template.shape[1]:
                res = self._match_template(image, resized_template)
                matched_bboxes += res
        bboxes = [bbox for (_, bbox) in matched_bboxes]
        scores = [score for (score, _) in matched_bboxes]
        
        bbox_indices = cv2.dnn.NMSBoxes(bboxes, scores, self.score_threshold, self.overlap_threshold, top_k=self.n_object)
        return [(scores[idx], bboxes[idx]) for idx in bbox_indices.flatten()] if len(bbox_indices) else []
    
    def __call__(self, image):
        assert len(image.shape) == 2, 'Image must be a 2D-matrix.'
        result = []

        for template_name, template in self.templates.items():
            bboxes = self._multi_scale_match_template(image, template)
            result.extend(bboxes)
        return result

def match_template(image, multi_scale=False, *args, **kwargs):
    if len(image.shape) == 3:
        gray_image = get_preprocess(image)
    matcher = MultiScaleTemplateMatching(*args, **kwargs) if multi_scale else TemplateMatching(*args, **kwargs)
    boxes = matcher(gray_image)
    return boxes

def crop_bbox(image, boxes):
    text_color = (0,0,255)
    for score, (x, y, w, h) in boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), text_color, 2)
        cv2.putText(image, str(score), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
    return image 

def calc_acc(json_data, boxes, label_name):
    for shape in json_data['shapes']:
        if shape['label'] in label_name:
            points = np.asarray(shape['points'], dtype=int)
            x1, y1 = points.min(axis=0)
            x2, y2 = points.max(axis=0)
            for _, (x, y, w, h) in boxes:
                if ioa((x1, y1, x2, y2), (x, y, x + w, y + h)) > 0.5:
                    return True
    return False