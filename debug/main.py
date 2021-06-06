import json
import time 
import pandas as pd
from pathlib import Path
import cv2
from template_matching import match_template, calc_acc, crop_bbox

if __name__ == "__main__":
    multi_scale = False
    img_folder = Path('2_CPP_Page2/samples')
    img_output = Path('2_CPP_Page2/outputs')
    min_scale, max_scale, steps = 0.7, 1.7, 11

    aver_time = []
    num_true = []
    num_false = []
    acc = []
    num_samples = len(list(img_folder.glob('*.jpg')))
    label_names = ['2', '4', '6', '8', '10', '12', '15', '18']
    
    for label_name in label_names:
        template_folder = '2_CPP_Page2/templates/' + label_name
        true_cases = Path(img_output, label_name, 'true_cases')
        false_cases = Path(img_output, label_name, 'false_cases')
        true_cases.mkdir(parents=True, exist_ok=True)
        false_cases.mkdir(parents=True, exist_ok=True)
        
        num_acc = 0
        time_process = 0.0
        for i, img_path in enumerate(img_folder.glob('*.jpg')):
            image = cv2.imread(str(img_path))

            start = time.time()
            if multi_scale:
                boxes = match_template(image, multi_scale, min_scale, max_scale, steps, template_folder, n_object=1, score_threshold=0)
            else:
                boxes = match_template(image, multi_scale, template_folder, n_object=1, score_threshold=0)
            end = time.time()
            
            time_process += end - start
            print('Time process:', end - start, 's')
            
            # calculate accuracy
            json_pth = str(img_path).replace('jpg', 'json')
            with open(json_pth) as json_file:
                json_data = json.load(json_file)
            
            if calc_acc(json_data, boxes, label_name):
                num_acc += 1
                bbox_img = crop_bbox(image, boxes)
                cv2.imwrite(str(Path(true_cases, img_path.name)), bbox_img)
            else:
                print('Wrong matching', str(img_path))
                bbox_img = crop_bbox(image, boxes)
                cv2.imwrite(str(Path(false_cases, img_path.name)), bbox_img)

        print('===>', num_acc, '/', num_samples, 'images got right')
        num_true.append(num_acc)
        num_false.append(num_samples - num_acc)
        aver_time.append(round(time_process/float(num_samples), 2))
        acc.append(num_acc/num_samples)
    
    df = pd.DataFrame({'Label name': label_names,
                       'Average time':aver_time, 
                       'Number of True cases':num_true, 
                       'Number of False cases':num_false, 
                       'Accuracy':acc})
    df.to_excel(str(Path(img_output,'report.xlsx')), index=False)