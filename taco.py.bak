import torch, torchvision
import cv2
import os
from collections import Counter

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.data.catalog import Metadata

def get_model_config(weights_path="./model/weights.pth"):
  num_classes = 60
  cfg = get_cfg()
  cfg.MODEL.DEVICE='cpu'
  cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
  cfg.MODEL.WEIGHTS = weights_path

  return cfg

def trash_detection_no_crop(cfg,input_path,thresholds=[0.3, 0.6]):

  trash_metadata = Metadata()
  objects = ['Aluminium foil', 'Battery', 'Aluminium blister pack', 'Carded blister pack', 'Other plastic bottle', 'Clear plastic bottle', 'Glass bottle', 'Plastic bottle cap', 'Metal bottle cap', 'Broken glass', 'Food Can', 'Aerosol', 'Drink can', 'Toilet tube', 'Other carton', 'Egg carton', 'Drink carton', 'Corrugated carton', 'Meal carton', 'Pizza box', 'Paper cup', 'Disposable plastic cup', 'Foam cup', 'Glass cup', 'Other plastic cup', 'Food waste', 'Glass jar', 'Plastic lid', 'Metal lid', 'Other plastic', 'Magazine paper', 'Tissues', 'Wrapping paper', 'Normal paper', 'Paper bag', 'Plastified paper bag', 'Plastic film', 'Six pack rings', 'Garbage bag', 'Other plastic wrapper', 'Single-use carrier bag', 'Polypropylene bag', 'Crisp packet', 'Spread tub', 'Tupperware', 'Disposable food container', 'Foam food container', 'Other plastic container', 'Plastic glooves', 'Plastic utensils', 'Pop tab', 'Rope & strings', 'Scrap metal', 'Shoe', 'Squeezable tube', 'Plastic straw', 'Paper straw', 'Styrofoam piece', 'Unlabeled litter', 'Cigarette']
  categories = ['paper','plastic','metal','glass','others']
  map = [2,2,2,1,1,1,3,1,2,3,2,2,2,0,0,0,0,0,0,0,0,1,1,3,1,4,3,1,2,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,4,2,4,1,1,0,1,4,0]
  trash_metadata.set(thing_classes = objects)

  result_json = {}
  for thres in thresholds:
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thres  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    im = cv2.imread(input_path)
    outputs = predictor(im)  
    v = Visualizer(im[:, :, ::-1],
                  metadata=trash_metadata,
                  scale=1
                  ,instance_mode=ColorMode.IMAGE
    )

    predicted_classes = outputs["instances"].get('pred_classes')
    
    #Count the number of pieces of trash at the current conf score
    predicted_categories = []
    for e in predicted_classes:
      predicted_categories.append(map[e])
    c_cat = Counter(predicted_categories)
    predicted_classes = predicted_classes.tolist()
    c_cls = Counter(predicted_classes)

    result_at_conf = {}
    for _, cat in enumerate(categories):
      if _ in list(c_cat.keys()):
        result_at_conf[cat]=c_cat[_]
      else:
        result_at_conf[cat]=0
    result_json[str(thres)] = result_at_conf

    #Draw bounding boxes
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    instances = outputs["instances"].to("cpu")
    result = out.get_image()[:, :, ::-1]

    #Write output image back 
    file_name = os.path.splitext(input_path)[0]
    extension = os.path.splitext(input_path)[1]
    new_file = file_name+"_output_confident_"+str(thres)+extension
    cv2.imwrite(new_file,result)

  return result_json

def trash_detection_crop(cfg,input_path,thresholds=[0.3,0.6],crop_size=1024):
  trash_metadata = Metadata()
  objects = ['Aluminium foil', 'Battery', 'Aluminium blister pack', 'Carded blister pack', 'Other plastic bottle', 'Clear plastic bottle', 'Glass bottle', 'Plastic bottle cap', 'Metal bottle cap', 'Broken glass', 'Food Can', 'Aerosol', 'Drink can', 'Toilet tube', 'Other carton', 'Egg carton', 'Drink carton', 'Corrugated carton', 'Meal carton', 'Pizza box', 'Paper cup', 'Disposable plastic cup', 'Foam cup', 'Glass cup', 'Other plastic cup', 'Food waste', 'Glass jar', 'Plastic lid', 'Metal lid', 'Other plastic', 'Magazine paper', 'Tissues', 'Wrapping paper', 'Normal paper', 'Paper bag', 'Plastified paper bag', 'Plastic film', 'Six pack rings', 'Garbage bag', 'Other plastic wrapper', 'Single-use carrier bag', 'Polypropylene bag', 'Crisp packet', 'Spread tub', 'Tupperware', 'Disposable food container', 'Foam food container', 'Other plastic container', 'Plastic glooves', 'Plastic utensils', 'Pop tab', 'Rope & strings', 'Scrap metal', 'Shoe', 'Squeezable tube', 'Plastic straw', 'Paper straw', 'Styrofoam piece', 'Unlabeled litter', 'Cigarette']
  categories = ['paper','plastic','metal','glass','others']
  map = [2,2,2,1,1,1,3,1,2,3,2,2,2,0,0,0,0,0,0,0,0,1,1,3,1,4,3,1,2,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,4,2,4,1,1,0,1,4,0]
  trash_metadata.set(thing_classes = objects)

  result_json = {}
  

  for thres in thresholds:
    initial = True
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thres  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    im = cv2.imread(input_path)
    output_image = im.copy()
    h,w = im.shape[0],im.shape[1]

    num_ims_w = w//crop_size
    num_ims_h = h//crop_size

    for i in range(0,num_ims_h+1):
      x_min = i*crop_size
      if i+1 == num_ims_h:
        x_max = h-1
      else:
        x_max = x_min+crop_size-1
      
      for j in range(0,num_ims_w+1):
        y_min = j*crop_size
        if j+1 == num_ims_w:
          y_max = w-1
        else:
          y_max = y_min+crop_size-1

        input = im[x_min:x_max,y_min:y_max]

        outputs = predictor(input)  
        v = Visualizer(input[:, :, ::-1],
                      metadata=trash_metadata,
                      scale=1
                      ,instance_mode=ColorMode.IMAGE
        )

        predicted_classes = outputs["instances"].get('pred_classes')
        
        #Count the number of pieces of trash at the current conf score
        predicted_categories = []
        for e in predicted_classes:
          predicted_categories.append(map[e])
        c_cat = Counter(predicted_categories)
        predicted_classes = predicted_classes.tolist()
        c_cls = Counter(predicted_classes)

        result_at_conf = {}
        for _, cat in enumerate(categories):
          if _ in list(c_cat.keys()):
            result_at_conf[cat]=c_cat[_]
          else:
            result_at_conf[cat]=0
        
        if initial:
          result_json[str(thres)] = result_at_conf
          initial = False
        else:
          result1 = Counter(result_at_conf)
          result2 = Counter(result_json[str(thres)])
          final_result = dict(result1+result2)
          result_json[str(thres)] = final_result
          

        #Draw bounding boxes
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        instances = outputs["instances"].to("cpu")
        result = out.get_image()[:, :, ::-1]
        output_image[x_min:x_max,y_min:y_max] = result
  
    #Write output image back 
    file_name = os.path.splitext(input_path)[0]
    extension = os.path.splitext(input_path)[1]
    new_file = file_name+"_output_confident_"+str(thres)+extension
    cv2.imwrite(new_file,output_image)
    
  return result_json


def trash_detection(cfg,input_path,thresholds=[0.3,0.6],crop_size=1024):
  if crop_size == 0:
    return trash_detection_no_crop(cfg,input_path,thresholds=[0.3,0.6])
  if crop_size > 0:
    return trash_detection_crop(cfg,input_path,thresholds=[0.3,0.6],crop_size=1024)