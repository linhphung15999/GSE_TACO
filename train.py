import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances


# parser = argparse.ArgumentParser(description='')
# parser.add_argument('--num_workers', required=False, default= 4, type = int, help='Number of dataloader workers')
# parser.add_argument('--batch_size', required=False, default= 8, type = int, help='Batch size')
# parser.add_argument('--base_lr', required=False, default= 0.0125, type = float, help='Base learning rate')
# parser.add_argument('--num_iterations', required=False, default= 4500, type = int, help='Total number of training iterations')
# parser.add_argument('--num_classes', required=False, default= 60, type = int, help='Number of data classes')

# args = parser.parse_args()

DatasetCatalog.clear()
register_coco_instances("my_dataset_train", {}, "./data/train.json", "./data")
register_coco_instances("my_dataset_val", {}, "./data/val.json", "./data")
register_coco_instances("my_dataset_test", {}, "./data/test.json", "./data")

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return COCOEvaluator("my_dataset_val", cfg, True, "./output")

def get_train_config(num_workers = 4, batch_size = 8, base_lr = 0.0125, num_iterations = 4500, eval_period=True, num_classes = 60, transfer_learning=''):
  cfg = get_cfg()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cfg.DEVICE = device
  cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
  cfg.DATASETS.TRAIN = ("my_dataset_train",)
  cfg.DATASETS.TEST = ("my_dataset_val",) 
  cfg.DATALOADER.NUM_WORKERS = num_workers

  if transfer_learning == '':
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
  else:
    cfg.MODEL.WEIGHTS = transfer_learning

  cfg.SOLVER.IMS_PER_BATCH = batch_size
  cfg.SOLVER.BASE_LR = base_lr
  cfg.SOLVER.MAX_ITER = (num_iterations)
  cfg.SOLVER.STEPS = (int(num_iterations*0.7),int(num_iterations*0.9))
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (256)
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

  if eval_period:
    cfg.TEST.EVAL_PERIOD = num_iterations/6

  cfg.OUTPUT_DIR = "./output"

  return cfg

def train_model(cfg):
  os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
  trainer = MyTrainer(cfg)
  trainer.resume_or_load(resume=False)
  trainer.train()

def evaluate_model(cfg):
  predictor = DefaultPredictor(cfg)
  evaluator = COCOEvaluator("my_dataset_test", output_dir="./output")
  val_loader = build_detection_test_loader(cfg, "my_dataset_test")
  print(inference_on_dataset(predictor.model, val_loader, evaluator))

# train_model()#num_workers=args.num_workers,batch_size=args.batch_size,base_lr=args.base_lr,num_iterations=args.num_iterations,num_classes=args.num_classes)
