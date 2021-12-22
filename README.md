# Trash detection with [TACO](http://tacodataset.org/) dataset using Mask R-CNN ([Detectron2](https://github.com/facebookresearch/detectron2))

## Usage (for Linux)
Install libraries (restarting runtime is needed)
  ```
  pip install -r requirements.txt -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.8/index.html
  ```
  
Download weights file
  ```
  gdown --id 1AYyg_2hZccXRRtVkjn9kcET3ct1y8iHA -O ./model/weights.pth
  ```


Get trash detection result (dictionary type). Output images will be saved in the same folder with the input image.

  ```python
  from taco import get_model_config, trash_detection
  cfg = get_model_config(<path to weights file>) #Blank for default ('./model/weights.pth')
  result = trash_detection(cfg,<path to input image>)
  print(result)
  ```
  
## Training from scratch with GPU
Install libraries (restarting runtime is needed)
  ```
  pip install -r requirements.txt -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.8/index.html
  ```
  
Download weights file
  ```
  gdown --id 1AYyg_2hZccXRRtVkjn9kcET3ct1y8iHA -O ./model/weights.pth
  ```


Get trash detection result (dictionary type). Output images will be saved in the same folder with the input image.

  ```python
  from taco import get_model_config, trash_detection
  cfg = get_model_config(<path to weights file>) #Blank for default ('./model/weights.pth')
  result = trash_detection(cfg,<path to input image>)
  print(result)
  ```
