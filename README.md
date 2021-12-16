# GSE_TACO

## Usage (for Linux)
Install libraries (restart is needed)
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
  cfg = get_model_config(<path to weights file>) #Blank for default
  result = trash_detection(cfg,<path to input image>)
  print(result)
  ```
