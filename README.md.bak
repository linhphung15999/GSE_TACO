# GSE_TACO

## Usage
Install libraries
  ```
  pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
  ```
  
Download weights file
  ```
  gdown --id 1AYyg_2hZccXRRtVkjn9kcET3ct1y8iHA -O ./model/weights.pth
  ```


Get trash detection result (dictionary type). Output images will be saved in the same folder with the input image.

  ```
  from taco import get_model_config, trash_detection
  cfg = get_model_config(<path to weights file. Default: './model/weights.pth'>)
  result = trash_detection(cfg,<path to input image>)
  print(result)
  ```
