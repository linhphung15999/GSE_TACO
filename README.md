# GSE_TACO

## Usage
Install libraries
  ```
  pip install requirements.txt
  ```
  
Get trash detection result (dictionary type). Output images will be saved in the same folder with the input image.

  ```
  from taco import get_model_config, trash_detection
  cfg = get_model_config(<path to weights file. Default: './model/weights.pth'>)
  result = trash_detection(cfg,<path to input image>)
  print(result)
  ```
