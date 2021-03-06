# Trash detection with [TACO](http://tacodataset.org/) dataset using Mask R-CNN ([Detectron2](https://github.com/facebookresearch/detectron2))

## Usage (for inference on Linux)
Install libraries (restarting runtime is needed)
  ```
  pip install -r requirements_cpu.txt -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.8/index.html
  ```
  
Download weights file
  ```
  gdown --id 1AYyg_2hZccXRRtVkjn9kcET3ct1y8iHA -O ./model/weights.pth
  ```


Get trash detection result (dictionary type). Output images will be saved in the same folder with the input image.
  ```python
  from taco import get_model_config, trash_detection
  cfg = get_model_config(<path to weights file>) #Blank for default ('./model/weights.pth')
  result = trash_detection(cfg,<path to input image>,crop_size) #crop_size: 1024, 2048,... Default = 0 (no cropping).
  print(result)
  average_trash_amount = result['0.3']['trash_amount']*0.4+result['0.6']['trash_amount']*0.6  #confident score 0.3 is 40% weighted and  confident score 0.6 is 60% weighted
  print("Detected trash amount:",average_trash_amount)
  ```

Example result:
  ```
  { 
	'0.3' (confident score):
		{'paper': 1, 'plastic': 2, 'glass': 3, 'others': 4, 'trash_amount': 0.003},
	'0.6' (confident score):
		{'paper': 1, 'plastic': 2, 'glass': 3, 'others': 4, 'trash_amount': 0.003},
  }
  ```

```trash_amount``` is the ratio between the number of trash pixels (the white ones in the Mask image below) and the total number of pixel of the image.
Input image                             |  Result image                             | Mask
:--------------------------------------:|:-----------------------------------------:|:-----------------------------------------:
![Input image](./img/test_image.jpg)    |  ![Result image](./img/result_image.jpg)  |![Mask image](./img/mask.jpg)

## Training from scratch with GPU (Linux)
Install libraries (restarting runtime is needed)
  ```
  pip install -r requirements_gpu.txt -f https://download.pytorch.org/whl/torch_stable.html -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
  ```
  
Download the dataset
  ```
  python download.py
  ```


Train and evaluate the model. Possible arguments for training configuration ```get_train_config()``` (leave blank for default):
  * **num_workers**: Number of dataloader workers. Default: 4.
  * **batch_size**: Batch size. Default: 8.
  * **base_lr**: Base learning rate. Default: 0.0125.
  * **num_iterations**: Total number of training iterations. Default: 4500.
  * **num_classes**: Number of data classes (60 for the TACO dataset).
  
  ```python
  from train import get_train_config, train_model, evaluate_model

  cfg = get_train_config()
  train_model(cfg)
  evaluate_model(cfg)
  ```

## Evaluation metrics ([Bounding box AP](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173#:~:text=methods%20will%20diverge.-,COCO%20mAP,-Latest%20research%20papers)) of the model used in the project
| Category                  | AP     | Category               | AP     | Category                | AP     |
|:--------------------------|:-------|:-----------------------|:-------|:------------------------|:-------|
| Aluminium foil            | 65.050 | Battery                | nan    | Aluminium blister pack  | nan    |
| Carded blister pack       | nan    | Other plastic bottle   | 4.742  | Clear plastic bottle    | 48.379 |
| Glass bottle              | 18.923 | Plastic bottle cap     | 31.944 | Metal bottle cap        | 31.891 |
| Broken glass              | nan    | Food Can               | 53.936 | Aerosol                 | 33.333 |
| Drink can                 | 31.786 | Toilet tube            | nan    | Other carton            | 9.887  |
| Egg carton                | nan    | Drink carton           | 2.456  | Corrugated carton       | 45.071 |
| Meal carton               | 0.000  | Pizza box              | nan    | Paper cup               | 12.304 |
| Disposable plastic cup    | 17.022 | Foam cup               | 0.000  | Glass cup               | nan    |
| Other plastic cup         | nan    | Food waste             | 0.000  | Glass jar               | nan    |
| Plastic lid               | 13.642 | Metal lid              | nan    | Other plastic           | 4.729  |
| Magazine paper            | nan    | Tissues                | 9.780  | Wrapping paper          | 3.684  |
| Normal paper              | 2.506  | Paper bag              | 48.696 | Plastified paper bag    | nan    |
| Plastic film              | 21.328 | Six pack rings         | nan    | Garbage bag             | 0.000  |
| Other plastic wrapper     | 14.339 | Single-use carrier bag | 20.969 | Polypropylene bag       | 0.000  |
| Crisp packet              | 33.342 | Spread tub             | 0.000  | Tupperware              | nan    |
| Disposable food container | nan    | Foam food container    | 75.149 | Other plastic container | nan    |
| Plastic glooves           | nan    | Plastic utensils       | 45.446 | Pop tab                 | 1.145  |
| Rope & strings            | 6.238  | Scrap metal            | 0.000  | Shoe                    | nan    |
| Squeezable tube           | nan    | Plastic straw          | 15.519 | Paper straw             | nan    |
| Styrofoam piece           | 27.518 | Unlabeled litter       | 2.827  | Cigarette               | 7.610  |