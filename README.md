# Flowers102-Resnet50

FelloshipAi project 

might takes some time to load, due to many plots and gifs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1M_odF1YhijOgr3FnrRSEtlDMCC7QRgQi?usp=sharing)


## Flowers102-Resnet50

### Main Workflow
- [x] Proprocessing
  - [X] downloading dataset
  - [X] building dataset pipeline
    - [X] Augmentation
    - [ ] Augmentation plot 
  - [x] EDA
    - [X] bar plot for class distribution
      - [x] slider for classes
    - [x] show images for each class
      - [x] slider for classes
    - [X] write explaination for EDA in notebook

- [x] Model
- [x] Training
  - [x] better training loop information
  - [x] Epxerimentation tracking - Tensorboard
    - [x] Experiment 1 : Resnet50 - 99 Epochs 
    - [x] Experiment 2 : change lr sheduler - 50 Epochs


- [X] Inference module  
  - [X] Test loop 
  - [X] grad-cam 
  - [X] adjust inference script
  
- [X] XAI - GradCam from scratch
- [X] Visualization
  - [X] metrics
    - [X] accuracy
    - [X] loss
    - [ ] adjust plot for 3 metrics [train , test, val]
  - [X] gradcam
    - [X] animated - 30 frames
    - [X] top 4 classes + CAM
      - [ ] that class distribution vs top 5 classes
    - [X] 1 img vs all CAM
      - [ ] bar plot
      - [ ] add top 5 classes in dataset

- [X] Clean up Notebook
  - [ ] add Insights section
- [ ] Clean up code


### Advancements
- [ ] top 5 accuracy  
- [ ] Train better model
- [ ] TPU - Training
