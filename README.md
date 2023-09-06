# Flowers102-Resnet50

FellowshipAi project 

might takes some time to load, due to many plots and gifs, please be patient

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1M_odF1YhijOgr3FnrRSEtlDMCC7QRgQi?usp=sharing)

The Notebook has everything from downloading dataset to fine tuning the model and inference.


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
    - [x] adjust plot for 3 metrics [train , test, val]
  - [X] gradcam
    - [X] animated - 30 frames
    - [X] top 4 classes + CAM
      - [ ] that class distribution vs top 5 classes
    - [X] 1 img vs all CAM
      - [ ] bar plot
      - [ ] add top 5 classes in dataset
  
- [X] Tuner - hyperparameter tuning
  - [X] RADAM optimizer
  - [X] Cyclical Learning Rates
  - [X] easy interface for tuning

- [X] Clean up Notebook
- [x] add Insights section
- [ ] Clean up code


### Advancements
- [X] Train better model
- [ ] top 5 accuracy  
- [ ] TPU - Training
