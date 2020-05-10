# Project 2: Image Classifier


Training a model to classify 102 different types of flowers.

**Training a model**
```
positional arguments:
  opts                  The directory of where training/validation/test directories are in.

optional arguments:
  -h, --help            show this help message and exit
  --save SAVE_DIR, -s SAVE_DIR
                        Save models in this directory (eg: checkpoints)
  --gpu                 Pass flag if you want to use the GPU, defaults to CPU.
  --arch ARCH           Load a pre-trained model from Pytorch (default=vgg16 [vgg16 | resnet101]).
  --epochs EPOCHS       Number of epochs when training (default=2).
  --hidden_units HIDDEN_UNITS HIDDEN_UNITS, -hu HIDDEN_UNITS HIDDEN_UNITS
                        Hidden units for the hidden layers (default=[1280, 256]).
  --learning_rate LEARNING_RATE, -lr LEARNING_RATE
                        Set the learning rate (default=0.001).
  --dropout DROPOUT DROPOUT
                        Dropout probability (default=[0.4, 0.2]).
```

**Predicting**
```
positional arguments:
  opts                  Add the path to the image and the path to the model.

optional arguments:
  -h, --help            show this help message and exit
  --gpu                 Pass flag if you want to use the GPU, defaults to CPU.
  --top_k TOP_K         Returns top k class probabilities (default=1).
  --category_names CATEGORY_NAMES
                        Provide the label mapping JSON file (default='cat_to_name.json').
```
