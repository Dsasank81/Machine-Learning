# Food Image Classification

Transfer-learning baseline using MobileNetV2. Organize images by class:

```
data/
  dosa/
  idli/
  pizza/
  ...
```

Install `tensorflow`, then run `python train.py --data data --epochs 8`. The script creates a stratified-by-directory 80/20 train/validation split and saves a Keras model plus class names.

Use images you have permission to use. The model is a prototype and requires evaluation on an independent test set before any real-world use.
