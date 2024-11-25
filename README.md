# SBIR Sketch Enhancement GANs

This repository is in fulfillment of COE 494 Computer Vision course. The repository contain the classifier, CycleGAN and consitional CycleGAN models used to transfer bad sketches from the QuickDraw-Extended! dataset to good sketches from Sketchy dataset. The purpose of the project is to eventually create a framework that would aid in Sketch-Based Image Retrieval (i.e., given a sketch as a query, retrieve a ranked list of queried images).

QuickDraw-Extended dataset contains real sketches of people highlighting the abstract nature of sketches. However, the sketches in Sketchy dataset were drawn from artists, which doesn't reflect the majority of real sketches. Therefore, we chose to follow an approach of improving the QuickDraw-Extended sketches using Sketchy dataset. Both dataset have a lot of classesfor training (more than 100), so we select only 12 common classes from both for sketch enhancement. The approach we followed is to train a cycleGAN for each class. But since this is not scalable, we also tried trainaing a conditional cycleGAN (similar to StarGAN) for all classes at once. However, for them to work in a real application, the class of the sketch has to be know prior to inference, which is why we train a classifier on the 12 classes. 


### Classifier

The classifier code is contained in `classifier.ipynb`. The model uses AlexNet backbone with few classification layers.In order to train the model, the notebook has a function `train_loop(model, criterion, optim, epochs, trainloader, testloader, device, num_classes)` which takes care of training the model and returns the loss and accuracy curves for both training and validation loaders.

To test the trained model, the notebook also includes `test(model, testloader, device)` which prints the classification report consisting of precision, recall, f1 score, and accuracy, along with a confusion matric display. (Both training and testing are already performed in the notebook).

### CycleGAN

The training code for CycleGAN is included in `cycleGAN.ipynb`. The notebook has a trainer class called `SketchEnhancer`. The class handles creating the generators and discrimnators and train them accordingly for a single sketch class. The notebook also has the function `train_all_classes(base_dir, classes, batch_size, num_epochs)` that invokes the trainer for each class and trains 12 cycleGAN models.

An external testing script is provided for enhanced sketch generation in `cycleGAN_test.py`. The script requires that a trained model is saved somewhere on the machine, which the trainer class already does. The script is invoked as follows:
```console
$ python3 cycleGAN_test.py

usage: cycleGAN_test.py [-h] --bad-sketch-dir BAD_SKETCH_DIR --good-sketch-dir GOOD_SKETCH_DIR --class-type CLASS_TYPE --output-dir OUTPUT_DIR [--num-images NUM_IMAGES] --checkpoint CHECKPOINT

optional arguments:
  -h, --help            show this help message and exit
  --bad-sketch-dir BAD_SKETCH_DIR
  --good-sketch-dir GOOD_SKETCH_DIR
  --class-type CLASS_TYPE
  --output-dir OUTPUT_DIR
  --num-images NUM_IMAGES
  --checkpoint CHECKPOINT
```

### Conditional CycleGAN

The training code for conditonal CycleGAN is included in `conditional_cycleGAN.ipynb`. The notebook has a trainer class called `Trainer`. The class handles creating the generators and discrimnators and train them accordingly for all sketches at once. 

For testing, an external script is provided `conditional_cycleGAN_test.py` that generates images using the trained generator. A model has to already be trained and stored in the machine, which the trainer class also handles. The usage is as follows:

```console
$ python3 conditional_cycleGAN_test.py

usage: conditional_cycleGAN_test.py [-h] --bad-sketch-dir BAD_SKETCH_DIR --good-sketch-dir GOOD_SKETCH_DIR --output-dir OUTPUT_DIR [--num-samples NUM_SAMPLES] --checkpoint-path CHECKPOINT_PATH

optional arguments:
  -h, --help            show this help message and exit
  --bad-sketch-dir BAD_SKETCH_DIR
  --good-sketch-dir GOOD_SKETCH_DIR
  --output-dir OUTPUT_DIR
  --num-samples NUM_SAMPLES
  --checkpoint-path CHECKPOINT_PATH
```

