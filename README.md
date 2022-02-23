# Object Cut and Random Paste Augmentation (OCARP Augmentation)
<b>Object Cut and Random Paste Augmentation Strategy for Semantic Segmentation of Crop and Weed</b>

This is an augmentation library that extracts crop or weed objects from a crop and weed semantic segmentation dataset and batchwise augment images by pasting the extracted objects randomly on original images. 

<b>Main Features</b> :
<ul>
  <li>Extracts a selected object and the corresponding mask from an image</li>
  <li>Retrieve batch wise augmented images by pasting the extracted images randomly on original images</li>
</ul>
To better understand the augmentation strategy, check <a href = "Documents/OCARP Augmentation Strategy.pdf">this</a>.

<h2>Prerequisites</h2>

<h3>Dataset Structure</h3>
<ul>
  <li>The dataset should be structured according to the following manner for the library to work properly.</li>
</ul>

            Dataset
              ├── test
              │   ├── img
              │   │   └── pic3.jpg
              │   └── lbl
              │       └── pic3.jpg
              ├── train
              │   ├── img
              │   │   └── pic1.jpg
              │   └── lbl
              │       └── pic1.jpg
              └── val
                  ├── img
                  │   └── pic2.jpg
                  └── lbl
                      └── pic2.jpg                 
<h3>Image Size</h3>
<ul>
  <li>The height and width of the images and masks should be the same. Eg. 256x256</li>
</ul>

**Note : The library provides additional functionality to restructure and resize the dataset. It is available <a href = "restructure.py">here</a>.**

<h3>Requirements</h3>
<ol>
  <li>NumPy</li>
  <li>pillow</li>
  <li>tensorflow</li>
  <li>opencv</li>
</ol>

<h2>Usage</h2>

Importing Dataset :
```ruby
from dataset import OCARPDataset

X_train = "path-to-train-images"
Y_train = "path-to-train-masks"
X_val = "path-to-val-images"
Y_val = "path-to-val-masks"

train_dataset = OCARPDataset(X_train, Y_train)
validation_dataset = OCARPDataset(X_val, Y_val)
```

Extract Objects (Crop Plants):
```ruby
greenHexCode = '#00FF00'
redHexCode = '#FF0000'
blackHexCode = '#000000'
objectPixelMinNumber = 200
image_size = 256

grabCuttedImageList, transMaskList = train_dataset.ocarpExtractObjects(greenHexCode, redHexCode, blackHexCode, objectPixelMinNumber, image_size)
```

Create Custom Data Generator:
```ruby
train_dataloader = OCARPDataloder(train_dataset, True, blackHexCode, batch_size=3, shuffle=False, grabCuttedImageList = grabCuttedImageList, transMaskList = transMaskList, isPasteAugment = False, isOnlyPasteOnBg = True, objectPasteCount = 1)
valid_dataloader = OCARPDataloder(validation_dataset, False, blackHexCode, batch_size=3, shuffle=False)
```
<h2>Examples</h2>

A full example scenario of the augmentation strategy including training and evaluation is available <a href = "train.py">here</a>.

<h2>Documentation</h2>

Latest documentation is avaliable on Read the Docs.

