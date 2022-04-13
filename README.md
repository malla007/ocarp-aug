# Object Cut and Random Paste Augmentation (OCARP Augmentation)
<b>Object Cut and Random Paste Augmentation Strategy for Semantic Segmentation of Crop and Weed</b>

This is an augmentation library that extracts crop or weed objects from a crop and weed semantic segmentation dataset and batchwise augment images by pasting the extracted objects randomly on original images. 

<b>Main Features</b> :
<ul>
  <li>Extracts a selected object and the corresponding mask from an image</li>
  <li>Retrieve batch wise augmented images by pasting the extracted images randomly on original images</li>
</ul>
To better understand the augmentation strategy, check <a href = "Documents/OCARP Augmentation Strategy.pdf">this</a>.

![ezgif-5-546b16de96](https://user-images.githubusercontent.com/74457911/156368127-44cac789-aea7-4cca-94af-5c8f9c63e3af.gif)

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
  <li>The height and width of the images and masks should be the same. Eg. 256 x 256</li>
</ul>

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
from ocarpaug import OCARPDataset

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
objectPixelMinimumCount = 50
hexArray = [green, red, black]

grabCuttedImageList, transMaskList = train_dataset.ocarpExtractObjects(hexArray, objectPixelMinimumCount)
```

Create Custom Data Generator:
```ruby
from ocarpaug import OCARPDataloder

train_dataloader = OCARPDataloder(train_dataset, True, hexArray, batch_size=3, shuffle=False, grabCuttedImageList = grabCuttedImageList, transMaskList = transMaskList, isPasteAugment = True, isOnlyPasteOnBg = True, objectPasteCount = 1)
valid_dataloader = OCARPDataloder(validation_dataset, False, hexArray, batch_size=3, shuffle=False)
```
<h2>Simple Augmentation Applied Training Pipeline</h2>

```ruby
from ocarpaug import OCARPDataset, OCARPDataloder

#import dataset
X_train = "path-to-train-images"
Y_train = "path-to-train-masks"
X_val = "path-to-val-images"
Y_val = "path-to-val-masks"

train_dataset = OCARPDataset(X_train, Y_train)
validation_dataset = OCARPDataset(X_val, Y_val)

#extract crop plant objects
greenHexCode = '#00FF00'
redHexCode = '#FF0000'
blackHexCode = '#000000'
objectPixelMinimumCount = 50
hexArray = [green, red, black]

grabCuttedImageList, transMaskList = train_dataset.ocarpExtractObjects(hexArray, objectPixelMinimumCount)

#create custom data generator
train_dataloader = OCARPDataloder(train_dataset, True, hexArray, batch_size=3, shuffle=False, grabCuttedImageList = grabCuttedImageList, transMaskList = transMaskList, isPasteAugment = True, isOnlyPasteOnBg = True, objectPasteCount = 1)
valid_dataloader = OCARPDataloder(validation_dataset, False, hexArray, batch_size=3, shuffle=False)

#fit model and train
history = model.fit(
    train_dataloader, 
    steps_per_epoch=len(train_dataloader), 
    epochs=10, 
    validation_data=valid_dataloader, 
    validation_steps=len(valid_dataloader),
    verbose = 1,
)
```

<h2>Examples</h2>

A full example scenario of the augmentation strategy including training and evaluation is available <a href = "train.py">here</a>.

<h2>Documentation</h2>

Latest documentation is avaliable on <a href = "https://github.com/malla007/ocarp-aug/wiki/Documentation">Read the Docs</a>.

