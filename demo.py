from ocarpaug import OCARPDataset, OCARPDataloder
from utils import get_model

import random
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.callbacks import ModelCheckpoint

#Dataset Paths
X_train = 'E:/Fyp/ImportantData/Dataset/CarrotDataset_resize_224_split_seed_41/train/img'
y_train = 'E:/Fyp/ImportantData/Dataset/CarrotDataset_resize_224_split_seed_41/train/lbl'

X_val = 'E:/Fyp/ImportantData/Dataset/CarrotDataset_resize_224_split_seed_41/val/img'
y_val= 'E:/Fyp/ImportantData/Dataset/CarrotDataset_resize_224_split_seed_41/val/lbl'

#Extract Object Input Parameters
green = '#00FF00'
red = '#FF0000'
black = '#000000'
objectPixelMinimumCount = 100
hexArray = [green, red, black]

#Import Dataset 
train_dataset = OCARPDataset(X_train, y_train)
validation_dataset = OCARPDataset(X_val, y_val)

print("Images for Training: ",len(train_dataset))
print("Images for Validation: ",len(validation_dataset))

#Extract objects
grabCuttedImageList, transMaskList = train_dataset.ocarpExtractObjects(hexArray, objectPixelMinimumCount)

image_number_crop = random.randint(0, len(grabCuttedImageList)-1)
plt.figure(figsize=(6, 3))
plt.subplot(121)
plt.title('Segmented Object')
plt.imshow(grabCuttedImageList[image_number_crop])
plt.subplot(122)
plt.title('Ground Truth')
plt.imshow(transMaskList[image_number_crop])
plt.show()

#Create Custom Dataloader and batchwise augment data
train_dataloader = OCARPDataloder(train_dataset, True, hexArray, batch_size=3, shuffle=False, grabCuttedImageList = grabCuttedImageList, 
                                  transMaskList = transMaskList, isPasteAugment = True, isOnlyPasteOnBg = False, objectPasteCount = 1)
valid_dataloader = OCARPDataloder(validation_dataset, False, hexArray, batch_size=3, shuffle=False)

print("Iterations per Epoch for Training: ",len(train_dataloader))
print("Iterations per Epoch for Validation: ",len(valid_dataloader))

batch = 1
x, y = train_dataloader[batch]
image = 1
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title('Training Image')
plt.imshow(x[image])
plt.subplot(122)
plt.title('Ground Truth')
plt.imshow(y[image])
plt.show()

batch = random.randint(0, len(valid_dataloader)-1)
x, y = valid_dataloader[batch]
image = random.randint(0, len(x)-1)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title('Validation Image')
plt.imshow(x[image])
plt.subplot(122)
plt.title('Ground Truth')
plt.imshow(y[image])
plt.show()

#Train Model 
n_classes = 3
IMG_CHANNELS = 3
model = get_model(n_classes, 224, 224, IMG_CHANNELS)
model.summary()

model_no = 1
model_no = model_no-1
metrics=['accuracy']
opt = Adadelta(learning_rate=1, rho=0.95, epsilon=1e-07, name="Adadelta")
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=metrics)
mc = ModelCheckpoint("model_path", monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

history = model.fit(
    train_dataloader, 
    steps_per_epoch=len(train_dataloader), 
    epochs=10, 
    callbacks =[mc],
    validation_data=valid_dataloader, 
    validation_steps=len(valid_dataloader),
    verbose = 1,
)




