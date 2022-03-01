from dataset import OCARPDataset
from dataloader import OCARPDataloder
from matplotlib import pyplot as plt
import random
from unet import get_model, jacard_coef
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from evaluate import evaluateModel, predictImage, plotGraph
import os

################################################################
seed=45
image_size = 224
TrainType = "OCARP_Aug#AB3"
ModelType = "ocarp_aug_x50_nonaug_bgpaste_1paste_#AB3"
original_folder = "CarrotDataset"
epochs = 385

folder = original_folder+"_resize_"+str(image_size)+"_split_seed_"+str(seed)
root_directory = 'E:/Fyp/ImportantData/Dataset/'+folder
print("Folder Path: ",root_directory)  
root_directory_train = root_directory+ '/train'
root_directory_validation = root_directory + '/val'
root_directory_test = root_directory + '/test'

X_train = root_directory_train +'/img'
y_train = root_directory_train +'/lbl'

X_val = root_directory_validation +'/img'
y_val= root_directory_validation +'/lbl'

X_test = root_directory_test +'/img'
y_test= root_directory_test +'/lbl'

green = '#00FF00'
red = '#FF0000'
black = '#000000'
objectPixelThreshold = 50

#Import Dataset and Extract objects
train_dataset = OCARPDataset(X_train, y_train)
grabCuttedImageList, transMaskList = train_dataset.ocarpExtractObjects(green, red, black, objectPixelThreshold, image_size)
validation_dataset = OCARPDataset(X_val, y_val)
test_dataset = OCARPDataset(X_test, y_test)

print("Images for Training: ",len(train_dataset))
print("Images for Validation: ",len(validation_dataset))
print("Images for Testing: ",len(test_dataset))

#Plot the extracted objects and masks

image_number_crop = random.randint(0, len(grabCuttedImageList)-1)
#image_number_crop = 157
plt.figure(figsize=(6, 3))
plt.subplot(121)
plt.title('Segmented Object')
plt.imshow(grabCuttedImageList[image_number_crop])
plt.subplot(122)
plt.title('Ground Truth')
plt.imshow(transMaskList[image_number_crop])
plt.show()

################################################################

#Create Custom Dataloader and batchwise augment data
train_dataloader = OCARPDataloder(train_dataset, True, black, batch_size=3, shuffle=False, grabCuttedImageList = grabCuttedImageList, transMaskList = transMaskList, isPasteAugment = False, isOnlyPasteOnBg = True, objectPasteCount = 1)
#train_dataloader = OCARPDataloder(train_dataset, False , black, batch_size=3, shuffle=False)
valid_dataloader = OCARPDataloder(validation_dataset, False, black, batch_size=3, shuffle=False)
test_dataloader = OCARPDataloder(test_dataset, False, black, batch_size=1, shuffle=False)

print("Iterations per Epoch for Training: ",len(train_dataloader))
print("Iterations per Epoch for Validation: ",len(valid_dataloader))
print("Iterations for Testing: ",len(test_dataloader))

#Plot random images and masks from train, validation and test 
batch = random.randint(0, len(train_dataloader)-1)
x, y = train_dataloader[batch]
image = random.randint(0, len(x)-1)
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

batch = random.randint(0, len(test_dataloader)-1)
x, y = test_dataloader[batch]
image = random.randint(0, len(x)-1)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title('Test Image')
plt.imshow(x[image])
plt.subplot(122)
plt.title('Ground Truth')
plt.imshow(y[image])
plt.show()
################################################################

#import model
n_classes = 3
IMG_CHANNELS = 3
model = get_model(n_classes, image_size, image_size, IMG_CHANNELS)
model.summary()

################################################################
#Train Model 

model_dir = 'E:/Fyp/ImportantData/Implementation/ImplementationRGBUNET/PCPythonFiles/Library/ocarp-aug/Evaluate/Models/'+str(image_size)+'Size/UNET_Models/Carrot/'+TrainType+'/Cross_Val/Seed_'+str(seed)+'/'
model_name = '_carrot_'+ModelType+'_unet_seed_'+str(seed)+'.hdf5'
model_path_arr = []
for i in range(1,6):
  # model_path_arr.append(model_dir + 'model'+str(i) +model_name)
  model_path_arr.append(model_dir + 'model' +model_name)
print(model_path_arr[0])

## Train one model
model_no = 1
model_no = model_no-1
metrics=['accuracy', jacard_coef]
opt = Adadelta(learning_rate=1, rho=0.95, epsilon=1e-07, name="Adadelta")
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=metrics)
print("\nModel :"+model_path_arr[model_no]+'\n')
model_path = model_path_arr[model_no]
mc = ModelCheckpoint(model_path, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
history = model.fit(
    train_dataloader, 
    steps_per_epoch=len(train_dataloader), 
    epochs=epochs, 
    callbacks =[mc],
    validation_data=valid_dataloader, 
    validation_steps=len(valid_dataloader),
    verbose = 1,
)
################################################################

#Plot training graph and save
path = 'E:/Fyp/ImportantData/Implementation/ImplementationRGBUNET/PCPythonFiles/Library/ocarp-aug/Evaluate/Models/'+str(image_size)+'Size/UNET_Models/Carrot/'+TrainType+'/Cross_Val/Seed_'+str(seed)+'/Graphs/'
plotGraph(history, model_name, path, seed, ModelType)

################################################################

#Predict and print results
test_image_dataset = []
test_labels_cat = []
for i in range(0, len(test_dataloader)):
    X, y = test_dataloader[i]
    test_image_dataset.append(X[0])
    test_labels_cat.append(y[0])
test_image_dataset = np.array(test_image_dataset)
test_labels_cat = np.array(test_labels_cat)
print(len(test_image_dataset))

model_path = model_path_arr[model_no]
print(model_path)

path = 'E:/Fyp/ImportantData/Implementation/ImplementationRGBUNET/PCPythonFiles/Library/ocarp-aug/Evaluate/Models/'+str(image_size)+'Size/UNET_Models/Carrot/'+TrainType+'/Cross_Val/Seed_'+str(seed)+'/'
y_test_argmax, model = evaluateModel(model_path,jacard_coef, test_image_dataset, test_labels_cat, history, model_name, path)

################################################################

#Predict Random Image and Display
test_img_number = 0
test_img, ground_truth, predicted_img = predictImage(test_img_number, model, test_image_dataset, y_test_argmax)

plt.figure(figsize=(20, 14))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img)
plt.subplot(232)
plt.title('Ground Truth')
plt.imshow(ground_truth)
plt.subplot(233)
plt.title('Prediction on Test Image')
plt.imshow(predicted_img)
plt.show()

################################################################

#Predict and Save all Images
path = 'E:/Fyp/ImportantData/Implementation/ImplementationRGBUNET/PCPythonFiles/Library/ocarp-aug/Evaluate/Models/'+str(image_size)+'Size/UNET_Models/Carrot/'+TrainType+'/Cross_Val/Seed_'+str(seed)+"/Predictions/"
print(path)
os.mkdir(path)
for test_img_number in range(0, len(test_image_dataset)):
    test_img, ground_truth, predicted_img = predictImage(test_img_number, model, test_image_dataset, y_test_argmax)
    plt.imsave(path+str(test_img_number+1)+"_"+ModelType+"_test_prediction_seed_"+str(seed)+".png", predicted_img)

################################################################
    


