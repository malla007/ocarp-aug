from dataset import OCARPDataset
from dataloader import OCARPDataloder
from matplotlib import pyplot as plt
import random
from unet import get_model, jacard_coef
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.models import load_model

################################################################
seed=45
image_size = 256
original_folder = "CarrotDataset"
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
objectPixelThreshold = 200

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
model = get_model(n_classes=3, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=3)
model.summary()

################################################################
#Train Model 

model_dir = 'C:/Users/Malinda/Desktop/Model/'+str(image_size)+'Size/UNET_Models/Carrot/OCARP_Aug/Cross_Val/'
model_name = '_carrot_ocarp_aug_unet_seed_'+str(seed)+'.hdf5'
model_path_arr = []
for i in range(1,6):
  # model_path_arr.append(model_dir + 'model'+str(i) +model_name)
  model_path_arr.append(model_dir + 'model' +model_name)
print(model_path_arr)

## Train one model
model_no = 1
model_no = model_no-1
metrics=['accuracy', jacard_coef]
opt = Adadelta(learning_rate=1, rho=0.95, epsilon=1e-07, name="Adadelta")
model = get_model()
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=metrics)
print("\nModel :"+model_path_arr[model_no]+'\n')
model_path = model_path_arr[model_no]
mc = ModelCheckpoint(model_path, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
history = model.fit(
    train_dataloader, 
    steps_per_epoch=len(train_dataloader), 
    epochs=358, 
    callbacks =[mc],
    validation_data=valid_dataloader, 
    validation_steps=len(valid_dataloader),
    verbose = 1,
)
################################################################

#Plot training graph
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

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
def getPercentage(value):
    value = '{:.2f}'.format(value*100)
    value = str(value)+str("%")
    return value

model = load_model(model_path,custom_objects={'jacard_coef':jacard_coef}, compile = False)

#IOU
y_pred=model.predict(test_image_dataset)
y_pred_argmax=np.argmax(y_pred, axis=3)
y_test_argmax=np.argmax(test_labels_cat, axis=3)


#Using built in keras function for IoU
n_classes = 3
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_test_argmax, y_pred_argmax)
print("---Test Set Accuracies---"+'\n')
print("Mean IoU : ", getPercentage(IOU_keras.result().numpy()))

'''
IOU = true_positive / (true_positive + false_positive + false_negative)
F1 Score =  (2*true_positive) / ((2*true_positive) + false_positive + false_negative)

-Weight Matrix -
[[  0,0    0,1    0,2  ]
 [  1,0    1,1    1,2  ]
 [  2,0    2,1    2,2  ]]
 '''

values = np.array(IOU_keras.get_weights()).reshape(n_classes,n_classes)
class0_IoU = values[0,0]/(values[0,0] + values[1,0] + values[2,0] + values[0,1] + values[0,2])
class1_IoU = values[1,1]/(values[1,1] + values[0,1] + values[2,1] + values[1,0] + values[1,2])
class2_IoU = values[2,2]/(values[2,2] + values[0,2] + values[1,2] + values[2,0] + values[2,1])

print("Weed IoU : "+str(getPercentage(class0_IoU)))
print("Crop IoU : "+str(getPercentage(class1_IoU)))
print("Soil IoU : "+str(getPercentage(class2_IoU))+"\n")

class0_F1 = (values[0,0]*2)/((values[0,0]*2) + values[1,0] + values[2,0] + values[0,1] + values[0,2])
class1_F1 = (values[1,1]*2)/((values[1,1]*2) + values[0,1] + values[2,1] + values[1,0] + values[1,2])
class2_F1 = (values[2,2]*2)/((values[2,2]*2) + values[0,2] + values[1,2] + values[2,0] + values[2,1])
averageF1 = (class0_F1+class1_F1+class2_F1)/3

print("Average F1 : "+str(getPercentage(averageF1)))
print("Weed F1 : "+str(getPercentage(class0_F1)))
print("Crop F1 : "+str(getPercentage(class1_F1)))
print("Soil F1 : "+str(getPercentage(class2_F1)))

################################################################

#Predict Random Image and Display
def label_to_rgb(predicted_image):
    
    Weed = '#FF0000'.lstrip('#')
    Weed = np.array(tuple(int(Weed[i:i+2], 16) for i in (0, 2, 4))) # 255, 0, 0

    Crop = '#00FF00'.lstrip('#')
    Crop = np.array(tuple(int(Crop[i:i+2], 16) for i in (0, 2, 4))) #0, 255, 0

    Unlabeled = '#000000'.lstrip('#') 
    Unlabeled = np.array(tuple(int(Unlabeled[i:i+2], 16) for i in (0, 2, 4))) #0, 0, 0
        
    
    segmented_img = np.empty((predicted_image.shape[0], predicted_image.shape[1], 3))
    
    segmented_img[(predicted_image == 0)] = Weed
    segmented_img[(predicted_image == 1)] = Crop
    segmented_img[(predicted_image == 2)] = Unlabeled
    
    segmented_img = segmented_img.astype(np.uint8)
    return(segmented_img)

test_img_number = 3
test_img = test_image_dataset[test_img_number]
ground_truth=y_test_argmax[test_img_number]
#test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img, 0)
prediction = (model.predict(test_img_input))
predicted_img=np.argmax(prediction, axis=3)[0,:,:]

#Convert to RGB
ground_truth = label_to_rgb(ground_truth)
predicted_img = label_to_rgb(predicted_img)

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