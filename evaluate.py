from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.models import load_model
import numpy as np
from matplotlib import pyplot as plt
import os


def getPercentage(value):
    value = '{:.2f}'.format(value*100)
    value = str(value)+str("%")
    return value

def calculateMIOU(values, n_classes):

    '''
    IOU = true_positive / (true_positive + false_positive + false_negative)
    '''
    
    class0_IoU = values[0,0]/(values[0,0] + values[1,0] + values[2,0] + values[0,1] + values[0,2])
    class1_IoU = values[1,1]/(values[1,1] + values[0,1] + values[2,1] + values[1,0] + values[1,2])
    class2_IoU = values[2,2]/(values[2,2] + values[0,2] + values[1,2] + values[2,0] + values[2,1])
    
    return class0_IoU, class1_IoU, class2_IoU
    
def calculateDSC(values, n_classes):

    '''
    F1 Score =  (2*true_positive) / ((2*true_positive) + false_positive + false_negative)
    '''
    
    class0_F1 = (values[0,0]*2)/((values[0,0]*2) + values[1,0] + values[2,0] + values[0,1] + values[0,2])
    class1_F1 = (values[1,1]*2)/((values[1,1]*2) + values[0,1] + values[2,1] + values[1,0] + values[1,2])
    class2_F1 = (values[2,2]*2)/((values[2,2]*2) + values[0,2] + values[1,2] + values[2,0] + values[2,1])
    averageF1 = (class0_F1+class1_F1+class2_F1)/n_classes
    
    return class0_F1, class1_F1, class2_F1, averageF1


def evaluateModel(model_path,jacard_coef, test_image_dataset, test_labels_cat, history, model_name, path):
    model = load_model(model_path,custom_objects={'jacard_coef':jacard_coef}, compile = False)
    
    #Predict the dataset
    y_pred=model.predict(test_image_dataset)
    y_pred_argmax=np.argmax(y_pred, axis=3)
    y_test_argmax=np.argmax(test_labels_cat, axis=3)
 
    n_classes = 3
    #Calculate meanIoU using Keras api
    IOU_keras = MeanIoU(num_classes=n_classes)  
    IOU_keras.update_state(y_test_argmax, y_pred_argmax)
    
    '''
    -Confusion Matrix -
    [[  0,0    0,1    0,2  ]
     [  1,0    1,1    1,2  ]
     [  2,0    2,1    2,2  ]]
    
    Refer : https://i.stack.imgur.com/AuTKP.png
    '''
    #Confusion Matrix
    values = np.array(IOU_keras.get_weights()).reshape(n_classes,n_classes)
    #Calculate MIoU per class
    class0_IoU, class1_IoU, class2_IoU = calculateMIOU(values, n_classes)
    #Calculate DSC/F1 score per class
    class0_F1, class1_F1, class2_F1, averageF1 = calculateDSC(values, n_classes)
    
    print("---Test Set Accuracies---"+'\n')
    print("Mean IoU : ", getPercentage(IOU_keras.result().numpy()))
    print("Weed IoU : "+str(getPercentage(class0_IoU)))
    print("Crop IoU : "+str(getPercentage(class1_IoU)))
    print("Soil IoU : "+str(getPercentage(class2_IoU))+"\n")

    print("Average F1 : "+str(getPercentage(averageF1)))
    print("Weed F1 : "+str(getPercentage(class0_F1)))
    print("Crop F1 : "+str(getPercentage(class1_F1)))
    print("Soil F1 : "+str(getPercentage(class2_F1))+"\n")

    f = open(path+'model' +model_name.replace(".hdf5","")+".txt", "w")
    highest = max(history.history['val_accuracy'])

    f.write("val_accuracy did not improve from : "+str(round(float(highest), 5))+'\n'+'\n'+"---Test Set Accuracies---"+'\n'+'\n'+"Mean IoU : "+str(getPercentage(IOU_keras.result().numpy()))+'\n'+
            "Weed IoU : "+str(getPercentage(class0_IoU))+"\n"+
            "Crop IoU : "+str(getPercentage(class1_IoU))+"\n"+
            "Soil IoU : "+str(getPercentage(class2_IoU))+"\n"+"\n"+
            "Average F1 : "+str(getPercentage(averageF1))+"\n"+
            "Weed F1 : "+str(getPercentage(class0_F1))+"\n"+
            "Crop F1 : "+str(getPercentage(class1_F1))+"\n"+
            "Soil F1 : "+str(getPercentage(class2_F1))+"\n")
    f.close()

    return y_test_argmax, model

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


def predictImage(test_img_number, model, test_image_dataset, y_test_argmax):
    
    test_img = test_image_dataset[test_img_number]
    ground_truth=y_test_argmax[test_img_number]
    #test_img_norm=test_img[:,:,0][:,:,None]
    test_img_input=np.expand_dims(test_img, 0)
    prediction = (model.predict(test_img_input))
    predicted_img=np.argmax(prediction, axis=3)[0,:,:]
    
    #Convert to RGB
    ground_truth = label_to_rgb(ground_truth)
    predicted_img = label_to_rgb(predicted_img)
    return test_img, ground_truth, predicted_img

def plotGraph(history, modelName, path, seed, ModelType):
    os.mkdir(path)
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training & Validation loss : model'+modelName)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path+"seed_"+str(seed)+"_"+ModelType+"_acc.png")
    plt.show()
    
    

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    plt.plot(epochs, acc, 'y', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training & Validation Accuracy : model'+modelName)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(path+"seed_"+str(seed)+"_"+ModelType+"_loss.png")
    plt.show()
    
    
    
    