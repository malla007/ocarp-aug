import cv2
import random
import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical, Sequence

#Break the image into 16 different parts
def breakImageParts(image, crop_box_size):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGBA)
    image = Image.fromarray(image)
    crop_boxes_array = []
    for i in range(0,4):
        for j in range(0,4):
            x = crop_box_size*i
            y = crop_box_size*j
            area = (x, y, x+crop_box_size, y+crop_box_size)
            cropped_img = image.crop(area)
            crop_boxes_array.append(cropped_img)
    return crop_boxes_array  

#Paste the obtained object transparent image and mask on top of original images
def pasteCropBoxOnImageAndMask(only_background_crop_box_index_array, original_mask, original_image, grabCuttedImageList, transMaskList, crop_box_size, isPasteAugment, isOnlyPasteOnBg):
    image = original_image
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGBA)
    image = Image.fromarray(image)
    
    mask = original_mask
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGBA)
    mask = Image.fromarray(mask)
    if isOnlyPasteOnBg == False:
      only_background_crop_box_index_array = [i for i in range(0, 16)]
    try:
      selected_pos = random.choice(only_background_crop_box_index_array)
    except:
      selected_pos = None
    if selected_pos != None:
      for position in only_background_crop_box_index_array:
          count = 0
          #Traverse through the 4x4 blocks
          for e in range(0,4):
              for j in range(0,4):
                  x = crop_box_size*e
                  y = crop_box_size*j
                  paste_position = (x , y)
                  if count == position:
                      if selected_pos == position:
                          random_object_number = random.randint(0,len(grabCuttedImageList)-1)
                          grabCuttedImage = grabCuttedImageList[random_object_number]
                          grabCuttedImage = cv2.cvtColor(np.array(grabCuttedImage), cv2.COLOR_RGB2RGBA)
                          
                          transMaskImage = transMaskList[random_object_number]
                          transMaskImage = cv2.cvtColor(np.array(transMaskImage), cv2.COLOR_RGB2RGBA)
                          
                          isFlip = False
                          isRotate = False
                          if isPasteAugment:
                            #Rotate -90, 180, 90 degrees the object randomly
                            isRotate = random.choice([True, False])
                            if isRotate:
                                rotation = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
                                rotate = random.choice(rotation)
                                grabCuttedImage = cv2.rotate(grabCuttedImage, rotate)
                                transMaskImage = cv2.rotate(transMaskImage, rotate)
                                isFlip= random.choice([True, False])
                                if isFlip:
                                    #Horizontal Flip the object randomly  
                                    grabCuttedImage = cv2.flip(grabCuttedImage, 0)
                                    transMaskImage = cv2.flip(transMaskImage, 0)
                            else:
                                isFlip = True
                                #Horizontal Flip the object randomly    
                                grabCuttedImage = cv2.flip(grabCuttedImage, 0)
                                transMaskImage = cv2.flip(transMaskImage, 0)

                          grabCuttedImage = Image.fromarray(grabCuttedImage)
                          transMaskImage = Image.fromarray(transMaskImage)
                          #Paste object mask and image on original image
                          image.paste(grabCuttedImage, paste_position, grabCuttedImage)
                          mask.paste(transMaskImage, paste_position, transMaskImage)
                  count = count+1
    return image, mask

#Paste extracted object on an original image and mask
def pasteOnOriginalImage(image, mask, grabCuttedImageList, transMaskList, backgroundHex, isPasteAugment ,isOnlyPasteOnBg):

    original_image = image 
    original_mask = mask
    original_image = np.array(original_image)
    original_mask = np.array(original_mask)
    
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_mask = cv2.cvtColor(original_mask, cv2.COLOR_BGR2RGB)

    height, width, channels = image.shape
    crop_box_size = int(height/4) 

    crop_boxes_mask_array = breakImageParts(original_mask,crop_box_size)
    #From the object box masks optained, find the ones with only background pixels in them
    only_background_crop_box_index_array = findIndexAllBoxesWithOnlyBackground(crop_boxes_mask_array, backgroundHex)
    #Paste the obtained object transparent image and mask on top of original images
    image, mask = pasteCropBoxOnImageAndMask(only_background_crop_box_index_array, original_mask, original_image, grabCuttedImageList, transMaskList, crop_box_size, isPasteAugment, isOnlyPasteOnBg)

    image = np.asarray(image)
    mask = np.asarray(mask)
    
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGBA2RGB)
    return image, mask

#From the object box masks obtained, find the ones with only background pixels in them
def findIndexAllBoxesWithOnlyBackground(crop_boxes_array, backgroundHex):
    backgroundHex = backgroundHex.lstrip('#')
    backgroundRGB = np.array(tuple(int(backgroundHex[i:i+2], 16) for i in (0, 2, 4)))
    backgroundRGB = backgroundRGB[0], backgroundRGB[1], backgroundRGB[2], 255
    only_background_crop_box_index_array = []

    for box in crop_boxes_array:
        soil_pixel_count = 0
        h,w = box.size
        for pixel in box.getdata():
            if pixel == (backgroundRGB):
                soil_pixel_count += 1
        only_background_crop_box_index_array.append(soil_pixel_count)
    only_background_crop_box_index_array = [i for i,v in enumerate(only_background_crop_box_index_array) if v == h*w]
    return only_background_crop_box_index_array

def rgb_to_2D_label(label, hexArray):

  Weed = '#FF0000'.lstrip('#')
  Weed = np.array(tuple(int(Weed[i:i+2], 16) for i in (0, 2, 4))) 

  Crop = '#00FF00'.lstrip('#')
  Crop = np.array(tuple(int(Crop[i:i+2], 16) for i in (0, 2, 4))) 

  Unlabeled = hexArray[2].lstrip('#') 
  Unlabeled = np.array(tuple(int(Unlabeled[i:i+2], 16) for i in (0, 2, 4))) 

  label_seg = np.zeros(label.shape,dtype=np.uint8)
  label_seg [np.all(label == Weed,axis=-1)] = 0
  label_seg [np.all(label==Crop,axis=-1)] = 1
  label_seg [np.all(label==Unlabeled,axis=-1)] = 2
  
  label_seg = label_seg[:,:,0]  
  
  return label_seg

class OCARPDataloder(Sequence):    
    def __init__(self, dataset, isOcarp , hexArray ,batch_size=1, shuffle=False, grabCuttedImageList = None, transMaskList = None, isPasteAugment = True, isOnlyPasteOnBg = True,  objectPasteCount = 1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        self.on_epoch_end()
        self.grabCuttedImageList = grabCuttedImageList
        self.transMaskList = transMaskList
        self.isOcarp = isOcarp
        self.hexArray = hexArray
        self.isPasteAugment = isPasteAugment
        self.isOnlyPasteOnBg = isOnlyPasteOnBg
        self.objectPasteCount = objectPasteCount

    def __getitem__(self, i):
        # Collect data in batches
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        X = []
        y = []
        for j in range(start, stop):
            image = self.dataset[j][0]
            mask = self.dataset[j][1]

            if self.isOcarp == True:
              for i in range(0, self.objectPasteCount):
                #Paste extracted object on an original image and mask
                image, mask = pasteOnOriginalImage(image, mask, self.grabCuttedImageList, self.transMaskList, self.hexArray[2], self.isPasteAugment, self.isOnlyPasteOnBg)

            #Convert rgb to 2d label
            label = rgb_to_2D_label(mask, self.hexArray)

            X.append(image)
            y.append(label)

        X = np.array(X)
        y = np.array(y)
        y = np.expand_dims(y, axis=3)
        n_classes = 3
        #One Hot encoding
        y = to_categorical(y, num_classes=n_classes)
        return X, y
    
    def __len__(self):
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)  