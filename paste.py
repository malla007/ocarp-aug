import cv2
import numpy as np
from PIL import Image
import random
from extract import breakImageParts

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
                            #Rotate the object randomly
                            isRotate = random.choice([True, False])
                            if isRotate:
                                rotation = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
                                rotate = random.choice(rotation)
                                grabCuttedImage = cv2.rotate(grabCuttedImage, rotate)
                                transMaskImage = cv2.rotate(transMaskImage, rotate)
                                isFlip= random.choice([True, False])
                                if isFlip:
                                    #Flip the object randomly  
                                    grabCuttedImage = cv2.flip(grabCuttedImage, 0)
                                    transMaskImage = cv2.flip(transMaskImage, 0)
                            else:
                                isFlip = True
                                #Flip the object randomly    
                                grabCuttedImage = cv2.flip(grabCuttedImage, 0)
                                transMaskImage = cv2.flip(transMaskImage, 0)

                          grabCuttedImage = Image.fromarray(grabCuttedImage)
                          transMaskImage = Image.fromarray(transMaskImage)
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

def rgb_to_2D_label(label):

  Weed = '#FF0000'.lstrip('#')
  Weed = np.array(tuple(int(Weed[i:i+2], 16) for i in (0, 2, 4))) # 255, 0, 0

  Crop = '#00FF00'.lstrip('#')
  Crop = np.array(tuple(int(Crop[i:i+2], 16) for i in (0, 2, 4))) #0, 255, 0

  Unlabeled = '#000000'.lstrip('#') 
  Unlabeled = np.array(tuple(int(Unlabeled[i:i+2], 16) for i in (0, 2, 4))) #0, 0, 0

  label_seg = np.zeros(label.shape,dtype=np.uint8)
  label_seg [np.all(label == Weed,axis=-1)] = 0
  label_seg [np.all(label==Crop,axis=-1)] = 1
  label_seg [np.all(label==Unlabeled,axis=-1)] = 2
  
  label_seg = label_seg[:,:,0]  
  
  return label_seg