import cv2
import numpy as np
from PIL import Image

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

#From the cropped box masks obtained, find the ones with objects in them
def findIndexAllBoxesWithObject(crop_boxes_array, extractObjectHex, objectPixelThreshold):
    extractObjectHex = extractObjectHex.lstrip('#')
    extractObjectRGB = np.array(tuple(int(extractObjectHex[i:i+2], 16) for i in (0, 2, 4)))
    object_rgba = extractObjectRGB[2], extractObjectRGB[1], extractObjectRGB[0], 255 
    object_included_crop_box_array = []
    for box in crop_boxes_array:
        object_pixel_count = 0
        for pixel in box.getdata():
            if pixel == (object_rgba):
                object_pixel_count += 1
        object_included_crop_box_array.append(object_pixel_count)
    object_included_crop_box_array = [i for i,v in enumerate(object_included_crop_box_array) if v > objectPixelThreshold]
    return object_included_crop_box_array
  
  
#Obtain the cropped object image and masks array from an original image          
def cropBoxMaskAndImageArray(object_included_index_array, original_mask, original_image, crop_box_size):
    object_cropped_masks = []
    object_cropped_boxes = []
    for i in object_included_index_array:
        image = breakImageParts(original_image,crop_box_size)[i]
        object_cropped_boxes.append(image)
        
    for j in object_included_index_array:
        mask = breakImageParts(original_mask,crop_box_size)[j]
        object_cropped_masks.append(mask)
    return object_cropped_boxes, object_cropped_masks

#Removing background and other objects from the mask
def obtainTransparentObjectMask(mask, otherObjectHex, backgroundHex):
    trans_mask = np.array(mask) 

    otherObjectHex = otherObjectHex.lstrip('#')
    otherObjectRGB = np.array(tuple(int(otherObjectHex[i:i+2], 16) for i in (0, 2, 4)))

    backgroundHex = backgroundHex.lstrip('#')
    backgroundRGB = np.array(tuple(int(backgroundHex[i:i+2], 16) for i in (0, 2, 4)))
    
    trans_mask [np.all(trans_mask== [otherObjectRGB[0], otherObjectRGB[1], otherObjectRGB[2], 255],axis=-1)] = [0,0,0,0]
    trans_mask [np.all(trans_mask== [backgroundRGB[0], backgroundRGB[1], backgroundRGB[2], 255],axis=-1)] = [0,0,0,0]
    return trans_mask

#Converting mask into black and white for GrabCut Algorithm
def obtainMaskForGrabCut(mask, extractObjectHex, otherObjectHex, backgroundHex):
    grab_cut_mask = np.array(mask) 

    extractObjectHex = extractObjectHex.lstrip('#')
    extractObjectRGB = np.array(tuple(int(extractObjectHex[i:i+2], 16) for i in (0, 2, 4)))

    otherObjectHex = otherObjectHex.lstrip('#')
    otherObjectRGB = np.array(tuple(int(otherObjectHex[i:i+2], 16) for i in (0, 2, 4)))

    backgroundHex = backgroundHex.lstrip('#')
    backgroundRGB = np.array(tuple(int(backgroundHex[i:i+2], 16) for i in (0, 2, 4)))

    grab_cut_mask [np.all(grab_cut_mask== [otherObjectRGB[0], otherObjectRGB[1], otherObjectRGB[2], 255],axis=-1)] = [0,0,0,255]
    grab_cut_mask [np.all(grab_cut_mask== [extractObjectRGB[0], extractObjectRGB[1], extractObjectRGB[2], 255],axis=-1)] = [255,255,255,255]
    grab_cut_mask [np.all(grab_cut_mask== [backgroundRGB[0], backgroundRGB[1], backgroundRGB[2], 255],axis=-1)] = [0,0,0,255]
    return grab_cut_mask

#Perform grab cut algorithm using the image and mask, to obtain the segmented object from the original image
def performGrabCutOnImage(image, grab_cut_mask):
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    mask = np.zeros(image.shape[:2],np.uint8)
    #Converting black and white pixels to 0 and 1 for grab cut algorithm
    mask[np.all(grab_cut_mask == [0, 0, 0],axis=-1)] = 0
    mask[np.all(grab_cut_mask == [255,255,255],axis=-1)] = 1
    #Perform grab cut algorithm
    mask, bgdModel, fgdModel = cv2.grabCut(image,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    grabCuttedImage = image*mask[:,:,np.newaxis]
    #Make the image background transparent
    grabCuttedImage = np.asarray(grabCuttedImage).astype(np.uint8)
    grabCuttedImage = cv2.cvtColor(grabCuttedImage, cv2.COLOR_RGB2RGBA)
    grabCuttedImage [np.all(grabCuttedImage== [0, 0, 0, 255],axis=-1)] = [0,0,0,0]
    return grabCuttedImage