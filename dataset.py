import os
import cv2
import numpy as np
from PIL import Image
from extract import breakImageParts, findIndexAllBoxesWithObject, cropBoxMaskAndImageArray, obtainTransparentObjectMask, obtainMaskForGrabCut, performGrabCutOnImage

class OCARPDataset:
    def __init__(self, images_dir, masks_dir):
        self.ids = os.listdir(images_dir)
        #Paths of the images and labels are saved in lists
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

    # Read images and labels into NumPy arrays and returns them
    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i], 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 1)
        mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
        return image, mask
    
    #The number of images in the dataset is returned
    def __len__(self):
        return len(self.ids)
    
    #Object Extraction Pathway
    def ocarpExtractObjects(self,  extractObjectHex, otherObjectHex, backgroundHex, objectPixelThreshold, image_size):
        grabCuttedImageList = []
        transMaskList = []
        crop_box_size = int(image_size/4) 

        for original_image_path, original_mask_path in zip(self.images_fps, self.masks_fps):
          original_image = cv2.imread(original_image_path, 1)
          original_mask = cv2.imread(original_mask_path, 1)

          original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
          original_mask = cv2.cvtColor(original_mask, cv2.COLOR_BGR2RGB)

          #Break the image into 16 different parts
          object_boxes_mask_array = breakImageParts(original_mask,crop_box_size)

          #From the cropped box masks obtained, find the ones with objects in them
          object_included_index_array = findIndexAllBoxesWithObject(object_boxes_mask_array, extractObjectHex, objectPixelThreshold)
          #Obtain the cropped object image and masks array from an original image        
          object_cropped_boxes, object_cropped_masks = cropBoxMaskAndImageArray(object_included_index_array, original_mask, original_image, crop_box_size)
          
          for object_cropped_box,object_cropped_mask  in zip(object_cropped_boxes,object_cropped_masks):

              object_cropped_mask = np.array(object_cropped_mask)
              object_cropped_mask = cv2.cvtColor(object_cropped_mask, cv2.COLOR_BGRA2RGBA)

              #Removing background and other objects from the mask
              trans_mask = obtainTransparentObjectMask(object_cropped_mask, otherObjectHex, backgroundHex)
              #Converting mask into black and white for GrabCut Algorithm
              grab_cut_mask = obtainMaskForGrabCut(object_cropped_mask, extractObjectHex, otherObjectHex, backgroundHex)
          
          
              object_cropped_box = np.asarray(object_cropped_box).astype(np.uint8)
              grab_cut_mask = np.asarray(grab_cut_mask).astype(np.uint8)
          
              object_cropped_box = cv2.cvtColor(object_cropped_box, cv2.COLOR_RGBA2RGB)
              grab_cut_mask = cv2.cvtColor(grab_cut_mask, cv2.COLOR_RGBA2RGB)
          
              #Perform grab cut algorithm using the image and mask, to obtain the segmented object from the original image
              grabCuttedImage = performGrabCutOnImage(object_cropped_box,grab_cut_mask)
              
              grabCuttedImage = Image.fromarray(np.uint8(grabCuttedImage))
              trans_mask = Image.fromarray(np.uint8(trans_mask))
              grabCuttedImageList.append(grabCuttedImage)
              transMaskList.append(trans_mask)

        print("Number of Extracted Object images: ", len(grabCuttedImageList))
        print("Number of Extracted Object masks: ", len(transMaskList))
        return grabCuttedImageList, transMaskList
        
