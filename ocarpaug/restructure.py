import os
import shutil
import cv2
import splitfolders

'''
The dataset should be structured accordingly for restructureDataset() to function.
        Dataset
            ├── img
            │   └── pic3.jpg
            └── lbl
                 └── pic3.jpg
'''

def restructureDataset(path, original_folder, image_size ,seed, trainRatio, validRatio, testRatio):
    input_folder_path = path + original_folder                   
    dirs_images = os.listdir(input_folder_path+"/img")                                       
    dirs_lbl = os.listdir(input_folder_path+"/lbl")
    
    resized_folder =  original_folder+"_resize_"+str(image_size)
    save_folder_path = input_folder_path.replace(original_folder,resized_folder)
    shutil.rmtree(save_folder_path, ignore_errors=True)
    os.mkdir(save_folder_path)
    
    def resize(dirs, subfolder):
        for image_name in dirs:
            image = cv2.imread(input_folder_path+subfolder+"/"+image_name, 1)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            SIZE_X = image_size
            SIZE_Y = image_size 
            image = cv2.resize(image, (SIZE_X, SIZE_Y),interpolation = cv2.INTER_NEAREST)
            save_path = save_folder_path+subfolder+"/"+image_name
            cv2.imwrite(save_path, cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    
    subfolder = "/img"
    os.mkdir(save_folder_path+subfolder)
    resize(dirs_images, subfolder)
    
    subfolder = "/lbl"
    os.mkdir(save_folder_path+subfolder)
    resize(dirs_lbl , subfolder)
    
    output_folder_path = save_folder_path + "_split_seed_"+str(seed)
    splitfolders.ratio(save_folder_path, output=output_folder_path, seed=seed, ratio=(trainRatio, validRatio, testRatio), group_prefix=None)
    shutil.rmtree(save_folder_path, ignore_errors=True)