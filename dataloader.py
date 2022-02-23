import numpy as np
from tensorflow.keras.utils import to_categorical, Sequence
from paste import rgb_to_2D_label, pasteOnOriginalImage

class OCARPDataloder(Sequence):    
    def __init__(self, dataset, isOcarp , backgroundHex ,batch_size=1, shuffle=False, grabCuttedImageList = None, transMaskList = None, isPasteAugment = True, isOnlyPasteOnBg = True,  objectPasteCount = 1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        self.on_epoch_end()
        self.grabCuttedImageList = grabCuttedImageList
        self.transMaskList = transMaskList
        self.isOcarp = isOcarp
        self.backgroundHex = backgroundHex
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
                image, mask = pasteOnOriginalImage(image, mask, self.grabCuttedImageList, self.transMaskList, self.backgroundHex, self.isPasteAugment, self.isOnlyPasteOnBg)

            #Convert rgb to 2d label
            label = rgb_to_2D_label(mask)

            X.append(image)
            y.append(label)

        X = np.array(X)
        y = np.array(y)
        y = np.expand_dims(y, axis=3)
        n_classes = 3
        y = to_categorical(y, num_classes=n_classes)
        return X, y
    
    def __len__(self):
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)  