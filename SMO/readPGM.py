import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


DEFAULT_IMAGE = r"C:\Users\vincchen\Documents\1_Assignment\184-Newron_UMC-L14\data\c0displaymi.pgm" 
image = Image.open(DEFAULT_IMAGE)
array = np.array(image)
input_array_nor = (array / float(array.max()))*255
plt.figure()
plt.imshow(array, cmap='gray')
plt.show()
