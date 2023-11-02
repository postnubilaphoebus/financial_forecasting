import torch
import torchvision
from torch import nn
import sys
import cv2
from PIL import Image
import numpy as np
from cnn import five_day_cnn
 
#Load the image
image = cv2.imread("20_days_num_120_up.png")
image = image.transpose(2, 0, 1)
image = np.asarray(image)
image = (image - np.min(image))/np.ptp(image)
image = torch.from_numpy(image).double()
image = torchvision.transforms.functional.rgb_to_grayscale(image)

model = five_day_cnn().double()
x = model.forward(image)
print("x", x)


 
# # Display the image
# cv2.imshow("Image", image)
 
# # Wait for the user to press a key
# cv2.waitKey(0)
 
# # Close all windows
# cv2.destroyAllWindows()
