
###Developed By : Prajeeth K T
###Register Number: 212222110034

### Smoothing Filters
# In[1]:Using Averaging Filter

import cv2
import matplotlib.pyplot as plt
import numpy as np
image1 = cv2.imread("demo.jpg")
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
kernel = np.ones((5, 5), np.float32) / 25 
image3 = cv2.filter2D(image2, -1, kernel)
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(image3)
plt.title("Averaged Image")
plt.axis("off")
plt.show()

# In[2]:Using Weighted Averaging Filter

image1 = cv2.imread("sample.jpg")
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
# Define a weighted averaging kernel
kernel1 = np.array([[1, 2, 1], 
                    [2, 4, 2], 
                    [1, 2, 1]], np.float32) / 16
image3 = cv2.filter2D(image2, -1, kernel1)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(image3)
plt.title("Weighted Average Filter Image")
plt.axis("off")
plt.show()

# In[3]:Using Gaussian Filter


image1 = cv2.imread("sample.jpg")
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

gaussian_blur = cv2.GaussianBlur(image2, (15, 15), 0)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(gaussian_blur)
plt.title("Gaussian Blur")
plt.axis("off")
plt.show()

# In[4]:Using Median Filter


image1 = cv2.imread("sample.jpg")
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

median = cv2.medianBlur(image2, 5)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(median)
plt.title("Median Filter")
plt.axis("off")
plt.show()
