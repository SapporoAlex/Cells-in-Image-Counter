import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("dots.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding
_, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Count the number of contours (dots)
number_of_dots = len(contours)
print(f"Number of black dots: {number_of_dots}")

# Draw the contours on the original image (optional)
output_image = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)

# Display the original image with contours
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.show()
