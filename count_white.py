import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('2.jpg')

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a binary threshold to isolate white shapes
_, thresh = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)

# Find contours of the white shapes
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Optional: Filter out small contours based on area
min_contour_area = 20  # Adjust this based on the size of your white shapes
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

# Count the number of white shapes (filtered contours)
white_shape_count = len(filtered_contours)

# Output the count
print(f"Number of white shapes: {white_shape_count}")

# Draw the contours on the original image for visualization
output_image = cv2.drawContours(image.copy(), filtered_contours, -1, (0, 255, 0), 2)

# Display the result
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title(f"Detected White Shapes: {white_shape_count}")
plt.show()
