import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('dots3.jpg')

# Convert to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define HSV color ranges for green and yellow
lower_green = np.array([50, 50, 50])  # adjust as needed
upper_green = np.array([90, 255, 255])  # adjust as needed

lower_yellow = np.array([20, 100, 100])  # adjust as needed
upper_yellow = np.array([30, 255, 255])  # adjust as needed

# Create masks for green and yellow
green_mask = cv2.inRange(hsv, lower_green, upper_green)
yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

# Apply Gaussian Blur to reduce noise
green_mask = cv2.GaussianBlur(green_mask, (5, 5), 0)
yellow_mask = cv2.GaussianBlur(yellow_mask, (5, 5), 0)

# Apply morphological operations to clean the masks
kernel = np.ones((5, 5), np.uint8)

# Opening to remove noise
green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

# Closing to close small gaps
green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)

# Find contours in the yellow mask
yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find contours in the green mask
green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize count for green circles with yellow interiors
count = 0

# Loop through each yellow contour and check if it's within a green area
for y_contour in yellow_contours:
    y_contour_mask = np.zeros_like(yellow_mask)
    cv2.drawContours(y_contour_mask, [y_contour], -1, 255, thickness=cv2.FILLED)

    for g_contour in green_contours:
        g_contour_mask = np.zeros_like(green_mask)
        cv2.drawContours(g_contour_mask, [g_contour], -1, 255, thickness=cv2.FILLED)

        # Check if the yellow contour is within the green contour
        overlap = cv2.bitwise_and(g_contour_mask, y_contour_mask)

        if cv2.countNonZero(overlap) > 0:
            count += 1
            break  # Move to the next yellow contour

# Output the count
print(f"Number of green circles with orange interiors: {count}")

# (Optional) Draw the contours on the original image for visualization
output_image = cv2.drawContours(image.copy(), yellow_contours, -1, (0, 255, 0), 2)

# Display the result
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title(f"Detected Green Circles with orange Interiors: {count}")
plt.show()
