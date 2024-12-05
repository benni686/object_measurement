import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('test6.jpg')

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Perform Canny edge detection
edges = cv2.Canny(blurred_image, threshold1=10, threshold2=150)

# Create a kernel for dilation
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Dilate the edges to connect them
dilated_edges = cv2.dilate(edges, kernel, iterations=1)

# # Optionally, apply morphological closing to fill gaps
# closed_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, kernel)

# Find contours with cv2.RETR_EXTERNAL
contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours by area to avoid small inner contours
min_contour_area = 1000  # Adjust this value as needed to filter out small details
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

# Create an output image
output_image = image.copy()
cv2.drawContours(output_image, filtered_contours, -1, (0, 255, 0), 2)

# Display the results
plt.figure(figsize=(10, 6))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 2)
plt.title("Edges Detected")
plt.imshow(edges, cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Filtered Contours")
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))

plt.tight_layout()
plt.show()
