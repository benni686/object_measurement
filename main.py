import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate the midpoint between two points
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# Load the image
image = cv2.imread("test.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# Use edge detection
edges = cv2.Canny(blurred, 50, 100)

# Find contours in the edged image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours from left-to-right (optional)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

# Known width of the reference object (e.g., credit card or coin in cm)
known_width = 8.5  # example for a credit card width in centimeters

# Loop over the contours to find the reference object and the target object
for contour in contours:
    # Ignore small contours that are not likely the objects we're looking for
    if cv2.contourArea(contour) < 500:
        continue

    # Compute the bounding box of the contour
    box = cv2.minAreaRect(contour)
    box = cv2.boxPoints(box)  # Get 4 corner points of the bounding box
    box = np.array(box, dtype="int")

    # Order the points for clarity (top-left, top-right, bottom-right, bottom-left)
    box = sorted(box, key=lambda x: x[0])

    # Draw the bounding box on the image
    box = np.array(box, dtype="int")  # Convert list to NumPy array
    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)


    # Compute the midpoints for the width and height of the bounding box
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # Calculate the Euclidean distance between the midpoints (width and height)
    width = np.sqrt(((tltrX - blbrX) ** 2) + ((tltrY - blbrY) ** 2))
    height = np.sqrt(((tlblX - trbrX) ** 2) + ((tlblY - trbrY) ** 2))

    # Assuming the first detected object is the reference object
    if 'scale' not in locals():
        # Compute the pixel-per-metric ratio (pixels per cm)
        scale = width / known_width
        print(f"Pixel per cm ratio: {scale}")

    else:
        # Measure the target object
        object_width_cm = width / scale
        object_height_cm = height / scale

        # Display the measurements on the image
        cv2.putText(image, f"Width: {object_width_cm:.1f} cm", (int(tltrX - 10), int(tltrY - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(image, f"Height: {object_height_cm:.1f} cm", (int(trbrX + 10), int(trbrY)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Show the final output image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for proper color display
plt.title("Measured Image")
plt.axis('off')  # Hide the axis
plt.show()
cv2.imwrite("measured_image_output.jpg", image)
print("Image saved as 'measured_image_output.jpg'")