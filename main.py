import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate the midpoint between two points
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# Load the image
image = cv2.imread("test7.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# Apply Adaptive Thresholding
thresh = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 9, 3
)

# Apply morphological opening to refine binary image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Detect edges using Canny edge detection with dynamic thresholds
sigma = 0.6
v = np.median(blurred)
lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255, (1.0 + sigma) * v))
edges = cv2.Canny(blurred, lower, upper)

# Use Hough Lines to enhance edge detection
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=10, maxLineGap=100)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(edges, (x1, y1), (x2, y2), 255, 1)

# Find contours
contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours by area to ignore noise
contours = [c for c in contours if cv2.contourArea(c) > 1000]

# Sort contours by area (smallest to largest)
contours = sorted(contours, key=cv2.contourArea)

# Known width of the reference object (e.g., credit card in cm)
known_width = 5.5

reference_found = False
sheet_measured = False

# Process contours
for contour in contours:
    # Compute the bounding box of the contour
    box = cv2.minAreaRect(contour)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    # Draw the bounding box on the image
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

    # Assume the first detected object is the reference object
    if not reference_found:
        scale = width / known_width
        reference_found = True
        print(f"Pixel per cm ratio: {scale:.2f}")
    elif reference_found and not sheet_measured:
        # Measure the larger object
        sheet_width_cm = width / scale
        sheet_height_cm = height / scale

        # Annotate the dimensions on the image
        cv2.putText(image, f"Width: {sheet_width_cm:.1f} cm", (int(tltrX - 10), int(tltrY - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(image, f"Height: {sheet_height_cm:.1f} cm", (int(trbrX + 10), int(trbrY)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        sheet_measured = True
        break  # Stop processing after measuring the sheet

# Display the final image
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Measured Sheet and Reference Object")
plt.show()

# Save the processed image
cv2.imwrite("measured_output.jpg", image)
print("Image saved as 'measured_output.jpg'")
