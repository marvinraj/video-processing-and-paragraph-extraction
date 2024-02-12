import cv2
import numpy as np

# Read the image
image = cv2.imread("003.png")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Show the grayscale image (uncomment the next 2 lines to view)
# cv2.imshow('Grayscale Image', gray)
# cv2.waitKey(0)

# Apply threshold to get image with only black and white
ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Show the thresholded image (uncomment the next 2 lines to view)
# cv2.imshow('Thresholded Image', th)
# cv2.waitKey(0)

# Dilation to connect individual text elements
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (36, 36))
dilation = cv2.dilate(th, rect_kernel, iterations=1)

# Show the dilated image (uncomment the next 2 lines to view)
# cv2.imshow('Dilated Image', dilation)
# cv2.waitKey(0)

# Find contours of the paragraphs
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Function to sort contours
def sort_contours(cnts):
    # Sort contours based on the x-coordinate to separate different columns
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in cnts]
    (cnts, bounding_boxes) = zip(*sorted(zip(cnts, bounding_boxes), key=lambda b:b[1][0], reverse=False))
    
    # Further sort the contours within the same column
    column_contours = []
    current_column_x = bounding_boxes[0][0]
    current_column = []
    
    for cnt, bbox in zip(cnts, bounding_boxes):
        if bbox[0] > current_column_x + bbox[2]:
            # New column detected, sort the current column by y coordinate
            current_column.sort(key=lambda b: b[1][1])
            column_contours.extend([cnt for cnt, _ in current_column])
            current_column = [(cnt, bbox)]
            current_column_x = bbox[0]
        else:
            current_column.append((cnt, bbox))
    
    # Sort and add the last column
    current_column.sort(key=lambda b: b[1][1])
    column_contours.extend([cnt for cnt, _ in current_column])
    
    return column_contours

# Sort the contours
sorted_contours = sort_contours(contours)

# Visualize the contours
contour_img = image.copy()
for cnt in sorted_contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(contour_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
# Show the image with contours (uncomment the next 2 lines to view)
# cv2.imshow('Contours', contour_img)
# cv2.waitKey(0)

# Crop and save the paragraphs
for i, cnt in enumerate(sorted_contours):
    x, y, w, h = cv2.boundingRect(cnt)
    cropped = image[y:y + h, x:x + w]
    
    # Show each cropped paragraph (uncomment the next 2 lines to view)
    # cv2.imshow(f'Cropped Paragraph {i+1}', cropped)
    # cv2.waitKey(0)
    
    # Save the cropped paragraph as an image
    cv2.imwrite(f"Paragraph_{i+1}.jpg", cropped)

# Close all windows
cv2.destroyAllWindows()