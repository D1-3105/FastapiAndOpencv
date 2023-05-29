import cv2
import numpy as np


def fill_contour(image_bytes, contour, background_color=(0, 0, 0), dilation_iterations=3):
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Create a blank mask with the same size as the input image
    mask = np.zeros(img.shape, dtype=np.uint8)

    # Draw the contour on the mask
    cv2.drawContours(mask, [contour], 0, (255, 255, 255), thickness=cv2.FILLED)

    # Dilate the contour to cover any gaps
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=dilation_iterations)

    # Create a mask with the inverse of the contour
    inverse_mask = cv2.bitwise_not(mask)

    # Apply the inverse mask to the original image
    img = cv2.bitwise_and(img, inverse_mask)

    return img
