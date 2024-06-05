import numpy as np
import cv2
import os

# Global variables to store coordinates of the selected region
rect_start = (0, 0)
rect_end = (0, 0)
selecting_region = False

def sketch(frame):
    """
    Generate sketch given an image.

    Args:
        frame: The input image.

    Returns:
        A sketch of the input image.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Invert the grayscale image
    inverted = 255 - gray

    # Apply Gaussian blur to the inverted image
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)

    # Blend the inverted image with the blurred image using color dodge
    blended = cv2.divide(gray, 255 - blurred, scale=256)

    # Apply dodging and burning for shading
    shaded = cv2.addWeighted(blended, 1.2, cv2.GaussianBlur(blended, (0, 0), 10), -0.2, 0)

    # Darken the human figures

    darkened = shaded.copy()
    human_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
    darkened[human_mask != 0] = darkened[human_mask != 0] * 0.8

    return darkened

def select_region(event, x, y, flags, param):
    """
    Mouse event callback function to select a region for detailed sketching.

    Args:
        event: The mouse event (e.g., cv2.EVENT_LBUTTONDOWN).
        x: The x-coordinate of the mouse event.
        y: The y-coordinate of the mouse event.
        flags: Additional flags passed by OpenCV.
        param: Additional parameters passed by OpenCV.
    """
    global rect_start, rect_end, selecting_region

    if event == cv2.EVENT_LBUTTONDOWN:
        rect_start = (x, y)
        rect_end = (x, y)  # Initialize both start and end points
        selecting_region = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if selecting_region:
            rect_end = (x, y)  # Update the end point while dragging

    elif event == cv2.EVENT_LBUTTONUP:
        rect_end = (x, y)
        selecting_region = False

# Capture video from webcam
capture = cv2.VideoCapture(0)

# Create a window and set the mouse event callback function
cv2.namedWindow("Sketch")
cv2.setMouseCallback("Sketch", select_region)

while True:
    # Read a frame from the webcam
    response, frame = capture.read()

    # Create a copy of the frame to display the selected region
    frame_copy = frame.copy()

    # Draw a rectangle around the selected region
    if not selecting_region and rect_start != rect_end:
        # Extract the selected region
        x, y, w, h = rect_start[0], rect_start[1], rect_end[0] - rect_start[0], rect_end[1] - rect_start[1]

        # Draw the rectangle on the copied frame
        cv2.rectangle(frame_copy, rect_start, rect_end, (0, 255, 0), 2)

        # Display the frame with the rectangle
        cv2.imshow("Sketch", frame_copy)

    # Sketch only the selected region
    if not selecting_region and rect_start != rect_end:
        # Extract the selected region
        x, y, w, h = rect_start[0], rect_start[1], rect_end[0] - rect_start[0], rect_end[1] - rect_start[1]
        selected_region = frame[y:y + h, x:x + w]

        # Create a sketch of the selected region
        sketch_region = sketch(selected_region)

        # Convert sketch_region to 3-channel (BGR) image
        sketch_region_bgr = cv2.cvtColor(sketch_region, cv2.COLOR_GRAY2BGR)

        # Replace the selected region with the sketch
        frame[y:y + h, x:x + w] = sketch_region_bgr

        # Save the sketch image
        sketch_image_path = "sketch_image.jpg"
        cv2.imwrite(sketch_image_path, sketch_region)

    # Display the frame with the sketch
    cv2.imshow("Sketch", frame)

    # Exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam resource
capture.release()

# Close all open windows
cv2.destroyAllWindows()
