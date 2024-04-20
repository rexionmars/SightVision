import cv2
import numpy as np
import copy


def stack_images(_image_list, cols, scale):
    """
    Stack a list of images horizontally and vertically to create a grid-like arrangement.

    Args:
        _image_list (list): A list of images to be stacked.
        cols (int): The number of columns in the grid.
        scale (float): The scale factor to resize the images.

    Returns:
        numpy.ndarray: The stacked image grid.
    """
    image_list = copy.deepcopy(_image_list)

    # Make the array full by adding blank img, otherwise the openCV can't work
    total_images = len(image_list)
    rows = total_images // cols if total_images // cols * cols == total_images else total_images // cols + 1
    blank_images = cols * rows - total_images

    width = image_list[0].shape[1]
    height = image_list[0].shape[0]

    # Add blank images
    image_blank = np.zeros((height, width, 3), np.uint8)
    image_list.extend([image_blank] * blank_images)

    # Resize the images
    for i in range(cols * rows):
        image_list[i] = cv2.resize(image_list[i], (0, 0), None, scale, scale)
        if len(image_list[i].shape) == 2:
            image_list[i] = cv2.cvtColor(image_list[i], cv2.COLOR_GRAY2BGR)

    # Put the images in a board
    horizontal = [image_blank] * rows
    for y in range(rows):
        line = []
        for x in range(cols):
            line.append(image_list[y * cols + x])
        horizontal[y] = np.hstack(line)

    vertical = np.vstack(horizontal)

    return vertical


def rounded_rectangle(img,
                      bbox,
                      lenght_of_corner=25,
                      thickness_of_line=1,
                      radius_corner=3,
                      color_rectangle=(25, 220, 255),
                      color_circle=(25, 220, 255)):
    """
    Draws a rounded rectangle on the image
    Args:
        img: Image on which we want to draw
        bbox: Bounding Box of the rectangle
        lenght_of_corner: Length of the corner lines
        thickness_of_line: Thickness of the lines
        radius_corner: Radius of the corner
        color_rectangle: Color of the rectangle
        color_circle: Color of the corner circles
    Returns:
        Image with rounded rectangle
    """
    x, y, w, h = bbox
    x1, y1 = x + w, y + h
    if radius_corner != 0:
        cv2.rectangle(img, bbox, color_rectangle, radius_corner)

    # Top Left  x,y
    cv2.line(img, (x, y), (x + lenght_of_corner, y), color_circle, thickness_of_line)
    cv2.line(img, (x, y), (x, y + lenght_of_corner), color_circle, thickness_of_line)

    # Top Right  x1,y
    cv2.line(img, (x1, y), (x1 - lenght_of_corner, y), color_circle, thickness_of_line)
    cv2.line(img, (x1, y), (x1, y + lenght_of_corner), color_circle, thickness_of_line)

    # Bottom Left  x,y1
    cv2.line(img, (x, y1), (x + lenght_of_corner, y1), color_circle, thickness_of_line)
    cv2.line(img, (x, y1), (x, y1 - lenght_of_corner), color_circle, thickness_of_line)

    # Bottom Right  x1,y1
    cv2.line(img, (x1, y1), (x1 - lenght_of_corner, y1), color_circle, thickness_of_line)
    cv2.line(img, (x1, y1), (x1, y1 - lenght_of_corner), color_circle, thickness_of_line)

    return img


def find_contours(img, imgPre, minArea=1000, sort=True, filter=0, drawCon=True, c=(255, 0, 0)):
    """
    Finds Contours in an image
    :param img: Image on which we want to draw
    :param imgPre: Image on which we want to find contours
    :param minArea: Minimum Area to detect as valid contour
    :param sort: True will sort the contours by area (biggest first)
    :param filter: Filters based on the corner points e.g. 4 = Rectangle or square
    :param drawCon: draw contours boolean
    :return: Foudn contours with [contours, Area, BoundingBox, Center]
    """
    conFound = []
    imgContours = img.copy()
    contours, hierarchy = cv2.findContours(imgPre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > minArea:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # print(len(approx))
            if len(approx) == filter or filter == 0:
                if drawCon:
                    cv2.drawContours(imgContours, cnt, -1, c, 3)
                x, y, w, h = cv2.boundingRect(approx)
                cx, cy = x + (w // 2), y + (h // 2)
                cv2.rectangle(imgContours, (x, y), (x + w, y + h), c, 2)
                cv2.circle(imgContours, (x + (w // 2), y + (h // 2)), 5, c, cv2.FILLED)
                conFound.append({"cnt": cnt, "area": area, "bbox": [x, y, w, h], "center": [cx, cy]})

    if sort:
        conFound = sorted(conFound, key=lambda x: x["area"], reverse=True)

    return imgContours, conFound


def overlayPNG(imgBack, imgFront, pos=[0, 0]):
    hf, wf, cf = imgFront.shape
    hb, wb, cb = imgBack.shape
    *_, mask = cv2.split(imgFront)
    maskBGRA = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
    maskBGR = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    img_rgbA = cv2.bitwise_and(imgFront, maskBGRA)
    img_rgb = cv2.cvtColor(img_rgbA, cv2.COLOR_BGRA2BGR)

    imgMaskFull = np.zeros((hb, wb, cb), np.uint8)
    imgMaskFull[pos[1]:hf + pos[1], pos[0]:wf + pos[0], :] = img_rgb
    imgMaskFull2 = np.ones((hb, wb, cb), np.uint8) * 255
    maskBGRInv = cv2.bitwise_not(maskBGR)
    imgMaskFull2[pos[1]:hf + pos[1], pos[0]:wf + pos[0], :] = maskBGRInv

    imgBack = cv2.bitwise_and(imgBack, imgMaskFull2)
    imgBack = cv2.bitwise_or(imgBack, imgMaskFull)

    return imgBack
