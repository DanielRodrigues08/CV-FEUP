import cv2
import numpy as np
from utils import equalize_hist_wrapper
import math
import copy
from utils import display_images
import sys

SAME_COLOR_THRESHOLD = 110
SAME_COLOR_THRESHOLD2 = 40
SAME_COLOR_THRESHOLD3 = 50
MIN_POINTS_COLOR = 0.35
PRECISION = 10

colors_hue = {
    "red": (0, 5),
    "red1": (166, 180),
    "orange": (6, 21),
    "yellow": (22, 35),
    "lime": (36, 45),
    "green": (46, 70),
    "turquoise": (71, 87),
    "cyan": (88, 100),
    "coral": (101, 110),
    "blue": (111, 125),
    "purple": (124, 135),
    "magenta": (136, 155),
    "pink": (156, 165),
}


"""
Check if a pixel represents the color black.

Args:
    pixel (tuple): A tuple representing the BGR values of a pixel.

Returns:
    bool: True if the pixel represents the color black, False otherwise.
"""


def is_black(pixel):
    return pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0


"""
Check if a pixel (i, j) belongs to the same cluster as the given cluster.

Args:
    image (numpy.ndarray): The input image.
    i (int): The row index of the pixel.
    j (int): The column index of the pixel.
    cluster (list): A list of pixels representing a cluster.
    ratio (int, optional): Ratio used for determining neighbors. Defaults to 1.

Returns:
    bool: True if the pixel belongs to the same cluster, False otherwise.
"""


def same_cluster(image, i, j, cluster, ratio=1):
    neighbors = [
        (i - ratio, j - ratio),
        (i - ratio, j),
        (i - ratio, j + ratio),
        (i, j - ratio),
        (i, j + ratio),
        (i + ratio, j - ratio),
        (i + ratio, j),
        (i + ratio, j + ratio),
    ]
    for neighbor in neighbors:
        if (
            neighbor[0] >= 0
            and neighbor[0] < image.shape[0]
            and neighbor[1] >= 0
            and neighbor[1] < image.shape[1]
        ):
            if neighbor in cluster:
                return True
    return False


"""
Merge clusters that are adjacent or overlapping.

Args:
    image (numpy.ndarray): The input image.
    clusters (list): List of clusters.
    ratio (int): Ratio used for determining neighbors.

Returns:
    list: List of merged clusters.
"""


def merge_clusters(image, clusters, ratio):
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            for pixel in clusters[j]:
                if same_cluster(image, pixel[0], pixel[1], clusters[i], ratio):
                    for ele in clusters[j]:
                        clusters[i].append(ele)
                    clusters.pop(j)
                    return clusters
    return clusters


"""
Clear clusters by merging adjacent or overlapping clusters.

Args:
    image (numpy.ndarray): The input image.
    clusters (list): List of clusters.
    ratio (int): Ratio used for determining neighbors.

Returns:
    list: List of cleared clusters.
"""


def clear_clusters(image, clusters, ratio):
    temp = -1
    while temp != len(clusters):
        temp = len(clusters)
        clusters = merge_clusters(image, clusters, ratio)
    return clusters


def create_clusters(img, bbs):
    clusters = []

    for bb in bbs:
        cluster = []
        x, y, w, h = bb

        for i in range(x, x + w, PRECISION):
            for j in range(y, y + h, PRECISION):
                r, g, b = img[j][i]
                if r != 0 or g != 0 or b != 0:
                    cluster.append((j, i))

        clusters.append((cluster, bb))

    return clusters


"""
    Scan clusters and determine their dominant colors.

    Args:
        clusters (list): List of clusters.
        image (numpy.ndarray): The input image.
        min_points_color (int, optional): Minimum points for a color to be considered dominant. Defaults to MIN_POINTS_COLOR.
        colors_hue (dict, optional): Dictionary containing hue values for different colors. Defaults to colors_hue.

    Returns:
        tuple: A tuple containing the number of pieces detected and the number of distinct colors.
    """


def color_scan(
    clusters, image, min_points_color=MIN_POINTS_COLOR, colors_hue=colors_hue
):
    c = 0
    full_colors = set()
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    final_mask = np.zeros(image.shape[:2], np.uint8)

    for item in clusters:
        cluster, bb = item

        x_bb, y_bb, w_bb, h_bb = bb

        bb_img = image_hsv[y_bb : y_bb + h_bb, x_bb : x_bb + w_bb].copy()
        temp_mask = np.zeros((h_bb, w_bb), np.uint8)

        colors = []
        colors_dict = {
            "red": [0, 0],
            "orange": [0, 0],
            "yellow": [0, 0],
            "lime": [0, 0],
            "green": [0, 0],
            "turquoise": [0, 0],
            "cyan": [0, 0],
            "coral": [0, 0],
            "blue": [0, 0],
            "purple": [0, 0],  # create a mask for the bb
            "magenta": [0, 0],
            "pink": [0, 0],
            "white": [0, 0],
        }
        for x, y in cluster:
            h, s, v = image_hsv[x][y]
            for color in colors_hue:
                if color == "red1":
                    continue

                if s <= 60:
                    if v >= 60:
                        colors_dict["white"][0] += 1
                    else:
                        colors_dict["white"][1] += 1
                    break
                if h <= colors_hue[color][1]:
                    if v >= 60:
                        colors_dict[color][0] += 1
                    else:
                        colors_dict[color][1] += 1
                    break
                if color == "pink":
                    if v >= 60:
                        colors_dict["red"][0] += 1
                    else:
                        colors_dict["red"][1] += 1
                    break

        for color in colors_dict:
            bright, dark = colors_dict[color]

            if bright >= len(cluster) * min_points_color:
                colors.append(color)

            if dark >= len(cluster) * min_points_color:
                colors.append("black" if color == "white" else f"dark {color}")

        for color in colors:
            dark_color = color.startswith("dark")
            color_name = color.split(" ")[1] if dark_color else color

            if color_name == "white":
                lower_limit = np.array([0, 0, 60])
                upper_limit = np.array([180, 60, 255])
            elif color_name == "black":
                lower_limit = np.array([0, 0, 0])
                upper_limit = np.array([180, 60, 60])
            else:
                lower_limit = np.array(
                    [colors_hue[color_name][0], 0 if dark_color else 60, 0]
                )
                upper_limit = np.array(
                    [colors_hue[color_name][1], 60 if dark_color else 255, 255]
                )

            mask_color = cv2.inRange(bb_img, lower_limit, upper_limit)
            if color_name == "red":
                lower_limit[0] = colors_hue["red1"][0]
                upper_limit[0] = colors_hue["red1"][1]
                mask_color = cv2.bitwise_or(
                    mask_color, cv2.inRange(bb_img, lower_limit, upper_limit)
                )

            temp_mask = cv2.bitwise_or(mask_color, temp_mask)

        final_mask[y_bb : y_bb + h_bb, x_bb : x_bb + w_bb] = temp_mask

        c += max(1, len(colors) - 1)

        full_colors.update(colors)

    return c, len(full_colors), final_mask


"""
Perform GrabCut segmentation on the given image using the specified rectangle.

Args:
    image (numpy.ndarray): The input image.
    rect (tuple): The rectangle (x, y, width, height) specifying the region of interest.

Returns:
    numpy.ndarray: The segmented image.
"""


def grab_cut(image, rect):
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    return image * mask2[:, :, np.newaxis]


"""
Perform segmentation on the input image using GrabCut with bounding rectangles specified by contours.

Args:
    image (numpy.ndarray): The input image.
    contours (list): List of contours.
    original_image (numpy.ndarray): The original image.

Returns:
    numpy.ndarray: The filtered image after segmentation.
"""


def image_segmentation(image, contours, original_image):
    temp = copy.deepcopy(original_image)
    bbs = [cv2.boundingRect(contour) for contour in contours]
    combination = (15, 160, 100)
    masks = []
    original_image = copy.deepcopy(temp)
    original_image = equalize_hist_wrapper(original_image, *combination)
    for bb_idx in range(len(bbs)):
        mask = np.zeros(original_image.shape[:2], np.uint8)
        bb = bbs[bb_idx]

        bgModel = np.zeros((1, 65), np.float64)
        fgModel = np.zeros((1, 65), np.float64)

        cv2.setRNGSeed(0)
        (mask, bgModel, fgModel) = cv2.grabCut(
            original_image, mask, bb, bgModel, fgModel, 10, cv2.GC_INIT_WITH_RECT
        )

        output_mask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1)

        output_mask = (output_mask * 255).astype("uint8")
        masks.append(copy.deepcopy(output_mask))

    merged_mask = np.zeros_like(masks[0])
    for mask in masks:
        merged_mask = cv2.bitwise_or(merged_mask, mask)
    filtered_image = cv2.bitwise_and(image, image, mask=merged_mask)

    return filtered_image


"""
Perform background removal on the given image using Canny edge detection and contour extraction.

Args:
    image (numpy.ndarray): The input image.

Returns:
    tuple: A tuple containing the resulting image with the background removed and the extracted contours.
"""


def background_removal(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_hsv[:, :, 1] = np.clip(image_hsv[:, :, 1] + 12, 0, 255)
    image_hsv[:, :, 2] = np.clip(image_hsv[:, :, 2] + 3, 0, 255)
    image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    image = cv2.medianBlur(image, 15)
    image = cv2.GaussianBlur(image, (3, 3), sigmaX=0)

    edges = cv2.Canny(image, 50, 125)

    edges = cv2.dilate(edges, None, iterations=6)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = np.zeros_like(image)
    for contour in contours:
        if cv2.contourArea(contour) > 0:
            cv2.drawContours(result, [contour], 0, (255, 255, 255), cv2.FILLED)

    result = cv2.bitwise_and(image, result)

    return result, contours


"""
Resize the input image while preserving aspect ratio.

Args:
    image (numpy.ndarray): The input image.
    height (int, optional): The target height for resizing. Defaults to 800.

Returns:
    numpy.ndarray: The resized image.
"""


def resize_image(image, height=800):
    ratio = image.shape[1] / image.shape[0]
    height = 800
    width = int(height * ratio)

    dim = (width, height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return image


def create_bounding_boxes(mask, img):
    img_with_bbs = img.copy()

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bbs = [cv2.boundingRect(contour) for contour in contours]

    for bb in bbs:
        x, y, w, h = bb
        cv2.rectangle(img_with_bbs, (x, y), (x + w, y + h), (0, 0, 255), 5)

    return bbs, img_with_bbs


def main(image_path):

    # 1. Load the image
    image = cv2.imread(image_path)

    # 2. Resize the image
    image = resize_image(image)

    original_image = image.copy()

    # 3. Image preprocessing: Background removal
    result, contours = background_removal(image)
    result = image_segmentation(result, contours, original_image)

    bbs = [cv2.boundingRect(contour) for contour in contours]

    # 4. Perform DBSCAN on the image
    clusters = create_clusters(result, bbs)

    # 5. Scan clusters and determine their dominant colors
    num_clusters, num_colors, mask = color_scan(clusters, result)

    bbs, img_with_bbs = create_bounding_boxes(mask, original_image)

    display_images([img_with_bbs], ["Bounding Boxes"])

    print(num_clusters)


if __name__ == "__main__":
    main(sys.argv[1])
