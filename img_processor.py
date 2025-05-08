from PIL import Image
import numpy as np
import cv2
import os
import config
from sklearn.metrics import pairwise_distances_argmin_min


def image_preprocessing(image_path):
    """
    Loads, mirrors, and converts an image to RGB and LAB formats.

    Args:
        image_path (str): Path to the input image.

    Returns:
        tuple: (img_rgb, img_lab, height, width, total_pixels)
    """
    # Load the image
    img = cv2.imread(image_path)

    # Mirror the image for correct tufting
    img = cv2.flip(img, 1)
    # Rotate image 90 degrees if height > width
    if img.shape[0] > img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    # Convert to RGB format for further processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get image dimensions
    height, width, _ = img.shape
    total_pixels = height * width

    # Convert to LAB color space for better clustering
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    return img_rgb, img_lab, height, width, total_pixels


def color_processor(img_lab, num_clusters=config.NUM_CLUSTERS, color_distance_threshold=config.COLOR_DISTANCE_THRESHOLD, min_percentage=config.MIN_PERCENTAGE):
    """
    Applies K-means clustering to extract dominant colors and refine them.

    Args:
        img_lab (numpy.ndarray): Image in LAB color space.
        num_clusters (int): Number of K-means clusters.
        color_distance_threshold (int): Minimum LAB distance to separate colors.
        min_percentage (float): Minimum % of image area a color must occupy.

    Returns:
        tuple: (refined_labels, significant_clusters)
    """
    # Reshape image for clustering
    reshaped = img_lab.reshape((-1, 3))
    reshaped = np.float32(reshaped)

    # K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(reshaped, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    labels = labels.flatten()

    # Refine clusters using LAB distance
    refined_clusters = []
    for i, center in enumerate(centers):
        if len(refined_clusters) == 0:
            refined_clusters.append(center)
            continue
        _, min_dist = pairwise_distances_argmin_min([center], refined_clusters)
        if np.min(min_dist) > color_distance_threshold:
            refined_clusters.append(center)

    refined_clusters = np.array(refined_clusters)

    # Assign refined labels
    refined_labels = pairwise_distances_argmin_min(centers[labels], refined_clusters)[0]
    refined_labels = refined_labels.reshape(img_lab.shape[:2])

    # Count pixels per cluster
    unique_labels, counts = np.unique(refined_labels, return_counts=True)

    # Filter significant clusters based on min_percentage
    significant_clusters = []
    for i, count in enumerate(counts):
        percentage = (count / img_lab.shape[0] / img_lab.shape[1]) * 100
        if percentage >= min_percentage:
            significant_clusters.append((i, percentage))

    # Sort by largest area
    significant_clusters.sort(key=lambda x: x[1], reverse=True)

    return refined_labels, significant_clusters

def shape_processor(img_rgb, refined_labels, significant_clusters, output_folder):
    extracted_images = []
    os.makedirs(output_folder, exist_ok=True)

    for color_index, (cluster_idx, _) in enumerate(significant_clusters):
        mask = (refined_labels == cluster_idx).astype(np.uint8) * 255
        color_image = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

        color_image_pil = Image.fromarray(color_image).convert("RGBA")
        new_color_data = [(0, 0, 0, 0) if item[:3] == (0, 0, 0) else item for item in color_image_pil.getdata()]
        color_image_pil.putdata(new_color_data)

        color_output_path = os.path.join(output_folder, f"color_{color_index+1}.png")
        color_image_pil.save(color_output_path)
        extracted_images.append(color_output_path)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        shape_counter = 0
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) < 100:
                continue

            shape_counter += 1

            shape_mask = np.zeros_like(mask)
            cv2.drawContours(shape_mask, [contour], -1, color=255, thickness=-1)

            isolated_shape = cv2.bitwise_and(color_image, color_image, mask=shape_mask)

            isolated_shape_pil = Image.fromarray(isolated_shape).convert("RGBA")
            new_shape_data = [(0, 0, 0, 0) if item[:3] == (0, 0, 0) else item for item in isolated_shape_pil.getdata()]
            isolated_shape_pil.putdata(new_shape_data)

            shape_output_path = os.path.join(output_folder, f"color_{color_index+1}_shape_{shape_counter}.png")
            isolated_shape_pil.save(shape_output_path)

            # --- Smooth Circumference using Approximation + Anti-Aliasing ---
            epsilon = config.EPSILON  # Adjust for more/less simplification
            approx_contour = cv2.approxPolyDP(contour, epsilon, closed=True)

            contour_mask = np.zeros_like(mask, dtype=np.uint8)

            # Draw anti-aliased, clean lines
            cv2.polylines(contour_mask, [approx_contour], isClosed=True, color=255, thickness=1, lineType=cv2.LINE_AA)

            # Optional: Gaussian blur + threshold for smooth effect
            blurred = cv2.GaussianBlur(contour_mask, (3, 3), 0)
            _, smooth_contour_mask = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

            # Create pure white outline on black
            white_outline = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            white_outline[smooth_contour_mask == 255] = (255, 255, 255)  # Set contour area to white

            # Save as transparent outside shape (optional)
            white_outline_pil = Image.fromarray(white_outline).convert("RGBA")
            new_white_outline_data = [(0, 0, 0, 0) if item[:3] == (0, 0, 0) else (255, 255, 255, 255) for item in
                                      white_outline_pil.getdata()]
            white_outline_pil.putdata(new_white_outline_data)

            outline_output_path = os.path.join(output_folder,
                                               f"color_{color_index + 1}_shape_{shape_counter}_circumference.png")
            white_outline_pil.save(outline_output_path)

            # --- Extract Filling (Without Inner Shapes) ---
            filled_mask = shape_mask.copy()
            for j, inner_contour in enumerate(contours):
                if hierarchy[0][j][3] == i:
                    cv2.drawContours(filled_mask, [inner_contour], -1, color=0, thickness=-1)

            # Remove possible remaining border using erosion
            filled_mask_eroded = cv2.erode(filled_mask, np.ones((3, 3), np.uint8), iterations=1)

            # Fill with pure white wherever mask is present
            white_fill = np.zeros_like(img_rgb)
            white_fill[filled_mask_eroded == 255] = (255, 255, 255)

            filled_pil = Image.fromarray(white_fill).convert("RGBA")
            new_filled_data = [(0, 0, 0, 0) if item[:3] == (0, 0, 0) else item for item in filled_pil.getdata()]
            filled_pil.putdata(new_filled_data)

            filled_output_path = os.path.join(output_folder, f"color_{color_index+1}_shape_{shape_counter}_filling.png")
            filled_pil.save(filled_output_path)

            extracted_images.extend([shape_output_path, outline_output_path, filled_output_path])

    return extracted_images

def extract_shapes(image_path, output_folder):
    """
    Extracts shapes from an image by processing colors and detecting shapes.

    Args:
        image_path (str): Path to the input image.
        output_folder (str): Folder where the output images will be saved.

    Returns:
        list: List of saved image file paths.
    """
    img_rgb, img_lab, height, width, total_pixels = image_preprocessing(image_path)
    refined_labels, significant_clusters = color_processor(img_lab)
    extracted_images = shape_processor(img_rgb, refined_labels, significant_clusters, output_folder)

    return extracted_images
