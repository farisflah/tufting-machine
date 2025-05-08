import cv2
import numpy as np
import config
import matplotlib.pyplot as plt

def plot_vectors(vectors, output_path):
    """
    Plots the generated vectors as separate arrows and saves the visualization.
    """
    if not vectors:
        print("No vectors to plot.")
        return

    plt.figure(figsize=(8, 6))

    x_vals, y_vals = [], []  # Store x and y values to set plot limits
    first = True

    for i, vec in enumerate(vectors):
        x1, x2, y1, y2, theta = map(float, vec.split(","))
        dx, dy = x2 - x1, y2 - y1

        # Collect x and y values for dynamic axis scaling
        x_vals.extend([x1, x2])
        y_vals.extend([y1, y2])

        # First arrow in a different color to indicate start
        color = 'b' if first else 'r'
        first = False

        # Use plt.quiver to ensure arrows are plotted separately
        plt.quiver(x1, y1, dx, dy, angles='xy', scale_units='xy', scale=1, color=color)

    # Dynamically set plot limits
    plt.xlim(min(x_vals) - 10, max(x_vals) + 10)
    plt.ylim(min(y_vals) - 10, max(y_vals) + 10)

    plt.gca().set_aspect('equal', adjustable='box')
    # Reverse Y-axis to start from bottom-left
    # Flip Y-axis correctly so (0,0) starts at the bottom-left

    plt.ylim(min(y_vals) - 10, max(y_vals) + 10)  # Invert to match coordinate system
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.title("Vector Visualization")

    vector_img_path = output_path.replace(".txt", "_vectors.png")
    plt.savefig(vector_img_path)
    plt.close()
    print(f"Vector visualization saved to {vector_img_path}")


def circumference_vectors(image_path, output_path, real_width):
    """
    Improved method for precise and stable circumference vector generation.
    """
    # Load and threshold
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not load image.")
        return



    _, binary = cv2.threshold(image, config.THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    img_height, img_width = image.shape
    print(f"Image dimensions (pixels): {img_width}x{img_height}")

    ratio = img_width / img_height
    real_height = real_width / ratio
    print(f"Real dimensions (mm): {real_width:.2f} x {real_height:.2f}")

    scale = real_width / img_width

    def pixel_to_real(point):
        x, y = point
        return x * scale, real_height - (y * scale)

    target_vector_length = config.TARGET_VECTOR_LENGTH#mm
    vectors = []

    for contour in contours:
        contour = contour.squeeze()
        if contour.ndim != 2:
            continue

        #convert each pixel coordinate in contour to real-world coordinates
        points_contour = np.array([pixel_to_real(pt) for pt in contour])
        #list of all euclidean distances between consecutive points
        distances = np.sqrt(np.sum(np.diff(points_contour, axis=0) ** 2, axis=1))
        #list of cumulative distance array starting with 0 total distance traveled along contour
        cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

        total_contour_length = cumulative_distances[-1]
        print(f"Contour total length: {total_contour_length:.2f} mm")

        # Step every 7.5mm
        num_vectors = int(total_contour_length / target_vector_length)
        print(f"➡️ Number of 7.5mm vectors: {num_vectors}")

        step_distances = np.linspace(0, total_contour_length, num_vectors + 1)

        points_for_vectors = []

        for dist_target in step_distances:
            idx = np.searchsorted(cumulative_distances, dist_target)

            if idx >= len(points_contour):
                idx = len(points_contour) - 1

            p1 = points_contour[idx - 1]
            p2 = points_contour[idx]

            # Linear interpolation between points
            segment_dist = cumulative_distances[idx] - cumulative_distances[idx - 1]
            if segment_dist == 0:
                interp_point = p1
            else:
                ratio = (dist_target - cumulative_distances[idx - 1]) / segment_dist
                interp_point = p1 + (p2 - p1) * ratio

            points_for_vectors.append(interp_point)

        # Create vectors
        for i in range(len(points_for_vectors) - 1):
            x1, y1 = points_for_vectors[i]
            x2, y2 = points_for_vectors[i + 1]

            dx = x2 - x1
            dy = y2 - y1
            theta = round(np.degrees(np.arctan2(dy, dx)) / 1.8) * 1.8

            vectors.append(f"{round(x1, 1)}, {round(x2, 1)}, {round(y1, 1)}, {round(y2, 1)}, {theta}")

    # Save vectors
    with open(output_path, "w") as f:
        for vec in vectors:
            f.write(vec + "\n")

    print(f"Saved {len(vectors)} vectors to {output_path}")
    plot_vectors(vectors, output_path)

def filling_vectors(image_path, output_path, real_width):
    """
    Generates horizontal filling vectors spaced every 5mm vertically for white-filled areas in the image.
    All vectors are approximately 7.5mm (±0.5) long and have an angle of 90°.
    """
    # Load and threshold
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not load image.")
        return

    _, binary = cv2.threshold(image, config.THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

    img_height, img_width = binary.shape
    print(f"Image dimensions (pixels): {img_width}x{img_height}")

    ratio = img_width / img_height
    real_height = real_width / ratio
    print(f"Real dimensions (mm): {real_width:.2f} x {real_height:.2f}")
    scale = real_width / img_width  # mm per pixel
    step_y_mm = config.TARGET_VECTOR_WIDTH

    step_y_px = int(step_y_mm / scale)
    if step_y_px < 1:
        step_y_px = 1

    target_vector_length = config.TARGET_VECTOR_LENGTH
    target_vector_approx = config.TARGET_VECTOR_APPROX
    vectors = []

    def pixel_to_real(x, y):
        return x * scale, real_height - (y * scale)

    for y in range(0, img_height, step_y_px):
        row = binary[y]
        inside = False
        start_x = 0

        for x in range(img_width):
            if row[x] == 255 and not inside:
                inside = True
                start_x = x
            elif row[x] != 255 and inside:
                inside = False
                end_x = x - 1

                real_start_x, real_y = pixel_to_real(start_x, y)
                real_end_x, _ = pixel_to_real(end_x, y)
                line_len = real_end_x - real_start_x

                if line_len < 1:
                    continue

                n_segments = int(round(line_len / target_vector_length))
                if n_segments == 0:
                    continue

                segment_lengths = []
                avg_len = line_len / n_segments

                for i in range(n_segments):
                    length = avg_len
                    if abs(length - target_vector_length) > target_vector_approx:
                        length = round(min(target_vector_length + target_vector_approx, max(target_vector_length - target_vector_approx, length)), 1)
                    segment_lengths.append(length)

                x_pos = real_start_x
                for seg_len in segment_lengths:
                    x1 = x_pos
                    x2 = x1 + seg_len
                    vectors.append(f"{round(x1, 1)}, {round(x2, 1)}, {round(real_y, 1)}, {round(real_y, 1)}, 90.0")
                    x_pos = x2

        # Handle trailing white area
        if inside:
            real_start_x, real_y = pixel_to_real(start_x, y)
            real_end_x, _ = pixel_to_real(img_width - 1, y)
            line_len = real_end_x - real_start_x

            if line_len > 1:
                n_segments = int(round(line_len / target_vector_length))
                segment_lengths = []
                if n_segments == 0:
                    print(f"Warning: No filling segments found in {image_path}. Skipping.")
                    return
                avg_len = line_len / n_segments
                for i in range(n_segments):
                    length = avg_len
                    if abs(length - target_vector_length) > target_vector_approx:
                        length = round(min(target_vector_length + target_vector_approx, max(target_vector_length - target_vector_approx, length)), 1)
                    segment_lengths.append(length)

                x_pos = real_start_x
                for seg_len in segment_lengths:
                    x1 = x_pos
                    x2 = x1 + seg_len
                    vectors.append(f"{round(x1, 1)}, {round(x2, 1)}, {round(real_y, 1)}, {round(real_y, 1)}, 90.0")
                    x_pos = x2

    with open(output_path, "w") as f:
        for vec in vectors:
            f.write(vec + "\n")

    print(f"Saved {len(vectors)} filling vectors to {output_path}")
    plot_vectors(vectors, output_path)
