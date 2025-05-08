def extract_vectors(shapes):
    """
    Converts detected shapes into vector coordinates.
    :param shapes: List of polygons.
    :return: List of vectors [(x, y, theta)].
    """
    vectors = []
    for shape in shapes:
        coords = list(shape.exterior.coords)
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]

            # Compute the angle theta
            theta = round((180 / 3.14159) * (np.arctan2(y2 - y1, x2 - x1)), 2)

            vectors.append((x2, y2, theta))  # Store as (X, Y, θ)

    return vectors
