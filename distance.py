
import numpy as np
import math

def haversine_distance(coord1, coord2):
    """
    Calculate the Haversine distance between two points (latitude, longitude).
    The result is in kilometers.
    """
    # Radius of Earth in kilometers
    R = 6371.0

    # Unpack latitude/longitude of both coordinates
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Compute differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance in kilometers
    distance = R * c

    return distance

def calculate_distance_matrix(customers):
    n = len(customers)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = np.sqrt((customers[i][0] - customers[j][0])**2 + (customers[i][1] - customers[j][1])**2)
    return dist_matrix

def calculate_distance_matrix_2(customers):
    """
    Calculate the distance matrix using the Haversine formula for real-world coordinates.
    """
    n = len(customers)
    dist_matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i, n):
            if i != j:
                dist_matrix[i][j] = haversine_distance(customers[i], customers[j])
                dist_matrix[j][i] = dist_matrix[i][j]  # Distance is symmetric

    return dist_matrix

def calculate_total_distance(routes, dist_matrix):
    total_distance = 0
    for route in routes:
        for i in range(len(route) - 1):
            total_distance += dist_matrix[route[i]][route[i + 1]]
    return total_distance