import math
import random

# Function to calculate the distance between two coordinates
def calculate_distance(loc1, loc2):
    return math.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)


# Function to generate random coordinates within Australia geography
def generate_random_coordinates(num_coords):
    """
    Generate random coordinates within Australia's convex hull without using GeoPandas.
    """
    # Convex hull of Australia (simplified)
    australia_convex_hull = [
        {"latitude": -10.6871474, "longitude": 142.5315554},  # Cape York
        {"latitude": -22.4460593, "longitude": 150.7536412},  # Reef Point
        {"latitude": -28.6333542, "longitude": 153.6385685},  # Cape Byron
        {"latitude": -37.5045900, "longitude": 149.9772920},  # Cape Howe
        {"latitude": -39.1367387, "longitude": 146.3738349},  # South Point
        {"latitude": -34.8424630, "longitude": 116.0022580},  # Point D'Entrecasteaux
        {"latitude": -26.1522827, "longitude": 113.1560635},  # Steep Point
        {"latitude": -21.9737528, "longitude": 113.9335147},  # North West Cape
        {"latitude": -11.1293700, "longitude": 131.9751000},  # Vashon Head
        {"latitude": -10.6871474, "longitude": 142.5315554}   # Closing the polygon
    ]

    # Helper function to check if a point is inside the convex hull
    def is_point_in_polygon(lat, lon, polygon):
        # Ray-Casting Algorithm
        num = len(polygon)
        inside = False
        x, y = lon, lat

        for i in range(num):
            j = (i + 1) % num
            xi, yi = polygon[i]["longitude"], polygon[i]["latitude"]
            xj, yj = polygon[j]["longitude"], polygon[j]["latitude"]

            intersect = ((yi > y) != (yj > y)) and \
                (x < (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi)
            if intersect:
                inside = not inside

        return inside

    # Latitude and longitude bounds for generating random coordinates
    australia_bounds = {
        "lat_min": -44.0, "lat_max": -10.0,
        "lon_min": 112.0, "lon_max": 154.0
    }

    # Generate random coordinates
    coordinates = []
    for _ in range(num_coords):
        while True:
            lat = random.uniform(australia_bounds["lat_min"], australia_bounds["lat_max"])
            lon = random.uniform(australia_bounds["lon_min"], australia_bounds["lon_max"])

            # Check if the generated point is inside the convex hull
            if is_point_in_polygon(lat, lon, australia_convex_hull):
                coordinates.append({"latitude": lat, "longitude": lon})
                break
    return coordinates
