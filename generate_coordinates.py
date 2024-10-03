import json
import random

# Latitude and longitude of the convex hull of Australia
australia_convex_hull = [
    {"latitude": -10.6871474, "longitude": 142.5315554, "location": "Cape York"},
    {"latitude": -10.6871870, "longitude": 142.5316444, "location": "Cape York"},
    {"latitude": -22.4460593, "longitude": 150.7536412, "location": "Reef Point"},
    {"latitude": -25.2762852, "longitude": 152.9081745, "location": "Urangan Pier"},
    {"latitude": -28.6333542, "longitude": 153.6385685, "location": "Cape Byron"},
    {"latitude": -37.5045900, "longitude": 149.9772920, "location": "Cape Howe"},
    {"latitude": -39.1367387, "longitude": 146.3738349, "location": "South Point"},
    {"latitude": -34.8424630, "longitude": 116.0022580, "location": "Point D'Entrecasteaux"},
    {"latitude": -26.1522827, "longitude": 113.1560635, "location": "Steep Point"},
    {"latitude": -22.5776932, "longitude": 113.6537093, "location": "Ningaloo"},
    {"latitude": -21.9737528, "longitude": 113.9335147, "location": "North West Cape"},
    {"latitude": -11.1293700, "longitude": 131.9751000, "location": "Vashon Head"},
    
    # Additional points
    {"latitude": -35.4937, "longitude": 138.7134, "location": "Adelaide"},
    {"latitude": -32.6003, "longitude": 137.7905, "location": "Near Adelaide"},
    {"latitude": -34.8473, "longitude": 135.8789, "location": "Near Adelaide 2"},
    {"latitude": -32.1179, "longitude": 133.2861, "location": "South Point 2"},
    {"latitude": -33.6101, "longitude": 123.3545, "location": "South Point 3"},
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

        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
        if intersect:
            inside = not inside

    return inside

# Latitude and longitude bounds for generating random coordinates (Australia's approximate bounds)
australia_bounds = {
    "lat_min": -44.0, "lat_max": -10.0,
    "lon_min": 112.0, "lon_max": 154.0
}

# Function to generate random land coordinates inside the convex hull
def generate_random_land_coordinates(num_coords):
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

# Generate random land coordinates
num_coords = random.randint(15, 20)
coordinates = generate_random_land_coordinates(num_coords)

# Australia's center point (starting and ending point) with a specific color
australia_center_lat = -25.0
australia_center_lon = 133.0
start_point = {"point": "start/end", "latitude": australia_center_lat, "longitude": australia_center_lon, "color": "red"}

# Add start and end points to the coordinates list
coordinates.insert(0, start_point)

# Convert convex hull points to dictionary format and add to the coordinates list
convex_hull_list = [{"latitude": pt["latitude"], "longitude": pt["longitude"], "color": "blue"} for pt in australia_convex_hull]

# Save the generated coordinates and convex hull to a JSON file
with open("coordinates.json", "w") as file:
    json.dump({
        "num_coords": num_coords,
        "coordinates": coordinates,
        "convex_hull": convex_hull_list
    }, file, indent=4)

print(f"Generated {num_coords} random land coordinates within Australia's convex hull, and saved them to 'coordinates.json'.")
