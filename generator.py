import numpy as np
import matplotlib.pyplot as plt
import random
import json
import folium

# Define the coordinates of the bounding box
sw_corner = [-37.8501, 145.0225]  # Southwest corner
ne_corner = [-37.7927, 145.1119]  # Northeast corner

# Generate a fixed depot location
def generate_depot():
    return [-37.8220, 145.0669]  # fixed location for the depot

def visualize_routes(customers, demands):
    customer_coords = np.array(customers)
    demands_array = np.array(demands)

    # Plotting the customers and depot
    plt.figure(figsize=(8, 6))
    plt.scatter(customer_coords[:, 0], customer_coords[:, 1], s=demands_array * 100, c='b', alpha=0.5, label='Customers')
    plt.scatter(customer_coords[0, 0], customer_coords[0, 1], s=300, c='r', label='Depot', edgecolor='black')

    # Adding labels for customers
    for i, txt in enumerate(demands):
        plt.annotate(f"C{i} (demand: {txt})", (customer_coords[i, 0] + 0.1, customer_coords[i, 1] + 0.1))

    # Set plot titles and labels
    plt.title("Customer and Depot Locations")
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to visualize customers and depot on a Folium map
def visualize_on_folium(customers, demands, map_name="cvrp_map.html"):
    # Create a Folium map centered around the depot (customer[0])
    m = folium.Map(location=customers[0], zoom_start=13)

    # Add a marker for the depot
    folium.Marker(customers[0], popup="Depot", icon=folium.Icon(color='red')).add_to(m)

    # Add markers for each customer
    for i, (location, demand) in enumerate(zip(customers[1:], demands[1:]), start=1):
        folium.CircleMarker(
            location=location,
            radius=5 + demand,  # The size of the marker depends on the demand
            popup=f"Customer {i} (Demand: {demand})",
            color='blue',
            fill=True,
            fill_opacity=0.7
        ).add_to(m)

    # Save the map to an HTML file
    m.save(map_name)
    print(f"Map saved as {map_name}")

# Generate random customers, demands, and time windows
def generate_customers(n_customers, bottom_left_corner, top_right_corner):
    customers = []
    for _ in range(n_customers):
        lat = random.uniform(bottom_left_corner[0], top_right_corner[0])
        lon = random.uniform(bottom_left_corner[1], top_right_corner[1])
        customers.append([lat, lon])
    return customers

def generate_demands(n_customers, min_demand=1, max_demand=1):
    return [0] + [random.randint(min_demand, max_demand) for _ in range(n_customers - 1)]

def generate_time_windows(n_customers, min_time=5, max_time=20):
    return [(0, 999)] + [(random.randint(min_time, max_time - 5), random.randint(min_time + 5, max_time)) for _ in range(n_customers - 1)]

# Save the generated data to a JSON file
def save_to_json(customers, demands, time_windows, file_name="coordinates.json"):
    data = {
        "customers": customers,
        "demands": demands,
        "time_windows": time_windows,
    }
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to {file_name}")

# Generate a set of customer locations (including the depot at index 0)
n_customers = 200  # Including depot as customer 0
depot = generate_depot()
customers = [depot] + generate_customers(n_customers - 1, sw_corner, ne_corner)

# Generate demands for the customers (0 for the depot)
demands = generate_demands(n_customers)

# Generate time windows for the customers (depot has a wide window)
time_windows = generate_time_windows(n_customers)

# Visualize routes
visualize_routes(customers, demands)

# Example usage with generated data
visualize_on_folium(customers, demands)

# Save data to JSON
save_to_json(customers, demands, time_windows)

print("Customers (Latitude, Longitude):")
print(customers)
print("Demands:")
print(demands)
print("Time Windows:")
print(time_windows)

