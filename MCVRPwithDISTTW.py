import numpy as np
import matplotlib.pyplot as plt
import json
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

def greedy_vrp_solver_with_time_windows(customers, vehicle_capacity, max_distance, demands, time_windows, service_time, dist_matrix, num_vehicles, depot=0):
    """
    Solve VRP with time windows for each customer and track arrival times.
    """
    num_customers = len(customers)
    visited = [False] * num_customers  # Keep track of visited customers
    vehicle_routes = []
    vehicle_info = []
    vehicle_arrival_info = []  # Track the time a vehicle arrives at each customer

    for vehicle_num in range(num_vehicles):
        vehicle_load = 0
        vehicle_distance = 0
        current_time = 0
        route = [depot]
        current = depot
        customer_count = 0
        arrival_times = []  # Track arrival times for this vehicle

        while True:
            next_customer = None
            min_dist = float('inf')

            # Find the closest unvisited customer that fits the vehicle's capacity and time window constraint
            for i in range(1, num_customers):  # Skip depot (i=0)
                if not visited[i] and vehicle_load < vehicle_capacity:
                    dist = dist_matrix[current][i]
                    arrival_time = current_time + dist
                    start_time, end_time = time_windows[i]

                    # Check if vehicle can arrive within the time window
                    if arrival_time <= end_time:
                        if arrival_time < start_time:
                            arrival_time = start_time  # Wait if vehicle arrives early
                        if arrival_time + service_time <= end_time:  # Check if service can be completed in time
                            if dist + vehicle_distance + dist_matrix[i][depot] <= max_distance and dist < min_dist:
                                min_dist = dist
                                next_customer = i

            if next_customer is None:  # No valid next customer, return to depot
                break

            # Add the customer to the route
            route.append(next_customer)
            visited[next_customer] = True
            vehicle_load += demands[next_customer]
            vehicle_distance += min_dist
            current_time += min_dist  # Add travel time
            current_time = max(current_time, time_windows[next_customer][0])  # Adjust for waiting time
            current_time += service_time  # Add service time
            arrival_times.append((next_customer, current_time, time_windows[next_customer]))  # Log customer arrival info
            current = next_customer
            customer_count += 1

        # Add the distance to return to the depot
        vehicle_distance += dist_matrix[current][depot]
        route.append(depot)
        vehicle_routes.append(route)

        # Store vehicle's info (distance traveled and number of customers)
        vehicle_info.append((vehicle_distance, customer_count))

        # Store arrival times info for this vehicle
        vehicle_arrival_info.append(arrival_times)

        # If all customers have been visited, stop assigning vehicles
        if all(visited[1:]):  # All non-depot customers visited
            break

    return vehicle_routes, vehicle_info, vehicle_arrival_info

def calculate_total_distance(routes, dist_matrix):
    total_distance = 0
    for route in routes:
        for i in range(len(route) - 1):
            total_distance += dist_matrix[route[i]][route[i + 1]]
    return total_distance


def visualize_routes(customers, demands, routes, dist_matrix):
    customer_coords = np.array(customers)
    demands_array = np.array(demands)

    # Different colors for different vehicle routes
    colors = ['g', 'b', 'm', 'c', 'y']  # Add more colors as needed

    # Plotting the customers and depot
    plt.figure(figsize=(8, 6))
    plt.scatter(customer_coords[:, 0], customer_coords[:, 1], s=demands_array * 100, c='b', alpha=0.5, label='Customers')
    plt.scatter(customer_coords[0, 0], customer_coords[0, 1], s=300, c='r', label='Depot', edgecolor='black')

    # Adding labels for customers
    for i, txt in enumerate(demands):
        plt.annotate(f"C{i} (demand: {txt})", (customer_coords[i, 0] + 0.1, customer_coords[i, 1] + 0.1))

    # Plotting the routes for each vehicle
    for idx, route in enumerate(routes):
        color = colors[idx % len(colors)]  # Cycle through the color list
        for i in range(len(route) - 1):
            start = customer_coords[route[i]]
            end = customer_coords[route[i + 1]]
            plt.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=2, alpha=0.6, label=f"Vehicle {idx + 1}" if i == 0 else "")

    # Set plot titles and labels
    plt.title("CVRP Routes by Vehicles")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

def read_json_data(json_file):
    """
    Function to read customer, demand, and time window data from a JSON file.
    """
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # Extracting the lists from the JSON data
    customers = data.get("customers", [])
    demands = data.get("demands", [])
    time_windows = data.get("time_windows", [])

    # Returning the structured data
    return customers, demands, time_windows



"""Reading in Set Coordinates"""
# Assuming the JSON data is saved in a file called 'coordinates.json'
json_file = 'set_coordinates.json'
customers, demands, time_windows = read_json_data(json_file)

# # New set of customer locations (x, y coordinates), including the depot as customer 0
# customers = [(0, 0), (2, 4), (5, 6), (8, 8), (10, 10), (3, 7), (9, 3), (6, 4), (1, 9), (8, 2), (5, 9), (-1, -5), (-5, 4)]
# demands = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# # Example time windows for each customer (start_time, end_time) and a service time
# time_windows = [(0, 999)] + [(5, 15), (8, 18), (7, 16), (10, 20), (6, 13), (12, 18), (7, 15), (10, 19), (5, 12), (8, 16), (3, 4), (4, 15)]
service_time = 0  # Assume 1 unit of time for service at each customer

# Number of vehicles available
num_vehicles = 7
# Vehicle capacity: Sets capacity of all vehicles
vehicle_capacity = 20
# Define the new maximum travel distance
max_distance = 10

# Create distance matrix
dist_matrix = calculate_distance_matrix_2(customers)
print(dist_matrix)

multi_vehicle_routes_with_time_windows, vehicle_info, vehicle_arrival_info = greedy_vrp_solver_with_time_windows(
    customers, vehicle_capacity, max_distance, demands, time_windows, service_time, dist_matrix, num_vehicles
)
# Print Route
print("Routes:", multi_vehicle_routes_with_time_windows)

# Calculate total distance
total_distance = calculate_total_distance(multi_vehicle_routes_with_time_windows, dist_matrix)
print("Total Distance:", total_distance)

# Print the distance traveled and number of customers visited for each vehicle
for idx, (distance, customer_count) in enumerate(vehicle_info):
    print(f"Vehicle {idx + 1}:")
    print(f" - Max Capacity: ", vehicle_capacity) 
    print(f" - Max Travel Distance: ", max_distance)
    print(f" - Distance traveled: {distance:.2f} units")
    print(f" - Customers visited: {customer_count}")
    print(" - Arrival details:")
    # Retrieve the arrival info for this vehicle
    for customer_idx, arrival_time, (start_time, end_time) in vehicle_arrival_info[idx]:
        x, y = customers[customer_idx]
        print(f"   Customer at location ({x}, {y})")
        print(f"   - Arrival time: {arrival_time:.2f}")
        print(f"   - Time window: {start_time} to {end_time}")

# Visualize the updated routes with max travel distance constraint
visualize_routes(customers, demands, multi_vehicle_routes_with_time_windows, dist_matrix)
