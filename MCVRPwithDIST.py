import numpy as np
import matplotlib.pyplot as plt

def calculate_distance_matrix(customers):
    n = len(customers)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = np.sqrt((customers[i][0] - customers[j][0])**2 + (customers[i][1] - customers[j][1])**2)
    return dist_matrix

def greedy_vrp_solver_with_max_distance(customers, vehicle_capacity, max_distance, demands, dist_matrix, num_vehicles, depot=0):
    num_customers = len(customers)
    visited = [False] * num_customers  # Keep track of visited customers
    vehicle_routes = []  # Store routes for each vehicle
    vehicle_info = []  # To store the distance traveled and customer count for each vehicle

    for vehicle_num in range(num_vehicles):
        vehicle_load = 0
        vehicle_distance = 0
        customer_count = 0
        route = [depot]
        current = depot

        while True:
            next_customer = None
            min_dist = float('inf')

            # Find the closest unvisited customer that fits the vehicle's capacity and travel distance constraint
            for i in range(1, num_customers):  # Skip depot (i=0)
                if not visited[i] and vehicle_load < vehicle_capacity:
                    dist = dist_matrix[current][i]
                    # Check if adding this customer exceeds the max travel distance
                    if dist + vehicle_distance + dist_matrix[i][depot] <= max_distance and dist < min_dist:
                        min_dist = dist
                        next_customer = i

            if next_customer is None:  # If no valid next customer, return to depot
                break

            # Add the customer to the route
            route.append(next_customer)
            visited[next_customer] = True
            vehicle_load += demands[next_customer]
            vehicle_distance += min_dist
            customer_count += 1
            current = next_customer

        # Add the distance back to the depot
        vehicle_distance += dist_matrix[current][depot]

        # Return to depot and finish the route
        route.append(depot)
        vehicle_routes.append(route)

        # Store vehicle's info (distance traveled and number of customers)
        vehicle_info.append((vehicle_distance, customer_count))

        # If all customers have been visited, stop assigning vehicles
        if all(visited[1:]):  # All non-depot customers visited
            break

    return vehicle_routes, vehicle_info

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

# New set of customer locations (x, y coordinates), including the depot as customer 0
customers = [(0, 0), (2, 4), (5, 6), (8, 8), (10, 10), (3, 7), (9, 3), (6, 4), (1, 9), (8, 2), (5, 9), (-1, -5), (-5, 4)]
# New customer demands (0 for the depot)
# demands = [0, 4, 3, 7, 6, 2, 5, 3, 4, 2, 7, 2, 3]
demands = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# Smaller size
# # Customer locations (x, y coordinates), the depot is customer 0
# customers = [(0, 0), (2, 4), (5, 6), (8, 8), (12, 5), (12, 2)]
# # Customer demands (0 for the depot)
# demands = [0, 4, 3, 7, 6, 3]

# Number of vehicles available
num_vehicles = 3
# Vehicle capacity: Sets capacity of all vehicles
vehicle_capacity = 17
# Define the new maximum travel distance
max_distance = 30

# Create distance matrix
dist_matrix = calculate_distance_matrix(customers)
print(dist_matrix)

# Re-running the solver with the new maximum travel distance constraint
multi_vehicle_routes_with_distance, vehicle_info = greedy_vrp_solver_with_max_distance(customers, vehicle_capacity, max_distance, demands, dist_matrix, num_vehicles)

# Print Route
print("Routes:", multi_vehicle_routes_with_distance)

# Calculate total distance
total_distance = calculate_total_distance(multi_vehicle_routes_with_distance, dist_matrix)
print("Total Distance:", total_distance)

# Print the distance traveled and number of customers visited for each vehicle
for idx, (distance, customer_count) in enumerate(vehicle_info):
    print(f"Vehicle {idx + 1}:")
    print(f" - Max Capacity: ", vehicle_capacity)
    print(f" - Max Travel Distance: ", max_distance)
    print(f" - Distance traveled: {distance:.2f} units")
    print(f" - Customers visited: {customer_count}")

# Visualize the updated routes with max travel distance constraint
visualize_routes(customers, demands, multi_vehicle_routes_with_distance, dist_matrix)
