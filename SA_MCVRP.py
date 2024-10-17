import numpy as np
import matplotlib.pyplot as plt
import random
import math

def calculate_distance_matrix(customers):
    n = len(customers)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = np.sqrt((customers[i][0] - customers[j][0])**2 + (customers[i][1] - customers[j][1])**2)
    return dist_matrix

def greedy_cvrp_solver_multiple_vehicles(customers, vehicle_capacity, demands, dist_matrix, num_vehicles, depot=0):
    num_customers = len(customers)
    visited = [False] * num_customers  # Keep track of visited customers
    vehicle_routes = []  # Store routes for each vehicle

    for _ in range(num_vehicles):
        vehicle_load = 0
        route = [depot]
        current = depot

        while True:
            next_customer = None
            min_dist = float('inf')

            # Find the closest unvisited customer that fits in the vehicle's remaining capacity
            for i in range(1, num_customers):  # Skip depot (i=0)
                if not visited[i] and demands[i] + vehicle_load <= vehicle_capacity:
                    dist = dist_matrix[current][i]
                    if dist < min_dist:
                        min_dist = dist
                        next_customer = i

            if next_customer is None:  # If no valid next customer, return to depot
                break

            # Add the customer to the route
            route.append(next_customer)
            visited[next_customer] = True
            vehicle_load += demands[next_customer]
            current = next_customer

        # Return to depot and finish the route
        route.append(depot)
        vehicle_routes.append(route)

        # If all customers have been visited, stop assigning vehicles
        if all(visited[1:]):  # All non-depot customers visited
            break

    return vehicle_routes

def generate_neighbor(routes):
    """Generate a neighboring solution by swapping two customers."""
    # Randomly select two routes
    new_routes = [route[:] for route in routes]  # Deep copy of routes
    route1, route2 = random.sample(new_routes, 2)
    
    if len(route1) > 2 and len(route2) > 2:  # Ensure routes have customers to swap
        # Randomly select a customer (excluding depot) from each route
        idx1, idx2 = random.randint(1, len(route1) - 2), random.randint(1, len(route2) - 2)
        # Swap the customers between routes
        route1[idx1], route2[idx2] = route2[idx2], route1[idx1]

    return new_routes

def simulated_annealing(initial_routes, dist_matrix, initial_temp=1000, cooling_rate=0.99, min_temp=0.01):
    """Simulated Annealing algorithm for CVRP."""
    current_solution = initial_routes
    current_cost = calculate_total_distance(current_solution, dist_matrix)
    best_solution = current_solution
    best_cost = current_cost

    temperature = initial_temp

    while temperature > min_temp:
        # Generate a neighboring solution
        neighbor_solution = generate_neighbor(current_solution)
        neighbor_cost = calculate_total_distance(neighbor_solution, dist_matrix)

        # If the new solution is better, accept it
        if neighbor_cost < current_cost:
            current_solution = neighbor_solution
            current_cost = neighbor_cost

            # Update the best solution found so far
            if neighbor_cost < best_cost:
                best_solution = neighbor_solution
                best_cost = neighbor_cost
        else:
            # Accept worse solutions with some probability
            prob_accept = math.exp((current_cost - neighbor_cost) / temperature)
            if random.random() < prob_accept:
                current_solution = neighbor_solution
                current_cost = neighbor_cost

        # Cool down the temperature
        temperature *= cooling_rate

    return best_solution, best_cost

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
demands = [0, 4, 3, 7, 6, 2, 5, 3, 4, 2, 7, 2, 3]
# demands = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# Smaller size
# # Customer locations (x, y coordinates), the depot is customer 0
# customers = [(0, 0), (2, 4), (5, 6), (8, 8), (12, 5), (12, 2)]
# # Customer demands (0 for the depot)
# demands = [0, 4, 3, 7, 6, 3]

# Number of vehicles available
num_vehicles = 3
# Vehicle capacity: Sets capacity of all vehicles
vehicle_capacity = 17

# Create distance matrix
dist_matrix = calculate_distance_matrix(customers)
print(dist_matrix)

# Initial greedy solution (use your current greedy solution to generate routes)
initial_routes = greedy_cvrp_solver_multiple_vehicles(customers, vehicle_capacity, demands, dist_matrix, num_vehicles)

# Apply Simulated Annealing
best_solution, best_cost = simulated_annealing(initial_routes, dist_matrix)


# print("Routes:", multi_vehicle_routes)

# # Calculate total distance
# total_distance = calculate_total_distance(multi_vehicle_routes, dist_matrix)
# print("Total Distance:", total_distance)

# Visualize the best solution found
visualize_routes(customers, demands, best_solution, dist_matrix)
