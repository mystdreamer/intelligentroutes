import numpy as np
import matplotlib.pyplot as plt

def calculate_distance_matrix(customers):
    n = len(customers)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = np.sqrt((customers[i][0] - customers[j][0])**2 + (customers[i][1] - customers[j][1])**2)
    return dist_matrix

def greedy_cvrp_solver(customers, vehicle_capacity, demands, dist_matrix, depot=0):
    num_customers = len(customers)
    visited = [False] * num_customers  # Keep track of visited customers
    routes = []

    while not all(visited[1:]):  # Ignore the depot in visited check
        load = 0
        route = [depot]
        current = depot

        while True:
            next_customer = None
            min_dist = float('inf')

            # Find the closest unvisited customer
            for i in range(1, num_customers):  # Skip depot (i=0)
                if not visited[i] and demands[i] + load <= vehicle_capacity:
                    dist = dist_matrix[current][i]
                    if dist < min_dist:
                        min_dist = dist
                        next_customer = i

            if next_customer is None:  # If no valid next customer, return to depot
                break

            # Add the customer to the route
            route.append(next_customer)
            visited[next_customer] = True
            load += demands[next_customer]
            current = next_customer

        # Return to depot and finish the route
        route.append(depot)
        routes.append(route)

    return routes

def calculate_total_distance(routes, dist_matrix):
    total_distance = 0
    for route in routes:
        for i in range(len(route) - 1):
            total_distance += dist_matrix[route[i]][route[i + 1]]
    return total_distance

# Function to visualize the CVRP routes on a scatterplot
def visualize_routes(customers, demands, routes, dist_matrix):
    customer_coords = np.array(customers)
    demands_array = np.array(demands)

    # Plotting the customers and depot
    plt.figure(figsize=(8, 6))
    plt.scatter(customer_coords[:, 0], customer_coords[:, 1], s=demands_array * 100, c='b', alpha=0.5, label='Customers')
    plt.scatter(customer_coords[0, 0], customer_coords[0, 1], s=300, c='r', label='Depot', edgecolor='black')

    # Adding labels for customers
    for i, txt in enumerate(demands):
        plt.annotate(f"C{i} (demand: {txt})", (customer_coords[i, 0] + 0.1, customer_coords[i, 1] + 0.1))

    # Plotting the routes
    for route in routes:
        for i in range(len(route) - 1):
            start = customer_coords[route[i]]
            end = customer_coords[route[i + 1]]
            plt.plot([start[0], end[0]], [start[1], end[1]], 'g-', linewidth=2, alpha=0.6)

    # Set plot titles and labels
    plt.title("CVRP Routes and Customer Demands")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()


# Customer locations (x, y coordinates), the depot is customer 0
customers = [(0, 0), (2, 4), (5, 6), (8, 8), (12, 5), (12, 2)]
# Vehicle capacity
vehicle_capacity = 30
# Customer demands (0 for the depot)
demands = [0, 4, 3, 7, 6, 3]

# Create distance matrix
dist_matrix = calculate_distance_matrix(customers)
print(dist_matrix)

# Solve the problem
routes = greedy_cvrp_solver(customers, vehicle_capacity, demands, dist_matrix)
print("Routes:", routes)

# Calculate total distance
total_distance = calculate_total_distance(routes, dist_matrix)
print("Total Distance:", total_distance)



# Calling the visualization function with the solved routes
visualize_routes(customers, demands, routes, dist_matrix)