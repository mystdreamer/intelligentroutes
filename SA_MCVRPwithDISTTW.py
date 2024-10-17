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

def greedy_cvrp_solver_multiple_vehicles(customers, vehicle_capacity, demands, dist_matrix, num_vehicles, max_distance, depot=0):
    num_customers = len(customers)
    visited = [False] * num_customers  # Keep track of visited customers
    vehicle_routes = []  # Store routes for each vehicle

    for vehicle_num in range(num_vehicles):
        vehicle_load = 0
        vehicle_distance = 0
        route = [depot]
        current = depot

        while True:
            next_customer = None
            min_dist = float('inf')

            # Find the closest unvisited customer that fits in the vehicle's remaining capacity and respects the max distance constraint
            for i in range(1, num_customers):  # Skip depot (i=0)
                if not visited[i] and demands[i] + vehicle_load <= vehicle_capacity:
                    dist_to_next = dist_matrix[current][i]
                    dist_to_depot = dist_matrix[i][depot]
                    # Check if adding this customer would exceed the max distance constraint
                    if vehicle_distance + dist_to_next + dist_to_depot <= max_distance and dist_to_next < min_dist:
                        min_dist = dist_to_next
                        next_customer = i

            if next_customer is None:  # If no valid next customer, return to depot
                break

            # Add the customer to the route
            route.append(next_customer)
            visited[next_customer] = True
            vehicle_load += demands[next_customer]
            vehicle_distance += min_dist
            current = next_customer

        # Return to depot and finish the route
        vehicle_distance += dist_matrix[current][depot]  # Add return to depot distance
        route.append(depot)
        vehicle_routes.append(route)

        # If all customers have been visited, stop assigning vehicles
        if all(visited[1:]):  # All non-depot customers visited
            break

    return vehicle_routes

def is_feasible(routes, dist_matrix, max_distance, time_windows, service_time):
    """
    Check if the routes are feasible with respect to max travel distance and time windows.
    """
    for route_idx, route in enumerate(routes):
        vehicle_distance = 0
        current_time = 0
        current = route[0]  # Start at depot
        
        for i in range(1, len(route) - 1):
            next_customer = route[i]
            dist = dist_matrix[current][next_customer]
            vehicle_distance += dist

            # Debugging: print the vehicle's distance at each step
            print(f"Route {route_idx + 1}, Step {i}, Current Distance: {vehicle_distance}")

            # Check max travel distance constraint
            if vehicle_distance > max_distance:
                print(f"Route {route_idx + 1}: Distance exceeded max limit. Distance: {vehicle_distance}, Max Distance: {max_distance}")
                return False  # Infeasible if the vehicle exceeds the maximum allowed distance

            # Calculate arrival time and check time window constraints
            arrival_time = current_time + dist
            start_time, end_time = time_windows[next_customer]
            
            # Check if arrival time is within time window
            if arrival_time > end_time:
                print(f"Route {route_idx + 1}: Arrival time exceeded time window. Arrival: {arrival_time}, Window: {start_time}-{end_time}")
                return False
            if arrival_time < start_time:
                arrival_time = start_time  # Wait if early

            current_time = arrival_time + service_time  # Add service time
            current = next_customer

        # Add distance to return to depot
        vehicle_distance += dist_matrix[current][route[-1]]
        print(f"Route {route_idx + 1}, Returning to Depot, Total Distance: {vehicle_distance}")

        # Check max travel distance constraint again for return trip to depot
        if vehicle_distance > max_distance:
            print(f"Route {route_idx + 1}: Distance exceeded max limit on return to depot. Total Distance: {vehicle_distance}, Max Distance: {max_distance}")
            return False

    return True

def generate_neighbor(routes):
    """Generate a neighboring solution by swapping two customers between routes or within a single route."""
    new_routes = [route[:] for route in routes]  # Deep copy of routes

    if len(new_routes) < 2:
        # If there's only one route, perform a swap within the same route
        route = new_routes[0]
        if len(route) > 2:  # Ensure the route has enough customers to swap (excluding depot)
            idx1, idx2 = random.sample(range(1, len(route) - 1), 2)  # Exclude depot
            route[idx1], route[idx2] = route[idx2], route[idx1]
    else:
        # Swap customers between two different routes
        route1, route2 = random.sample(new_routes, 2)
        
        if len(route1) > 2 and len(route2) > 2:  # Ensure routes have customers to swap
            idx1, idx2 = random.randint(1, len(route1) - 2), random.randint(1, len(route2) - 2)
            # Swap the customers between routes
            route1[idx1], route2[idx2] = route2[idx2], route1[idx1]

    return new_routes

def simulated_annealing_cvrp(
    initial_routes, dist_matrix, max_distance, time_windows, service_time,
    initial_temp=1000, cooling_rate=0.95, min_temp=0.01, max_iterations=10000
):
    """Simulated Annealing algorithm for CVRP with multiple vehicles, max distance, and time windows."""
    current_solution = initial_routes
    current_cost = calculate_total_distance(current_solution, dist_matrix)
    best_solution = current_solution
    best_cost = current_cost

    temperature = initial_temp
    iterations = 0

    while temperature > min_temp and iterations < max_iterations:
        # Generate a neighboring solution
        neighbor_solution = generate_neighbor(current_solution)

        # Check if the neighbor solution is feasible
        if not is_feasible(neighbor_solution, dist_matrix, max_distance, time_windows, service_time):
            print(f"Iteration {iterations}: Neighbor solution is infeasible due to max distance or time windows.")
            iterations += 1
            continue  # Skip if the neighbor solution is not feasible

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

        # Cool down the temperature and increment iterations
        temperature *= cooling_rate
        iterations += 1

    return best_solution, best_cost


def calculate_total_distance(routes, dist_matrix):
    total_distance = 0
    for route in routes:
        for i in range(len(route) - 1):
            total_distance += dist_matrix[route[i]][route[i + 1]]
    return total_distance

# Add the information printing for each vehicle after Simulated Annealing
def print_vehicle_info(customers, dist_matrix, best_solution, vehicle_capacity, max_distance, time_windows, service_time):
    # Calculate and store vehicle information
    vehicle_info = []  # [(distance_traveled, customers_visited)]
    vehicle_arrival_info = []  # [[(customer_idx, arrival_time, (start_time, end_time)), ...], ...]

    for idx, route in enumerate(best_solution):
        vehicle_distance = 0
        customer_count = 0
        arrival_times = []  # Track arrival times for this vehicle
        current_time = 0
        current = route[0]

        for i in range(1, len(route) - 1):
            next_customer = route[i]
            dist = dist_matrix[current][next_customer]
            vehicle_distance += dist
            arrival_time = current_time + dist
            start_time, end_time = time_windows[next_customer]
            
            # Adjust the arrival time based on the time window (waiting if necessary)
            if arrival_time < start_time:
                arrival_time = start_time  # Vehicle waits if it arrives too early
            
            current_time = arrival_time + service_time  # Add service time
            arrival_times.append((next_customer, current_time, (start_time, end_time)))
            current = next_customer
            customer_count += 1
        
        # Add distance back to depot
        vehicle_distance += dist_matrix[current][route[-1]]
        
        # Store vehicle distance and customer count info
        vehicle_info.append((vehicle_distance, customer_count))
        vehicle_arrival_info.append(arrival_times)

    # Print the distance traveled and number of customers visited for each vehicle
    for idx, (distance, customer_count) in enumerate(vehicle_info):
        print(f"Vehicle {idx + 1}:")
        print(f" - Max Capacity: {vehicle_capacity}")
        print(f" - Max Travel Distance: {max_distance}")
        print(f" - Distance traveled: {distance:.2f} units")
        print(f" - Customers visited: {customer_count}")
        print(" - Arrival details:")
        
        # Retrieve the arrival info for this vehicle
        for customer_idx, arrival_time, (start_time, end_time) in vehicle_arrival_info[idx]:
            x, y = customers[customer_idx]
            print(f"   Customer at location ({x}, {y})")
            print(f"   - Arrival time: {arrival_time:.2f}")
            print(f"   - Time window: {start_time} to {end_time}")


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

# Number of vehicles available
num_vehicles = 2
# Vehicle capacity: Sets capacity of all vehicles
vehicle_capacity = 10
# Define the new maximum travel distance
max_distance = 30

# Example time windows for each customer (start_time, end_time) and a service time
time_windows = [(0, 999)] + [(5, 15), (8, 18), (7, 16), (10, 20), (6, 13), (12, 18), (7, 15), (10, 19), (5, 12), (8, 16), (3, 4), (4, 15)]
service_time = 0.3  # Assume 1 unit of time for service at each customer



# Create distance matrix
dist_matrix = calculate_distance_matrix(customers)
print(dist_matrix)

# Initial greedy solution (use your current greedy solution to generate routes, with max distance check)
initial_routes = greedy_cvrp_solver_multiple_vehicles(customers, vehicle_capacity, demands, dist_matrix, num_vehicles, max_distance)

# Apply Simulated Annealing for CVRP with time windows and max distance
best_solution, best_cost = simulated_annealing_cvrp(
    initial_routes, dist_matrix, max_distance, time_windows, service_time
)

# Calculate total distance
total_distance = calculate_total_distance(best_solution, dist_matrix)
print("Total Travel Distance of all Vehicle:", total_distance)

print("Routes:", best_solution)

# Print vehicle information (including arrival times and route details)
print_vehicle_info(customers, dist_matrix, best_solution, vehicle_capacity, max_distance, time_windows, service_time)

# Visualize the best solution found
visualize_routes(customers, demands, best_solution, dist_matrix)

