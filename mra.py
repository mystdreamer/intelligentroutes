import numpy as np
import matplotlib.pyplot as plt
import random
import math
import json

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

def greedy_algorithm(customers, vehicle_capacities, demands, num_vehicles, vehicle_max_distances, time_windows, service_times, depot=0):
    """
    Greedy algorithm to solve the CVRP with multiple vehicles, each with its own capacity and max distance, and time windows.
    """
    num_customers = len(customers)
    visited = [False] * num_customers  # Keep track of visited customers
    vehicle_routes = []  # Store routes for each vehicle
    # Create distance matrix
    dist_matrix = calculate_distance_matrix_2(customers)

    for vehicle_num in range(num_vehicles):
        vehicle_capacity = vehicle_capacities[vehicle_num]
        max_distance = vehicle_max_distances[vehicle_num]
        
        vehicle_load = 0
        vehicle_distance = 0
        current_time = 0
        route = [depot]
        current = depot

        while True:
            next_customer = None
            min_dist = float('inf')

            # Find the closest unvisited customer that fits in the vehicle's remaining capacity and respects the time window
            for i in range(1, num_customers):  # Skip depot (i=0)
                if not visited[i] and demands[i] + vehicle_load <= vehicle_capacity:
                    dist_to_next = dist_matrix[current][i]
                    dist_to_depot = dist_matrix[i][depot]

                    # Calculate the potential arrival time at the next customer
                    arrival_time = current_time + dist_to_next
                    start_time, end_time = time_windows[i]

                    # Check if the vehicle can arrive within the time window and max distance constraint
                    if arrival_time <= end_time and vehicle_distance + dist_to_next + dist_to_depot <= max_distance:
                        if arrival_time < start_time:
                            # If arriving early, vehicle will wait until the start time
                            arrival_time = start_time

                        if dist_to_next < min_dist:
                            min_dist = dist_to_next
                            next_customer = i

            if next_customer is None:  # If no valid next customer, return to depot
                break

            # Add the customer to the route
            route.append(next_customer)
            visited[next_customer] = True
            vehicle_load += demands[next_customer]
            vehicle_distance += min_dist

            # Update the current time (including waiting time if arrived early)
            current_time = max(current_time + min_dist, time_windows[next_customer][0]) + service_times
            current = next_customer

        # Return to depot and finish the route
        vehicle_distance += dist_matrix[current][depot]  # Add return to depot distance
        route.append(depot)
        vehicle_routes.append(route)

        # If all customers have been visited, stop assigning vehicles
        if all(visited[1:]):  # All non-depot customers visited
            break

    return vehicle_routes, dist_matrix


def is_feasible(routes, dist_matrix, vehicle_capacities, vehicle_max_distances, time_windows, service_times):
    """
    Check if the routes are feasible with respect to vehicle-specific capacities, max travel distances, and time windows.
    Each customer has an associated service time.
    """
    for route_idx, route in enumerate(routes):
        vehicle_distance = 0
        current_time = 0
        current = route[0]  # Start at depot
        current_capacity = 0

        # Get the specific capacity and max distance for the current vehicle
        max_capacity = vehicle_capacities[route_idx]
        max_distance = vehicle_max_distances[route_idx]

        for i in range(1, len(route) - 1):
            next_customer = route[i]
            dist = dist_matrix[current][next_customer]
            vehicle_distance += dist
            current_capacity += demands[next_customer]

            # Check max travel distance constraint
            if vehicle_distance > max_distance:
                print(f"Vehicle {route_idx + 1} exceeded max distance constraint at customer {next_customer}.")
                return False  # Infeasible if the vehicle exceeds the maximum allowed distance

            # Check capacity constraint
            if current_capacity > max_capacity:
                print(f"Vehicle {route_idx + 1} exceeded max capacity constraint at customer {next_customer}.")
                return False  # Infeasible if the vehicle exceeds the max capacity

            # Calculate arrival time at the next customer
            arrival_time = current_time + dist
            start_time, end_time = time_windows[next_customer]

            # Check if arrival time is within time window
            if arrival_time > end_time:
                print(f"Vehicle {route_idx + 1} missed the time window at customer {next_customer}.")
                return False  # Arrival time is too late
            if arrival_time < start_time:
                # Vehicle arrives too early and has to wait
                arrival_time = start_time  # Wait until the start of the time window
                print(f"Vehicle {route_idx + 1} waiting until {start_time} to service customer {next_customer}.")

            # Update the current time (after waiting, if necessary)
            current_time = arrival_time + service_times  # Add the service time
            current = next_customer

        # Add distance to return to depot
        vehicle_distance += dist_matrix[current][route[-1]]

        # Check max travel distance constraint again for return trip to depot
        if vehicle_distance > max_distance:
            print(f"Vehicle {route_idx + 1} exceeded max distance on return to depot.")
            return False

    return True


def reverse_segment_in_route(routes):
    """Reverse a segment of a route."""
    new_routes = [route[:] for route in routes]  # Deep copy of routes

    # Pick a random route
    route_idx = random.randint(0, len(new_routes) - 1)
    route = new_routes[route_idx]

    if len(route) > 3:  # Ensure there are enough customers to reverse a segment
        i, j = sorted(random.sample(range(1, len(route) - 1), 2))  # Exclude depot
        route[i:j+1] = reversed(route[i:j+1])  # Reverse the segment between i and j

    return new_routes

def relocate_segment_between_routes(routes):
    """Relocate a segment of customers from one route to another."""
    new_routes = [route[:] for route in routes]  # Deep copy of routes

    if len(new_routes) > 1:
        # Select two different routes randomly
        route1, route2 = random.sample(new_routes, 2)

        if len(route1) > 3:  # Ensure there's enough to relocate
            # Select a segment from route1
            start_idx, end_idx = sorted(random.sample(range(1, len(route1) - 1), 2))
            segment_to_move = route1[start_idx:end_idx]

            # Remove the segment from route1
            del route1[start_idx:end_idx]

            # Insert the segment into a random position in route2
            insert_pos = random.randint(1, len(route2) - 1)
            route2[insert_pos:insert_pos] = segment_to_move

    return new_routes


def generate_neighbor(routes):
    """Generate a neighboring solution by swapping, reversing segments, or relocating a customer segment, with vehicle-specific constraints."""
    new_routes = [route[:] for route in routes]  # Deep copy of routes

    # Choose between different neighbor generation strategies
    random_choice = random.random()

    if random_choice < 0.33:
        # Perform customer swap between routes or within a route
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

    elif random_choice < 0.66:
        # Reverse a segment of a route
        new_routes = reverse_segment_in_route(new_routes)

    else:
        # Relocate a segment of customers between two routes
        new_routes = relocate_segment_between_routes(new_routes)

    return new_routes

def simulated_annealing(
    initial_routes, dist_matrix, demands, vehicle_capacities, vehicle_max_distances, time_windows, service_times,
    initial_temp=20000, cooling_rate=0.999, min_temp=0.01, max_iterations=1000):
    """Simulated Annealing algorithm for CVRP with vehicle-specific capacities, max distances, and time windows."""
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
        if not is_feasible(neighbor_solution, dist_matrix, vehicle_capacities, vehicle_max_distances, time_windows, service_times):
            iterations += 1
            continue  # Skip if the neighbor solution is not feasible

        neighbor_cost = calculate_total_distance(neighbor_solution, dist_matrix)

        # If the new solution is better, accept it
        if neighbor_cost < current_cost:
            print(f"Iteration {iterations}: Found better solution with cost {neighbor_cost:.2f}")
            current_solution = neighbor_solution
            current_cost = neighbor_cost

            # Update the best solution found so far
            if neighbor_cost < best_cost:
                print(f"Iteration {iterations}: Updating best solution with cost {neighbor_cost:.2f}")
                best_solution = neighbor_solution
                best_cost = neighbor_cost
        else:
            # Accept worse solutions with some probability
            prob_accept = math.exp((current_cost - neighbor_cost) / temperature)
            if random.random() < prob_accept:
                print(f"Iteration {iterations}: Accepting worse solution with cost {neighbor_cost:.2f}")
                current_solution = neighbor_solution
                current_cost = neighbor_cost

        # Cool down the temperature and increment iterations
        temperature *= cooling_rate
        iterations += 1

    # #visualize when it's half of the iteration
    # if iterations % (max_iterations/2) == 0:
    #     visualize_routes(customers, demands, current_solution, dist_matrix)
    #     # Calculate total distance
    #     total_distance = calculate_total_distance(current_solution, dist_matrix)
    #     print("Total Travel Distance of all Vehicle during : ",max_iterations, total_distance)

    # #visualize the last iteration's route
    # if iterations % max_iterations == 0:
    #     visualize_routes(customers, demands, current_solution, dist_matrix)
    #     # Calculate total distance
    #     total_distance = calculate_total_distance(current_solution, dist_matrix)
    #     print("Total Travel Distance of all Vehicle during : ",max_iterations, total_distance)
    print(f"Final best cost: {best_cost:.2f}")
    return best_solution, best_cost


def calculate_total_distance(routes, dist_matrix):
    total_distance = 0
    for route in routes:
        for i in range(len(route) - 1):
            total_distance += dist_matrix[route[i]][route[i + 1]]
    return total_distance

def print_vehicle_info(customers, best_solution, vehicle_capacities, vehicle_max_distances, time_windows, service_time):
    # Calculate and store vehicle information
    vehicle_info = []  # [(distance_traveled, customers_visited)]
    vehicle_arrival_info = []  # [[(customer_idx, arrival_time, (start_time, end_time)), ...], ...]

    for idx, route in enumerate(best_solution):
        vehicle_distance = 0
        customer_count = 0
        arrival_times = []  # Track arrival times for this vehicle
        current_time = 0
        current = route[0]

        # Get the specific capacity and max distance for this vehicle
        max_capacity = vehicle_capacities[idx]
        max_distance = vehicle_max_distances[idx]

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
        max_capacity = vehicle_capacities[idx]
        max_distance = vehicle_max_distances[idx]
        print(f"Vehicle {idx + 1}:")
        print(f" - Max Capacity: {max_capacity}")
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
    colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'k', 'b', 'g', 'r', 'c', 'm', 'y'] # Add more colors as needed

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

def read_coordinate_data(filename):
    """
    Function to read customer, demand, and time window data from a JSON file.
    """
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
        
        # Extracting the lists from the JSON data
        customers = data.get("customers", [])
        demands = data.get("demands", [])
        time_windows = data.get("time_windows", [])

        # Returning the structured data
        return customers, demands, time_windows

    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error parsing JSON from '{filename}'.")
        return None
    except ValueError as ve:
        print(f"Error in data format: {ve}")
        return None

def calculate_distance(node1, node2, dist_matrix):
    """Return the distance between two nodes."""
    return dist_matrix[node1][node2]

def three_opt_swap(route, i, j, k):
    """
    Perform a 3-opt swap by reconnecting three segments in a different order.
    Returns a list of the possible new route configurations.
    """
    # Slices of the route
    A = route[:i]         # From start to i
    B = route[i:j]        # From i to j
    C = route[j:k]        # From j to k
    D = route[k:]         # From k to end
    
    # Possible new connections after breaking three edges
    # All possible re-arrangements of the segments A, B, C, D
    return [
        A + B + C + D,    # No change
        A + B + C[::-1] + D,  # Reverse C
        A + B[::-1] + C + D,  # Reverse B
        A + B[::-1] + C[::-1] + D,  # Reverse B and C
        A + C + B + D,    # Swap B and C
        A + C[::-1] + B + D,  # Reverse C and then swap B and C
        A + C + B[::-1] + D,  # Reverse B and then swap B and C
        A + C[::-1] + B[::-1] + D  # Reverse B and C, then swap B and C
    ]

def apply_3opt_route(route, dist_matrix):
    """Apply 3-opt local search to a single route and return an improved route."""
    best_route = route
    best_distance = calculate_total_distance([best_route], dist_matrix)

    for i in range(1, len(route) - 2):  # Choose three different points
        for j in range(i + 1, len(route) - 1):
            for k in range(j + 1, len(route)):
                # Generate possible new routes after 3-opt
                new_routes = three_opt_swap(route, i, j, k)
                
                # Evaluate each new route
                for new_route in new_routes:
                    new_distance = calculate_total_distance([new_route], dist_matrix)
                    if new_distance < best_distance:
                        best_route = new_route
                        best_distance = new_distance

    return best_route

def apply_3opt_all_routes(routes, dist_matrix):
    """Apply 3-opt local search to all routes."""
    improved_routes = []
    for route in routes:
        improved_route = apply_3opt_route(route, dist_matrix)
        improved_routes.append(improved_route)
    return improved_routes

def print_total_visited_customers(solution):
    """Print the total number of unique customers visited by all vehicles in the solution."""
    total_customers = 0
    
    # Iterate over each vehicle's route
    for route in solution:
        # Exclude depot (usually represented by 0) from the customer count
        visited_customers = [customer for customer in route if customer != 0]
        total_customers += len(visited_customers)
    
    print(f"Total customers visited by all vehicles: {total_customers}")
    return total_customers

def parse_vehicle_json(filename):
    """
    Parse the JSON file to extract the number of vehicles, their capacities, and maximum travel distances.
    
    Args:
    filename (str): The path to the JSON file containing vehicle data.

    Returns:
    dict: A dictionary with parsed data for number of vehicles, capacities, and max distances.
    """
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Extract number of vehicles, capacities, and max distances
        num_vehicles = int(data.get("num_vehicles", 0))
        vehicle_capacities = data.get("vehicle_capacities", [])
        vehicle_max_distances = data.get("vehicle_max_distances", [])
        
        # Check if the data is valid (lengths of capacities and distances must match num_vehicles)
        if len(vehicle_capacities) != num_vehicles or len(vehicle_max_distances) != num_vehicles:
            raise ValueError("The number of vehicles does not match the length of vehicle capacities or max distances.")
        
        return num_vehicles, vehicle_capacities, vehicle_max_distances

    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error parsing JSON from '{filename}'.")
        return None
    except ValueError as ve:
        print(f"Error in data format: {ve}")
        return None

def run_greedy_algorithm(customers, vehicle_capacity, demands, num_vehicles, vehicle_max_distance, time_windows, service_time):
    initial_routes, dist_matrix = greedy_algorithm(customers, vehicle_capacity, demands, num_vehicles, vehicle_max_distance, time_windows, service_time)
    return initial_routes, dist_matrix

def run_simulated_annealing(customers, num_vehicles, demands, vehicle_capacity, vehicle_max_distance, time_windows, service_time):
    initial_routes, dist_matrix = greedy_algorithm(customers, vehicle_capacity, demands, num_vehicles, vehicle_max_distance, time_windows, service_time)
    initial_routes_3opt = apply_3opt_all_routes(initial_routes, dist_matrix)
    best_solution, best_cost = simulated_annealing(initial_routes_3opt, dist_matrix, demands, vehicle_capacity, vehicle_max_distance, time_windows, service_time)
    return best_solution, best_cost, initial_routes, dist_matrix, initial_routes_3opt

# """Manual Input Coordinates"""
# # New set of customer locations (x, y coordinates), including the depot as customer 0
# customers = [(0, 0), (2, 4), (5, 6), (8, 8), (10, 10), (3, 7), (9, 3), (6, 4), (1, 9), (8, 2), (5, 9), (-1, -5), (-5, 4)]
# # New customer demands (0 for the depot)
# # demands = [0, 4, 3, 7, 6, 2, 5, 3, 4, 2, 7, 2, 3]
# demands = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# # Example time windows for each customer (start_time, end_time) and a service time
# time_windows = [(0, 999)] + [(5, 15), (8, 18), (7, 16), (10, 20), (6, 13), (12, 18), (7, 15), (10, 19), (5, 12), (8, 16), (3, 4), (4, 15)]



"""Reading in Set Coordinates"""
# Assuming the JSON data is saved in a file called 'coordinates.json'

# # Print out the data from the json file
# print("Customers:", customers)
# print("Demands:", demands)
# print("Time Windows:", time_windows)

"""User Input""" #to be implemented
# # Number of vehicles available
# num_vehicles = 5
# # Vehicle capacity: Sets capacity of all vehicles
# vehicle_capacity = 20
# # Define the new maximum travel distance
# max_distance = 60

# # Example vehicle-specific capacities and distances
# num_vehicles = 5
# vehicle_capacity = [20, 25, 30, 22, 18]  # Specific capacity for each vehicle
# vehicle_max_distance = [100, 120, 150, 90, 110]  # Specific max distance for each vehicle

"""
Streamlit app for user input such as:
    - Number of vehicle 
    - Capacity of each vehicle 
    - Max Travel Distance of each vehicle
    - (show recommended amount for each input)
Streamlit app use case:
    - User can select whether to use pre-set coordinates or generate random coordinates, or upload their own (via parser)
    - User can first click to run the GA/Initial Solution
    - User can then select through a drop down menu which optimization technique they'd like to use
    - User can secondly click to run the optimization
"""

"""
A parser to parse in uploaded coordinates from user
    - Will still work if there's only coordinates, no demands and no time window
    - The parser will assume that the demand is 1 for all location
    - The parser will assume that there's no time window if not specified
"""

# Read file from coordinates, generated by generator.py
coordinates_file = 'set_coordinates.json'
customers, demands, time_windows = read_coordinate_data(coordinates_file)

# Read json from streamlit app user input
vehicle_file = 'vehicle_data.json'
num_vehicles, vehicle_capacity, vehicle_max_distance = parse_vehicle_json(vehicle_file)




service_time = 0  # Assume 1 unit of time for service at each customer

# # Initial greedy solution (use your current greedy solution to generate routes, with max distance check)
# initial_routes, dist_matrix = run_greedy_algorithm(customers, vehicle_capacity, demands, num_vehicles, vehicle_max_distance, time_windows, service_time)

# # Apply 3-opt local search to improve the initial greedy solution
# initial_routes_3opt = apply_3opt_all_routes(initial_routes, dist_matrix)

# # Run the simulated annealing algorithm
# best_solution, best_cost = simulated_annealing(initial_routes, dist_matrix, demands, vehicle_capacities, vehicle_max_distances, time_windows, service_times)

# Apply Simulated Annealing for CVRP with time windows and max distance
best_solution, best_cost, initial_routes, dist_matrix, initial_routes_3opt = run_simulated_annealing(
    customers, num_vehicles, demands, vehicle_capacity, vehicle_max_distance, time_windows, service_time
)

# Print improved routes after 3-opt
print("Routes after 3-opt local search:")
for route_idx, route in enumerate(initial_routes_3opt):
    print(f"Vehicle {route_idx + 1}: {route}")

print("Greedy Algo/Initial Algo: ")
# Print vehicle information (including arrival times and route details)
print_vehicle_info(customers, initial_routes, vehicle_capacity, vehicle_max_distance, time_windows, service_time)
# Visualize the best solution found
visualize_routes(customers, demands, initial_routes, dist_matrix)

print("Simulated Annealing: ")
# Print vehicle information (including arrival times and route details)
print_vehicle_info(customers, best_solution, vehicle_capacity, vehicle_max_distance, time_windows, service_time)
# Visualize the best solution found
visualize_routes(customers, demands, best_solution, dist_matrix)

print("Greedy Algo/Initial Algo: ")
print_total_visited_customers(initial_routes)
total_distance = calculate_total_distance(initial_routes, dist_matrix)
print("Total Travel Distance of all Vehicle:", total_distance)
print("Routes:", initial_routes)

print("Simulated Annealing: ")
print_total_visited_customers(best_solution)
total_distance_2 = calculate_total_distance(best_solution, dist_matrix)
print("Total Travel Distance of all Vehicle:", total_distance_2)
print("Routes:", best_solution)