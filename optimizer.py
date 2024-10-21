
import random
import math
from distance import calculate_total_distance

class SimulatedAnnealingOptimizer:
    def __init__(self, initial_routes, dist_matrix, vehicle_capacities, vehicle_max_distances, demands, time_windows, service_times):
        self.initial_routes = self.apply_3opt_all_routes(initial_routes, dist_matrix)
        self.dist_matrix = dist_matrix
        self.vehicle_capacities = vehicle_capacities
        self.vehicle_max_distances = vehicle_max_distances
        self.demands = demands
        self.time_windows = time_windows
        self.service_times = service_times
        self.best_solution = initial_routes

    def is_feasible(self, routes, demands, dist_matrix, vehicle_capacities, vehicle_max_distances, time_windows, service_times):
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
                    # print(f"Vehicle {route_idx + 1} exceeded max distance constraint at customer {next_customer}.")
                    return False  # Infeasible if the vehicle exceeds the maximum allowed distance

                # Check capacity constraint
                if current_capacity > max_capacity:
                    # print(f"Vehicle {route_idx + 1} exceeded max capacity constraint at customer {next_customer}.")
                    return False  # Infeasible if the vehicle exceeds the max capacity

                # Calculate arrival time at the next customer
                arrival_time = current_time + dist
                start_time, end_time = time_windows[next_customer]

                # Check if arrival time is within time window
                if arrival_time > end_time:
                    # print(f"Vehicle {route_idx + 1} missed the time window at customer {next_customer}.")
                    return False  # Arrival time is too late
                if arrival_time < start_time:
                    # Vehicle arrives too early and has to wait
                    arrival_time = start_time  # Wait until the start of the time window
                    # print(f"Vehicle {route_idx + 1} waiting until {start_time} to service customer {next_customer}.")

                # Update the current time (after waiting, if necessary)
                current_time = arrival_time + service_times  # Add the service time
                current = next_customer

            # Add distance to return to depot
            vehicle_distance += dist_matrix[current][route[-1]]

            # Check max travel distance constraint again for return trip to depot
            if vehicle_distance > max_distance:
                # print(f"Vehicle {route_idx + 1} exceeded max distance on return to depot.")
                return False

        return True

    def optimize(self, initial_temp=20000, cooling_rate=0.999, min_temp=0.01, max_iterations=1000):
        """Simulated Annealing algorithm for CVRP with vehicle-specific capacities, max distances, and time windows."""
        current_solution = self.initial_routes
        current_cost = calculate_total_distance(current_solution, self.dist_matrix)
        best_solution = current_solution
        best_cost = current_cost

        temperature = initial_temp
        iterations = 0

        while temperature > min_temp and iterations < max_iterations:
            # Generate a neighboring solution
            neighbor_solution = self.generate_neighbor(current_solution)

            # Check if the neighbor solution is feasible
            if not self.is_feasible(neighbor_solution, self.demands,self.dist_matrix, self.vehicle_capacities, self.vehicle_max_distances, self.time_windows, self.service_times):
                iterations += 1
                continue  # Skip if the neighbor solution is not feasible

            neighbor_cost = calculate_total_distance(neighbor_solution, self.dist_matrix)

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

    def generate_neighbor(self, routes):
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
            new_routes = self.reverse_segment_in_route(new_routes)

        else:
            # Relocate a segment of customers between two routes
            new_routes = self.relocate_segment_between_routes(new_routes)

        return new_routes

    def reverse_segment_in_route(self, routes):
        """Reverse a segment of a route."""
        new_routes = [route[:] for route in routes]  # Deep copy of routes

        # Pick a random route
        route_idx = random.randint(0, len(new_routes) - 1)
        route = new_routes[route_idx]

        if len(route) > 3:  # Ensure there are enough customers to reverse a segment
            i, j = sorted(random.sample(range(1, len(route) - 1), 2))  # Exclude depot
            route[i:j+1] = reversed(route[i:j+1])  # Reverse the segment between i and j

        return new_routes

    def relocate_segment_between_routes(self, routes):
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

    def calculate_distance(self, node1, node2, dist_matrix):
        """Return the distance between two nodes."""
        return dist_matrix[node1][node2]

    def three_opt_swap(self, route, i, j, k):
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

    def apply_3opt_route(self, route, dist_matrix):
        """Apply 3-opt local search to a single route and return an improved route."""
        best_route = route
        best_distance = calculate_total_distance([best_route], dist_matrix)

        for i in range(1, len(route) - 2):  # Choose three different points
            for j in range(i + 1, len(route) - 1):
                for k in range(j + 1, len(route)):
                    # Generate possible new routes after 3-opt
                    new_routes = self.three_opt_swap(route, i, j, k)
                    
                    # Evaluate each new route
                    for new_route in new_routes:
                        new_distance = calculate_total_distance([new_route], dist_matrix)
                        if new_distance < best_distance:
                            best_route = new_route
                            best_distance = new_distance

        return best_route

    def apply_3opt_all_routes(self, routes, dist_matrix):
        """Apply 3-opt local search to all routes."""
        improved_routes = []
        for route in routes:
            improved_route = self.apply_3opt_route(route, dist_matrix)
            improved_routes.append(improved_route)
        return improved_routes

