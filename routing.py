
from vehicle import Vehicle
from distance import calculate_distance_matrix_2

class GreedySolver:
    @staticmethod
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
    