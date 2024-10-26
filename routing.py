import heapq
from vehicle import Vehicle
from distance import calculate_distance_matrix_2

class GreedySolver:
    @staticmethod
    def greedy_algorithm(customers, vehicle_capacities, demands, num_vehicles, vehicle_max_distances, time_windows, service_times, depot=0):
        """
        Greedy algorithm to solve the CVRP with multiple vehicles, each with its own capacity and max distance, and time windows.
        """
        # Filter out customers with nighttime time windows
        filtered_customers = [i for i in range(len(customers)) if time_windows[i][0] >= 6 or time_windows[i][1] <= 22]
        visited = [False] * len(customers)  # Keep track of visited customers for all original customers
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

            # Priority queue to store feasible customers based on distance
            pq = []

            while True:
                next_customer = None

                # Find the closest unvisited customer that fits in the vehicle's remaining capacity and respects the time window
                for i in filtered_customers:  # Iterate only over filtered customers
                    if not visited[i] and demands[i] + vehicle_load <= vehicle_capacity:
                        dist_to_next = dist_matrix[current][i]
                        dist_to_depot = dist_matrix[i][depot]

                        # Calculate the potential arrival time at the next customer
                        arrival_time = current_time + dist_to_next
                        start_time, end_time = time_windows[i]

                        # Allow waiting if arrival is earlier than the start time
                        if arrival_time < start_time:
                            waiting_time = start_time - arrival_time
                            total_time = start_time + service_times
                        else:
                            waiting_time = 0
                            total_time = arrival_time + service_times

                        # Check if the vehicle can arrive within the max distance constraint and within the time window
                        if vehicle_distance + dist_to_next + dist_to_depot <= max_distance and start_time <= total_time <= end_time:
                            # Use a combination of distance and waiting time to prioritize customers
                            heapq.heappush(pq, (dist_to_next + waiting_time, i))  # Push customer with combined distance and waiting time as priority

                if not pq:  # If no valid next customer, return to depot
                    break

                # Get the next customer from the priority queue
                _, next_customer = heapq.heappop(pq)

                # Verify that adding this customer still respects all constraints
                dist_to_next = dist_matrix[current][next_customer]
                dist_to_depot = dist_matrix[next_customer][depot]
                arrival_time = current_time + dist_to_next
                start_time, end_time = time_windows[next_customer]

                # Allow waiting time if necessary
                if arrival_time < start_time:
                    current_time = start_time + service_times
                else:
                    current_time = arrival_time + service_times

                if (vehicle_load + demands[next_customer] <= vehicle_capacity and
                        vehicle_distance + dist_to_next + dist_to_depot <= max_distance and
                        start_time <= current_time <= end_time):
                    # Add the customer to the route
                    route.append(next_customer)
                    visited[next_customer] = True
                    vehicle_load += demands[next_customer]
                    vehicle_distance += dist_to_next
                    current = next_customer

            # Return to depot and finish the route
            vehicle_distance += dist_matrix[current][depot]  # Add return to depot distance
            route.append(depot)
            vehicle_routes.append(route)

            # If all customers have been visited, stop assigning vehicles
            if all(visited[i] for i in filtered_customers):  # All non-depot customers visited
                break

        return vehicle_routes, dist_matrix
