import streamlit as st
from routing import GreedySolver
from optimizer import SimulatedAnnealingOptimizer
from data_parser import read_coordinate_data, parse_vehicle_json, save_vehicle_data, parse_json_coordinates
from visualization import visualize_routes, visualize_routes_st, visualize_two_routes_side_by_side
from distance import calculate_total_distance
from generator import generate_all

def print_total_visited_customers(solution):
    """Print the total number of unique customers visited by all vehicles in the solution."""
    total_customers = 0
    
    # Iterate over each vehicle's route
    for route in solution:
        # Exclude depot (usually represented by 0) from the customer count
        visited_customers = [customer for customer in route if customer != 0]
        total_customers += len(visited_customers)

    return total_customers

def calculate_total_items_delivered(routes, demands):
    """
    Calculate the total amount of items delivered by all vehicles in the solution.
    Args:
    - routes: A list of vehicle routes (each route is a list of customer indices).
    - demands: A list of demands for each customer (index 0 is the depot with demand 0)
    
    Returns:
    - total_items_delivered: The total number of items delivered by all vehicles.
    """
    total_items_delivered = 0
    
    for route in routes:
        # Exclude the depot (which is at index 0 in each route)
        for customer in route[1:-1]:  # Skip the first (depot) and last (return to depot) indices
            total_items_delivered += demands[customer]
    
    return total_items_delivered

def print_vehicle_info(self, customers, best_solution, vehicle_capacities, vehicle_max_distances, time_windows, service_time):
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
            dist = self.dist_matrix[current][next_customer]
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
        vehicle_distance += self.dist_matrix[current][route[-1]]
        
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

def print_vehicle_info_st(customers, dist_matrix, best_solution, vehicle_capacities, vehicle_max_distances, time_windows, service_time, demands):
    # Calculate and store vehicle information
    vehicle_info = []  # [(distance_traveled, customers_visited, items_delivered)]
    vehicle_arrival_info = []  # [[(customer_idx, arrival_time, (start_time, end_time)), ...], ...]
    vehicle_output = []  # Store the output to return instead of printing it

    for idx, route in enumerate(best_solution):
        vehicle_distance = 0
        customer_count = 0
        items_delivered = 0
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
            items_delivered += demands[next_customer]  # Increment items delivered by customer's demand
        
        # Add distance back to depot
        vehicle_distance += dist_matrix[current][route[-1]]
        
        # Store vehicle distance, customer count, and items delivered info
        vehicle_info.append((vehicle_distance, customer_count, items_delivered))
        vehicle_arrival_info.append(arrival_times)

        # Prepare vehicle output for Streamlit
        vehicle_output.append(f"Vehicle {idx + 1}:")
        vehicle_output.append(f" - Max Capacity: {max_capacity}")
        vehicle_output.append(f" - Max Travel Distance: {max_distance}")
        vehicle_output.append(f" - Distance traveled: {vehicle_distance:.2f} units")
        vehicle_output.append(f" - Customers visited: {customer_count}")
        vehicle_output.append(f" - Items delivered: {items_delivered}")
        vehicle_output.append(" - Arrival details:")

        # Retrieve the arrival info for this vehicle
        for customer_idx, arrival_time, (start_time, end_time) in vehicle_arrival_info[idx]:
            x, y = customers[customer_idx]
            vehicle_output.append(f"   Customer at location ({x}, {y})")
            vehicle_output.append(f"   - Arrival time: {arrival_time:.2f}")
            vehicle_output.append(f"   - Time window: {start_time} to {end_time}")
    
    return vehicle_output  # Return the collected output

def test():
    # Load customer and vehicle data
    customers, demands, time_windows = read_coordinate_data("set_coordinates.json")
    num_vehicles, vehicle_capacities, vehicle_max_distances = parse_vehicle_json("vehicle_data.json")

    # Run Greedy Algorithm
    initial_routes, dist_matrix = GreedySolver.greedy_algorithm(customers, vehicle_capacities, demands, num_vehicles, vehicle_max_distances, time_windows, service_times=0)
    # Visualize initial routes
    visualize_routes(customers, demands, initial_routes, title="Initial Route")

    # Optimize routes using Simulated Annealing
    sa_optimizer = SimulatedAnnealingOptimizer(initial_routes, dist_matrix, vehicle_capacities, vehicle_max_distances, demands, time_windows, service_times=0)
    optimized_routes, best_cost = sa_optimizer.optimize()
    print(best_cost)
    # Visualize optimized routes
    visualize_routes(customers, demands, optimized_routes, title="Optimized(SA) Route")
    
    total_visited = print_total_visited_customers(initial_routes)
    print(total_visited)
    total_distance = calculate_total_distance(initial_routes, dist_matrix)
    print("Total Travel Distance of all Vehicle:", total_distance)
    print("Routes:", initial_routes)

    total_visited_2 = print_total_visited_customers(optimized_routes)
    print(total_visited_2)
    total_distance_2 = calculate_total_distance(optimized_routes, dist_matrix)
    print("Total Travel Distance of all Vehicle:", total_distance_2)
    print("Routes:", optimized_routes)

def main():
    # Streamlit sidebar interface
    st.sidebar.title("Vehicle Constraints Input")

    # Number of vehicles input in the sidebar
    num_vehicles = st.sidebar.number_input("Enter the number of vehicles:", min_value=1, max_value=10, value=1, step=1)

    # Initialize lists to store vehicle capacities and max distances
    vehicle_capacities = []
    vehicle_max_distances = []
    service_time = 0.2

    # Ensure persistence of solutions
    if 'initial_routes' not in st.session_state:
        st.session_state['initial_routes'] = []

    if 'best_solution' not in st.session_state:
        st.session_state['best_solution'] = []

    if 'dist_matrix' not in st.session_state:
        st.session_state['dist_matrix'] = []

    # Sidebar dynamic input fields for each vehicle
    for i in range(num_vehicles):
        st.sidebar.subheader(f"Vehicle {i + 1} Details")
        vehicle_capacity = st.sidebar.number_input(f"Vehicle {i + 1} Capacity (Recommended: 15):", min_value=1, value=15, step=1, key=f'capacity_{i}')
        vehicle_max_distance = st.sidebar.number_input(f"Vehicle {i + 1} Max Travel Distance (Recommended: 35):", min_value=1, value=35, step=1, key=f'distance_{i}')
        
        vehicle_capacities.append(vehicle_capacity)
        vehicle_max_distances.append(vehicle_max_distance)

    # Button to save input data
    if st.sidebar.button("Save Vehicle Data"):
        vehicle_data = {
            'num_vehicles': num_vehicles,
            'vehicle_capacities': vehicle_capacities,
            'vehicle_max_distances': vehicle_max_distances
        }
        save_vehicle_data(vehicle_data)
        # Read in the json file immediately
        vehicle_file = 'vehicle_data.json'
        num_vehicles, vehicle_capacity, vehicle_max_distance = parse_vehicle_json(vehicle_file)

    if st.sidebar.checkbox("Show Input Data"):
        st.sidebar.write({
            'num_vehicles': num_vehicles,
            'vehicle_capacities': vehicle_capacities,
            'vehicle_max_distances': vehicle_max_distances
        })

    # Coordinate selection options
    st.title("Coordinate Selection")
    coordinate_option = st.selectbox(
        "Choose coordinate input method:",
        ("Use Preset Coordinates", "Generate Random Coordinates", "Upload Custom Coordinates")
    )

    # Handle the user input for coordinates
    if coordinate_option == "Use Preset Coordinates":
        # Reading coordinates file from set coordinate json\
        coordinates_file = 'set_coordinates.json'
        customers, demands, time_windows, customer_groups, num_customers, total_demand = read_coordinate_data(coordinates_file)

    elif coordinate_option == "Generate Random Coordinates":
        total_demand = st.number_input("Total number of items:", min_value=20, value=50, step=1)
        num_customers = st.number_input("Number of customers (items delivery locations):", min_value=15, max_value=total_demand, value=50, step=1)

        if st.button("Generate Coordinates"):
            # Call the generate function with customer and demand details
            generate_all(num_customers + 1, total_demand=total_demand)  # Update function call
            st.success("Random coordinates generated.")
            # Read the generated json file
        coordinates_file = 'coordinates.json'
        customers, demands, time_windows, customer_groups, num_customers, total_demand = read_coordinate_data(coordinates_file)


    elif coordinate_option == "Upload Custom Coordinates":
        uploaded_file = st.file_uploader("Upload a JSON file with coordinates", type=["json"])
        if uploaded_file is not None:
            try:
                customers, demands, time_windows, customer_groups, num_customers, total_demand = parse_json_coordinates(uploaded_file)
                st.write("Customers:", customers)
                st.write("Demands:", demands)
                st.write("Time Windows:", time_windows)
                st.write("Customer Groups:", customer_groups)
            except ValueError as e:
                st.error(f"Error: {e}")

    st.title("Initial Route")
    # Run initial solution (greedy algorithm)
    if st.button("Run Initial Solution (Greedy Algorithm)"):
        if customers:
            st.session_state['initial_routes'], st.session_state['dist_matrix'] = GreedySolver.greedy_algorithm(customers, vehicle_capacities, demands, num_vehicles, vehicle_max_distances, time_windows, service_time)
            st.success("Initial Solution executed.")
            st.write("Numbers of customers: ", num_customers)
            st.write("Numbers of items: ", total_demand)
            visualize_routes_st(customers, customer_groups, demands, st.session_state['initial_routes'], title="Greedy Algorithm Routes")
            st.write("Total Visited Customers (Initial Solution):", print_total_visited_customers(st.session_state['initial_routes']))
            st.write("Total Item Delivered (Initial Solution):", calculate_total_items_delivered(st.session_state['initial_routes'], demands))
            st.write("Total Distance Traveled (Initial Solution):", calculate_total_distance(st.session_state['initial_routes'], st.session_state['dist_matrix']))
            if st.session_state['initial_routes']:
                st.write("Vehicle Information (Initial Solution):", print_vehicle_info_st(customers, st.session_state['dist_matrix'], st.session_state['initial_routes'], vehicle_capacities, vehicle_max_distances, time_windows, service_time, demands))
        else:
            st.error("No coordinates provided.")

    # Show additional information for initial solution
    # # Dropdown menu for selecting optimization technique
    # st.title("Optimization Techniques")
    # optimization_option = st.selectbox(
    #     "Choose an optimization technique:",
    #     ("Simulated Annealing", "Ant Colony Optimization")
    # )

    st.title("Optimizate Route")
    # Run the selected optimization technique
    if (st.session_state['initial_routes']):
        if st.button("Run Optimization"):
        # if optimization_option == "Simulated Annealing":
            sa_optimizer = SimulatedAnnealingOptimizer(st.session_state['initial_routes'], st.session_state['dist_matrix'], vehicle_capacities, vehicle_max_distances, demands, time_windows, service_times=0)
            st.session_state['best_solution'], best_cost = sa_optimizer.optimize()

            st.write("Simulated Annealing Solution Visualization:")
            st.write("Numbers of customers: ", num_customers)
            st.write("Numbers of items: ", total_demand)
            visualize_routes_st(customers, customer_groups, demands, st.session_state['best_solution'], title="Simulated Annealing Routes")
            st.write("Total Visited Customers (Optimized Solution):", print_total_visited_customers(st.session_state['best_solution']))
            st.write("Total Item Delivered (Initial Solution):", calculate_total_items_delivered(st.session_state['initial_routes'], demands))
            st.write("Total Distance Traveled (Optimized Solution):", calculate_total_distance(st.session_state['best_solution'], st.session_state['dist_matrix']))
            if st.session_state['best_solution']:
                    st.write("Vehicle Information (Optimized Solution):", print_vehicle_info_st(customers, st.session_state['dist_matrix'], st.session_state['initial_routes'], vehicle_capacities, vehicle_max_distances, time_windows, service_time, demands))
        # elif optimization_option == "Ant Colony Optimization":
        #     st.write("Unavailable")
        # else:
        #     st.write("No optimization selected.")

    # Run the selected optimization technique
    if (st.session_state['best_solution']):
        if st.button("Compare Results"):
        # if optimization_option == "Simulated Annealing":  
            sa_optimizer = SimulatedAnnealingOptimizer(st.session_state['initial_routes'], st.session_state['dist_matrix'], vehicle_capacities, vehicle_max_distances, demands, time_windows, service_times=0)
            st.session_state['best_solution'], best_cost = sa_optimizer.optimize()
            if st.session_state['initial_routes'] and st.session_state['best_solution']:
                st.write("Comparing Initial Solution and Optimized Solution")
                st.write("Numbers of customers: ", num_customers)
                st.write("Numbers of items: ", total_demand)
                visualize_two_routes_side_by_side(customers, customer_groups, demands, st.session_state['initial_routes'], st.session_state['best_solution'])
                st.write("Total Visited Customers (Initial Solution):", print_total_visited_customers(st.session_state['initial_routes']))
                st.write("Total Item Delivered (Initial Solution):", calculate_total_items_delivered(st.session_state['initial_routes'], demands))
                st.write("Total Distance Traveled (Initial Solution):", calculate_total_distance(st.session_state['initial_routes'], st.session_state['dist_matrix']))
                # Show additional information for initial solution
                if st.session_state['initial_routes']:
                    st.write("Vehicle Information (Initial Solution):", print_vehicle_info_st(customers, st.session_state['dist_matrix'], st.session_state['initial_routes'], vehicle_capacities, vehicle_max_distances, time_windows, service_time, demands))
                st.write("Total Visited Customers (Optimized Solution):", print_total_visited_customers(st.session_state['best_solution']))
                st.write("Total Item Delivered (Initial Solution):", calculate_total_items_delivered(st.session_state['initial_routes'], demands))
                st.write("Total Distance Traveled (Optimized Solution):", calculate_total_distance(st.session_state['best_solution'], st.session_state['dist_matrix']))
                # Show additional information for optimized solution
                if st.session_state['best_solution']:
                    st.write("Vehicle Information (Optimized Solution):", print_vehicle_info_st(customers, st.session_state['dist_matrix'], st.session_state['initial_routes'], vehicle_capacities, vehicle_max_distances, time_windows, service_time, demands))

        # elif optimization_option == "Ant Colony Optimization":
        #     st.write("Unavailable")
        # else:
        #     st.write("No optimization selected.")

    # # Display additional information for initial solution
    # if st.session_state['initial_routes']:
    #     st.write("Vehicle Information (Initial Solution):", print_vehicle_info_st(customers, st.session_state['dist_matrix'], st.session_state['initial_routes'], vehicle_capacities, vehicle_max_distances, time_windows, service_time))

if __name__ == "__main__":
    main()