"""
Psuedo code
Streamlit app for user input such as:
    - Number of vehicle 
    - Capacity of each vehicle 
    - Max Travel Distance of each vehicle
    - (show recommended amount for each input)
Streamlit app use case:
    - User can select whether to use pre-set coordinates or generate random coordinates, or upload their own (via parser)
    - User can first click to run the GA/Initial Solution
    - User will be able to see the visualized result below
    - User can then select through a drop down menu which optimization technique they'd like to use
    - User can secondly click to run the optimization
    - User will be able to see the visualized result below
"""


import streamlit as st
import json
import matplotlib.pyplot as plt
import numpy as np
from parser import parse_json_coordinates
from generator import generate_all
from mra import run_greedy_algorithm, run_simulated_annealing, read_coordinate_data, parse_vehicle_json, print_total_visited_customers, calculate_total_distance

# Function to save the vehicle data to a JSON file
def save_to_json(vehicle_data, filename='vehicle_data.json'):
    with open(filename, 'w') as f:
        json.dump(vehicle_data, f, indent=4)
    st.sidebar.success(f'Data saved to {filename}')

def visualize_routes(customers, demands, routes, title):
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
    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)

    # Replace plt.show() with st.pyplot() to display the plot in Streamlit
    st.pyplot(plt)

def visualize_two_routes_side_by_side(customers, demands, routes1, routes2, title1="Initial Solution", title2="Optimized Solution"):
    # Create columns for side-by-side plots
    col1, col2 = st.columns(2)
    customer_coords = np.array(customers)
    demands_array = np.array(demands)

    # Different colors for different vehicle routes
    colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'k', 'b', 'g', 'r', 'c', 'm', 'y'] # Add more colors as needed
    # First plot in the first column
    with col1:
        # Plotting the customers and depot
        plt.figure(figsize=(8, 6))
        plt.scatter(customer_coords[:, 0], customer_coords[:, 1], s=demands_array * 100, c='b', alpha=0.5, label='Customers')
        plt.scatter(customer_coords[0, 0], customer_coords[0, 1], s=300, c='r', label='Depot', edgecolor='black')

        # Adding labels for customers
        for i, txt in enumerate(demands):
            plt.annotate(f"C{i} (demand: {txt})", (customer_coords[i, 0] + 0.1, customer_coords[i, 1] + 0.1))

        # Plotting the routes for each vehicle
        for idx, route in enumerate(routes1):
            color = colors[idx % len(colors)]  # Cycle through the color list
            for i in range(len(route) - 1):
                start = customer_coords[route[i]]
                end = customer_coords[route[i + 1]]
                plt.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=2, alpha=0.6, label=f"Vehicle {idx + 1}" if i == 0 else "")

        # Set plot titles and labels
        plt.title(title1)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.grid(True)

        # Replace plt.show() with st.pyplot() to display the plot in Streamlit
        st.pyplot(plt)

    # Second plot in the second column
    with col2:
        # Plotting the customers and depot
        plt.figure(figsize=(8, 6))
        plt.scatter(customer_coords[:, 0], customer_coords[:, 1], s=demands_array * 100, c='b', alpha=0.5, label='Customers')
        plt.scatter(customer_coords[0, 0], customer_coords[0, 1], s=300, c='r', label='Depot', edgecolor='black')

        # Adding labels for customers
        for i, txt in enumerate(demands):
            plt.annotate(f"C{i} (demand: {txt})", (customer_coords[i, 0] + 0.1, customer_coords[i, 1] + 0.1))

        # Plotting the routes for each vehicle
        for idx, route in enumerate(routes2):
            color = colors[idx % len(colors)]  # Cycle through the color list
            for i in range(len(route) - 1):
                start = customer_coords[route[i]]
                end = customer_coords[route[i + 1]]
                plt.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=2, alpha=0.6, label=f"Vehicle {idx + 1}" if i == 0 else "")

        # Set plot titles and labels
        plt.title(title2)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.grid(True)

        # Replace plt.show() with st.pyplot() to display the plot in Streamlit
        st.pyplot(plt)

def print_vehicle_info(customers, dist_matrix, best_solution, vehicle_capacities, vehicle_max_distances, time_windows, service_time):
    # Calculate and store vehicle information
    vehicle_info = []  # [(distance_traveled, customers_visited)]
    vehicle_arrival_info = []  # [[(customer_idx, arrival_time, (start_time, end_time)), ...], ...]
    vehicle_output = []  # Store the output to return instead of printing it

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

        # Prepare vehicle output for Streamlit
        vehicle_output.append(f"Vehicle {idx + 1}:")
        vehicle_output.append(f" - Max Capacity: {max_capacity}")
        vehicle_output.append(f" - Max Travel Distance: {max_distance}")
        vehicle_output.append(f" - Distance traveled: {vehicle_distance:.2f} units")
        vehicle_output.append(f" - Customers visited: {customer_count}")
        vehicle_output.append(" - Arrival details:")

        # Retrieve the arrival info for this vehicle
        for customer_idx, arrival_time, (start_time, end_time) in vehicle_arrival_info[idx]:
            x, y = customers[customer_idx]
            vehicle_output.append(f"   Customer at location ({x}, {y})")
            vehicle_output.append(f"   - Arrival time: {arrival_time:.2f}")
            vehicle_output.append(f"   - Time window: {start_time} to {end_time}")
    
    return vehicle_output  # Return the collected output

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

# Streamlit sidebar interface
st.sidebar.title("Vehicle Constraints Input")

# Number of vehicles input in the sidebar
num_vehicles = st.sidebar.number_input("Enter the number of vehicles:", min_value=1, max_value=30, value=1, step=1)

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
    save_to_json(vehicle_data)
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
    # Reading coordinates file from set coordinate json
    coordinates_file = 'set_coordinates.json'
    customers, demands, time_windows = read_coordinate_data(coordinates_file)

elif coordinate_option == "Generate Random Coordinates":
    num_customers = st.number_input("Number of customers:", min_value=1, value=5, step=1)
    if st.button("Generate Coordinates"):
        generate_all(num_customers)  # Use the generator.py function
        st.success("Random coordinates generated.")
        # Reading the generated json file
    coordinates_file = 'coordinates.json'
    customers, demands, time_windows = read_coordinate_data(coordinates_file)

elif coordinate_option == "Upload Custom Coordinates":
    uploaded_file = st.file_uploader("Upload a JSON file with coordinates", type=["json"])
    if uploaded_file is not None:
        try:
            customers, demands, time_windows = parse_json_coordinates(uploaded_file)
            st.write("Customers:", customers)
            st.write("Demands:", demands)
            st.write("Time Windows:", time_windows)
        except ValueError as e:
            st.error(f"Error: {e}")


# Run initial solution (greedy algorithm)
if st.button("Run Initial Solution (Greedy Algorithm)"):
    if customers:
        st.session_state['initial_routes'], st.session_state['dist_matrix'] = run_greedy_algorithm(customers, vehicle_capacities, demands, num_vehicles, vehicle_max_distances, time_windows, service_time)
        st.success("Initial Solution executed.")
        visualize_routes(customers, demands, st.session_state['initial_routes'], title="Greedy Algorithm Routes")
        st.write("Total Visited Customers (Initial Solution):", print_total_visited_customers(st.session_state['initial_routes']))
        st.write("Total Distance Traveled (Initial Solution):", calculate_total_distance(st.session_state['initial_routes'], st.session_state['dist_matrix']))
    else:
        st.error("No coordinates provided.")

# Show additional information for initial solution
if st.session_state['initial_routes'] is not None:
    if st.checkbox("Show Additional Information for Initial Solution"):
        st.write("Vehicle Information (Initial Solution):", print_vehicle_info(customers, st.session_state['dist_matrix'], st.session_state['initial_routes'], vehicle_capacities, vehicle_max_distances, time_windows, service_time))
# Dropdown menu for selecting optimization technique
st.title("Optimization Techniques")
optimization_option = st.selectbox(
    "Choose an optimization technique:",
    ("Simulated Annealing", "Ant Colony Optimization")
)

# Run the selected optimization technique
if st.button("Run Optimization"):
    if optimization_option == "Simulated Annealing":
        st.session_state['best_solution'], best_cost, st.session_state['initial_routes'], st.session_state['dist_matrix'], initial_routes_3opt = run_simulated_annealing(customers, num_vehicles, demands, vehicle_capacities, vehicle_max_distances, time_windows, service_time) 
        optimization_solution_container = st.container()
        with optimization_solution_container:
            st.write("Simulated Annealing Solution Visualization:")
            visualize_routes(customers, demands, st.session_state['best_solution'], title="Simulated Annealing Routes")
            st.write("Total Visited Customers (Optimized Solution):", print_total_visited_customers(st.session_state['best_solution']))
            st.write("Total Distance Traveled (Optimized Solution):", calculate_total_distance(st.session_state['best_solution'], st.session_state['dist_matrix']))
    elif optimization_option == "Ant Colony Optimization":
        st.write("Unavailable")
    else:
        st.write("No optimization selected.")

# Run the selected optimization technique
if st.button("Compare Results"):
    if optimization_option == "Simulated Annealing":
        st.session_state['best_solution'], best_cost, st.session_state['initial_routes'], st.session_state['dist_matrix'], initial_routes_3opt = run_simulated_annealing(customers, num_vehicles, demands, vehicle_capacities, vehicle_max_distances, time_windows, service_time) 
        optimization_solution_container = st.container()
        if st.session_state['initial_routes'] and st.session_state['best_solution']:
            st.write("Comparing Initial Solution and Optimized Solution")
            visualize_two_routes_side_by_side(customers, demands, st.session_state['initial_routes'], st.session_state['best_solution'])
            st.write("Total Visited Customers (Initial Solution):", print_total_visited_customers(st.session_state['initial_routes']))
            st.write("Total Distance Traveled (Initial Solution):", calculate_total_distance(st.session_state['initial_routes'], st.session_state['dist_matrix']))
            st.write("Total Visited Customers (Optimized Solution):", print_total_visited_customers(st.session_state['best_solution']))
            st.write("Total Distance Traveled (Optimized Solution):", calculate_total_distance(st.session_state['best_solution'], st.session_state['dist_matrix']))
    elif optimization_option == "Ant Colony Optimization":
        st.write("Unavailable")
    else:
        st.write("No optimization selected.")

# Display additional information for initial solution
if st.session_state['initial_routes'] is not None:
    if st.checkbox("Show Additional Information for Initial Solution", key=1):
        st.write("Vehicle Information (Initial Solution):", print_vehicle_info(customers, st.session_state['dist_matrix'], st.session_state['initial_routes'], vehicle_capacities, vehicle_max_distances, time_windows, service_time))

# Show additional information for optimized solution
if st.session_state['best_solution'] is not None:
    if st.checkbox("Show Additional Information for Optimized Solution", key=2):
        st.write("Vehicle Information (Optimized Solution):", print_vehicle_info(customers, st.session_state['dist_matrix'], st.session_state['initial_routes'], vehicle_capacities, vehicle_max_distances, time_windows, service_time))

