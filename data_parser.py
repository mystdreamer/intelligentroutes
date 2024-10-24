import json
import streamlit as st

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
        customer_groups = data.get("customer_groups", {i: 'ungrouped' for i in range(1, len(customers))})  # Default group

        # Returning the structured data
        return customers, demands, time_windows, customer_groups

    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error parsing JSON from '{filename}'.")
        return None
    except ValueError as ve:
        print(f"Error in data format: {ve}")
        return None

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
    
# Function to save the vehicle data to a JSON file
def save_vehicle_data(vehicle_data, filename='vehicle_data.json'):
    with open(filename, 'w') as f:
        json.dump(vehicle_data, f, indent=4)
    st.sidebar.success(f'Data saved to {filename}')

def parse_json_coordinates(file):
    """
    A parser to parse in uploaded coordinates from user.
    - Will still work if there's only coordinates, no demands and no time window.
    - The parser will assume that the demand is 1 for all locations.
    - The parser will assume that there's no time window if not specified.
    
    Args:
    file: The uploaded JSON file containing customer information.
    
    Returns:
    customers: A list of customer coordinates.
    demands: A list of demands for each customer.
    time_windows: A list of time windows for each customer.
    """
    
    try:
        # Load the JSON data
        data = json.load(file)

        # Extract customer coordinates
        customers = data.get("customers", [])
        
        if not customers:
            raise ValueError("No customer coordinates provided in the JSON file.")

        # Extract demands, assume demand is 1 for all locations if not provided
        if "demands" in data:
            demands = data["demands"]
        else:
            demands = [1] * len(customers)  # Default demand of 1 for all customers

        # Extract time windows, assume no time windows if not provided
        if "time_windows" in data:
            time_windows = data["time_windows"]
        else:
            time_windows = [(0, float('inf'))] * len(customers)  # No time window constraints

        # Extract customer groups, assume no groups if not provided
        if "customer_groups" in data:
            customer_groups = data["customer_groups"]  # Should be a dictionary
        else:
            customer_groups = {i: 'ungrouped' for i in range(1, len(customers))}  # Default group as 'ungrouped'


        return customers, demands, time_windows, customer_groups

    except json.JSONDecodeError:
        raise ValueError("Error decoding JSON file. Please make sure it's a valid JSON.")
    except Exception as e:
        raise ValueError(f"An error occurred: {e}")

