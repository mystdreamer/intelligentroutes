import json

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

        return customers, demands, time_windows

    except json.JSONDecodeError:
        raise ValueError("Error decoding JSON file. Please make sure it's a valid JSON.")
    except Exception as e:
        raise ValueError(f"An error occurred: {e}")

