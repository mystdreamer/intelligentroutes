import numpy as np
import matplotlib.pyplot as plt
import random
import json
import folium

# Define the coordinates of the bounding box
sw_corner = [-37.8501, 145.0225]  # Southwest corner
ne_corner = [-37.7927, 145.1119]  # Northeast corner

# Generate a fixed depot location
def generate_depot():
    return [-37.8220, 145.0669]  # fixed location for the depot

def visualize_routes(customers, demands):
    customer_coords = np.array(customers)
    demands_array = np.array(demands)

    # Plotting the customers and depot
    plt.figure(figsize=(8, 6))
    plt.scatter(customer_coords[:, 0], customer_coords[:, 1], s=demands_array * 100, c='b', alpha=0.5, label='Customers')
    plt.scatter(customer_coords[0, 0], customer_coords[0, 1], s=300, c='r', label='Depot', edgecolor='black')

    # Adding labels for customers
    for i, txt in enumerate(demands):
        plt.annotate(f"C{i} (demand: {txt})", (customer_coords[i, 0] + 0.1, customer_coords[i, 1] + 0.1))

    # Set plot titles and labels
    plt.title("Customer and Depot Locations")
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to visualize groups on Matplotlib with different colors
def visualize_groups_matplotlib(customers, customer_groups):
    colors = {
        'morning': 'yellow',
        'afternoon': 'blue',
        'evening': 'green',
        'night': 'purple'
    }

    plt.figure(figsize=(8, 6))
    
    # Plot depot (red)
    plt.scatter(customers[0][0], customers[0][1], s=300, c='red', label='Depot', edgecolor='black')

    # To track which group has already been added to the legend
    group_plotted = set()

    # Plot customers by group
    for i, customer in enumerate(customers[1:], start=1):
        group = customer_groups[i]
        # Add to the legend only the first time this group is encountered
        if group not in group_plotted:
            plt.scatter(customer[0], customer[1], s=100, c=colors[group], label=group.capitalize(), alpha=0.6)
            group_plotted.add(group)
        else:
            plt.scatter(customer[0], customer[1], s=100, c=colors[group], alpha=0.6)

    plt.title("Customer Groups by Time Window")
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.legend()  # Show the legend with all groups
    plt.grid(True)
    plt.show()

# Function to visualize customers and depot on a Folium map
def visualize_on_folium(customers, demands, map_name="cvrp_map.html"):
    # Create a Folium map centered around the depot (customer[0])
    m = folium.Map(location=customers[0], zoom_start=13)

    # Add a marker for the depot
    folium.Marker(customers[0], popup="Depot", icon=folium.Icon(color='red')).add_to(m)

    # Add markers for each customer
    for i, (location, demand) in enumerate(zip(customers[1:], demands[1:]), start=1):
        folium.CircleMarker(
            location=location,
            radius=5 + demand,  # The size of the marker depends on the demand
            popup=f"Customer {i} (Demand: {demand})",
            color='blue',
            fill=True,
            fill_opacity=0.7
        ).add_to(m)

    # Save the map to an HTML file
    m.save(map_name)
    print(f"Map saved as {map_name}")

# Generate random customers, demands, and time windows
def generate_customers(n_customers, bottom_left_corner, top_right_corner):
    customers = []
    for _ in range(n_customers):
        lat = random.uniform(bottom_left_corner[0], top_right_corner[0])
        lon = random.uniform(bottom_left_corner[1], top_right_corner[1])
        customers.append([lat, lon])
    return customers

def generate_demands(n_customers, total_demand):
    """
    Distribute the total demand randomly across all customers, ensuring each customer has at least 1 demand.
    The first customer (depot) always has a demand of 0.
    """
    # Step 1: Assign a minimum demand of 1 to all customers (excluding the depot)
    demands = [0] + [1] * (n_customers - 1)  # Depot gets 0, every other customer gets at least 1

    # Step 2: Calculate remaining demand after giving 1 to each customer
    remaining_demand = total_demand - (n_customers - 1)

    # Step 3: Randomly distribute the remaining demand among the customers (excluding the depot)
    while remaining_demand > 0:
        i = random.randint(1, n_customers - 1)  # Choose a random customer (excluding the depot)
        demands[i] += 1
        remaining_demand -= 1

    return demands

# # Time window generation with grouping
# def generate_time_windows(n_customers):
#     time_windows = [(0, 999)]  # Depot time window

#     morning_group = (5, 12)
#     afternoon_group = (12, 17)
#     evening_group = (17, 22)
#     night_group = (22, 5)

#     for _ in range(n_customers - 1):
#         group_choice = random.choice(['morning', 'afternoon', 'evening', 'night'])

#         if group_choice == 'morning':
#             start_time = random.randint(morning_group[0], morning_group[1] - 1)
#             end_time = random.randint(start_time + 1, morning_group[1])
#         elif group_choice == 'afternoon':
#             start_time = random.randint(afternoon_group[0], afternoon_group[1] - 1)
#             end_time = random.randint(start_time + 1, afternoon_group[1])
#         elif group_choice == 'evening':
#             start_time = random.randint(evening_group[0], evening_group[1] - 1)
#             end_time = random.randint(start_time + 1, evening_group[1])
#         elif group_choice == 'night':
#             start_time = random.randint(night_group[0], 24) % 24  # Handle wrapping of time
#             if start_time < 22:
#                 end_time = random.randint(start_time + 1, 24) % 24  # Wrap-around at midnight
#             else:
#                 end_time = random.randint(0, 5)

#         time_windows.append((start_time, end_time))

#     return time_windows

# Time window generation with predefined groups
def assign_time_windows(n_customers):
    time_windows = [(0, 999)]  # Depot time window
    customer_groups = {}

    # Define time groups
    groups = {
        'morning': (5, 12),
        'afternoon': (12, 17),
        'evening': (17, 22),
        'night': (22, 5)
    }

    group_names = list(groups.keys())

    # Randomly assign each customer to a time group
    for i in range(1, n_customers):
        group_choice = random.choice(group_names)
        customer_groups[i] = group_choice

        if group_choice == 'morning':
            time_windows.append(groups['morning'])
        elif group_choice == 'afternoon':
            time_windows.append(groups['afternoon'])
        elif group_choice == 'evening':
            time_windows.append(groups['evening'])
        elif group_choice == 'night':
            time_windows.append(groups['night'])

    return time_windows, customer_groups

# Save the generated data to a JSON file
def save_to_json(customers, demands, time_windows, customer_groups, file_name="coordinates.json"):
    data = {
        "customers": customers,
        "demands": demands,
        "time_windows": time_windows,
        "customer_groups": customer_groups,
    }
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to {file_name}")

def generate_all(n_customers, total_demand):
    # Generate a set of customer locations (including the depot at index 0)
    depot = generate_depot()
    customers = [depot] + generate_customers(n_customers - 1, sw_corner, ne_corner)

    # Generate demands for the customers (0 for the depot)
    demands = generate_demands(n_customers, total_demand)

    # Assign time window groups for the customers
    time_windows, customer_groups = assign_time_windows(n_customers)

    # Save data to JSON
    save_to_json(customers, demands, time_windows, customer_groups)  
    
    return customers, demands, time_windows, customer_groups



# # Generate a set of customer locations (including the depot at index 0)
# n_customers = 150  # Including depot as customer 0
# total_demands = 200
# customers, demands, time_windows, customer_groups = generate_all(n_customers, total_demands)

# # Visualize groups on Matplotlib
# visualize_groups_matplotlib(customers, customer_groups)

# # Visualize routes
# visualize_routes(customers, demands)



# # Example usage with generated data
# visualize_on_folium(customers, demands)

# print("Customers (Latitude, Longitude):")
# print(customers)
# print("Demands:")
# print(demands)
# print("Time Windows:")
# print(time_windows)
