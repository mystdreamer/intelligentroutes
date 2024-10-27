import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def visualize_routes(customers, demands, routes, title):
    customer_coords = np.array(customers)
    demands_array = np.array(demands)

    # Different colors for different vehicle routes
    colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan', 'k', 'b', 'g', 'r', 'c', 'm', 'y'] # Add more colors as needed
    # Plotting the customers and depot
    plt.figure(figsize=(1, 10))
    plt.scatter(customer_coords[:, 0], customer_coords[:, 1], s=demands_array * 100, c='tab:gray', alpha=0.5, label='Customers')
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
    plt.show()

def visualize_routes_st(customers, customer_groups, demands, routes, title):
    customer_coords = np.array(customers)
    demands_array = np.array(demands)

    # Different colors for different vehicle routes
    colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan', 'k', 'b', 'g', 'r', 'c', 'm', 'y'] # Add more colors as needed
    coord_colors = {
        'morning': 'blue',
        'afternoon': 'yellow',
        'evening': 'purple',
        'ungrouped': 'magenta'
    }
    plt.figure(figsize=(16, 12))

    # Plotting the depot and demand ring
    plt.scatter(customer_coords[0, 0], customer_coords[0, 1], s=300, c='r', label='Depot', edgecolor='black')
    plt.scatter(customer_coords[:, 0], customer_coords[:, 1], s=demands_array * 100, c='tab:gray', alpha=0.5, label='Item Amount')
    

    # To track which group has already been added to the legend
    group_plotted = set()
    # Plot customers by group
    for i, customer in enumerate(customers[1:], start=1):
        group = customer_groups.get(str(i), 'ungrouped')  # Default to 'ungrouped' if key is missing
        # Add to the legend only the first time this group is encountered
        if group not in group_plotted:
            plt.scatter(customer[0], customer[1], s=100, c=coord_colors[group], label=group.capitalize(), alpha=0.6)
            group_plotted.add(group)
        else:
            plt.scatter(customer[0], customer[1], s=100, c=coord_colors[group], alpha=0.6)

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

def visualize_two_routes_side_by_side(customers, customer_groups, demands, routes1, routes2, title1="Initial Solution", title2="Optimized Solution"):
    # Create columns for side-by-side plots
    col1, col2 = st.columns(2)
    customer_coords = np.array(customers)
    demands_array = np.array(demands)

    # Different colors for different vehicle routes
    colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan', 'k', 'b', 'g', 'r', 'c', 'm', 'y'] # Add more colors as needed
    coord_colors = {
        'morning': 'blue',
        'afternoon': 'yellow',
        'evening': 'purple',
        'ungrouped': 'magenta'
    }
    # First plot in the first column
    with col1:
        # Plotting the customers and depot
        plt.figure(figsize=(8, 6))

        # Plotting the depot and demand ring
        plt.scatter(customer_coords[0, 0], customer_coords[0, 1], s=300, c='r', label='Depot', edgecolor='black')
        plt.scatter(customer_coords[:, 0], customer_coords[:, 1], s=demands_array * 100, c='tab:gray', alpha=0.5, label='Item Amount')

        # To track which group has already been added to the legend
        group_plotted = set()
        # Plot customers by group
        for i, customer in enumerate(customers[1:], start=1):
            group = customer_groups.get(str(i), 'ungrouped')  # Default to 'ungrouped' if key is missing
            # Add to the legend only the first time this group is encountered
            if group not in group_plotted:
                plt.scatter(customer[0], customer[1], s=100, c=coord_colors[group], label=group.capitalize(), alpha=0.6)
                group_plotted.add(group)
            else:
                plt.scatter(customer[0], customer[1], s=100, c=coord_colors[group], alpha=0.6)


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
        # Plotting the depot and demand ring
        plt.scatter(customer_coords[0, 0], customer_coords[0, 1], s=300, c='r', label='Depot', edgecolor='black')
        plt.scatter(customer_coords[:, 0], customer_coords[:, 1], s=demands_array * 100, c='tab:gray', alpha=0.5, label='Item Amount')

        # To track which group has already been added to the legend
        group_plotted = set()
        # Plot customers by group
        for i, customer in enumerate(customers[1:], start=1):
            group = customer_groups.get(str(i), 'ungrouped')  # Default to 'ungrouped' if key is missing
            # Add to the legend only the first time this group is encountered
            if group not in group_plotted:
                plt.scatter(customer[0], customer[1], s=100, c=coord_colors[group], label=group.capitalize(), alpha=0.6)
                group_plotted.add(group)
            else:
                plt.scatter(customer[0], customer[1], s=100, c=coord_colors[group], alpha=0.6)

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
