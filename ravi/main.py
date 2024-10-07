import streamlit as st
from optimization import MasterRoutingAgent
from coordinates import generate_random_coordinates
from plotting import plot_routes

# Main function to run the Streamlit app
def main():
    st.title("Optimized Delivery Routing in Australia")

    if 'coordinates' not in st.session_state:
        st.session_state['coordinates'] = None
    if 'best_routes' not in st.session_state:
        st.session_state['best_routes'] = None
    if 'parcels' not in st.session_state:
        st.session_state['parcels'] = None
    if 'vehicles' not in st.session_state:
        st.session_state['vehicles'] = None
    if 'optimization_method' not in st.session_state:
        st.session_state['optimization_method'] = None
    if 'optimization_performed' not in st.session_state:
        st.session_state['optimization_performed'] = False

    # Option to load delivery items from a text file or generate randomly
    st.header("Delivery Items Input")
    input_option = st.radio("Choose input method for delivery items:", ("Generate Randomly", "Upload from File"))

    if input_option == "Generate Randomly":
        # Number of Coordinates Input
        num_coords = st.number_input(
            "Enter number of delivery locations",
            min_value=1, max_value=20, value=5, step=1
        )
        # Button to generate random coordinates
        if st.button("Generate Random Coordinates"):
            coordinates = generate_random_coordinates(int(num_coords))
            st.session_state['coordinates'] = coordinates
            # Reset parcels and best_routes when new coordinates are generated
            st.session_state['parcels'] = None
            st.session_state['best_routes'] = None
            st.session_state['optimization_performed'] = False
            st.success(f"Generated {num_coords} random coordinates.")
    else:
        uploaded_file = st.file_uploader("Upload a text file with parcel locations (latitude,longitude):", type=["txt"])
        if uploaded_file is not None:
            content = uploaded_file.read().decode('utf-8')
            coordinates = []
            for line in content.strip().split('\n'):
                lat_str, lon_str = line.strip().split(',')
                coordinates.append({"latitude": float(lat_str), "longitude": float(lon_str)})
            st.session_state['coordinates'] = coordinates
            # Reset parcels and best_routes when new coordinates are uploaded
            st.session_state['parcels'] = None
            st.session_state['best_routes'] = None
            st.session_state['optimization_performed'] = False
            st.success(f"Loaded {len(coordinates)} coordinates from the file.")

    st.sidebar.title("Vehicle Configuration")
    num_vehicles = st.sidebar.number_input(
        "Number of Vehicles", min_value=1, max_value=5, value=2, step=1, key='num_vehicles'
    )
    vehicles = []
    for i in range(int(num_vehicles)):
        vehicle_id = f"vehicle_{i+1}"
        capacity = st.sidebar.number_input(
            f"Capacity of Vehicle {i+1}", min_value=1, value=5, step=1, key=f'capacity_{i}'
        )
        max_distance = st.sidebar.number_input(
            f"Max Distance for Vehicle {i+1}", min_value=1, value=1000, step=1, key=f'max_distance_{i}'
        )
        vehicles.append({
            "id": vehicle_id,
            "capacity": capacity,
            "max_distance": max_distance
        })

    # If vehicles configuration  changed, reset best_routes
    if 'vehicles' not in st.session_state or st.session_state['vehicles'] != vehicles:
        st.session_state['vehicles'] = vehicles
        st.session_state['best_routes'] = None  # Reset best_routes when vehicles change
        st.session_state['optimization_performed'] = False

    # Option - select optimization method
    st.header("Optimization Method")

    # Use st.form to group optimization inputs and button
    with st.form("optimization_form"):
        optimization_method = st.selectbox(
            "Choose optimization method:",
            ("Baseline Method", "OR-Tools Method")
        )
        submit_button = st.form_submit_button("Optimize Routes")

    # Update optimization method in session state
    if 'optimization_method' not in st.session_state or st.session_state['optimization_method'] != optimization_method:
        st.session_state['optimization_method'] = optimization_method
        st.session_state['best_routes'] = None  # Reset best_routes when optimization method changes
        st.session_state['optimization_performed'] = False

    # Initialize the MasterRoutingAgent
    master_agent = MasterRoutingAgent()

    # Set the parcel list based on generated coordinates
    if st.session_state['coordinates'] and st.session_state['parcels'] is None:
        coordinates = st.session_state['coordinates']
        parcels = [{
            "id": f"parcel_{i+1}",
            "location": (coord["latitude"], coord["longitude"])
        } for i, coord in enumerate(coordinates)]
        st.session_state['parcels'] = parcels
    else:
        parcels = st.session_state.get('parcels', [])

    if parcels:
        master_agent.set_parcel_list(parcels)
        master_agent.set_vehicle_list(vehicles)

        # Provide vehicle capacity and max distance constraints to the agent
        for vehicle in vehicles:
            master_agent.receive_capacity(vehicle['id'], vehicle['capacity'])
            master_agent.receive_max_distance(vehicle['id'], vehicle['max_distance'])

        # Run optimization if the form is submitted
        if submit_button:
            with st.spinner("Optimizing routes..."):
                if optimization_method == "Baseline Method":
                    best_routes = master_agent.baseline_optimize_routes()
                else:
                    best_routes = master_agent.ortools_optimize_routes()
            if best_routes:
                st.success("Optimized routes calculated successfully!")
                total_items_delivered = sum(len(route) for route in best_routes.values())
                total_distance = 0
                vehicle_distances = {}
                for vehicle_id, route in best_routes.items():
                    route_distance = master_agent.calculate_total_route_distance(route)
                    vehicle_distances[vehicle_id] = route_distance
                    total_distance += route_distance
                st.write(f"**Total items delivered:** {int(total_items_delivered)}")
                st.write(f"**Total travel distance:** {total_distance:.2f}")
                # Display total distance for each vehicle
                st.write("**Total distance for each vehicle:**")
                for vehicle_id, distance in vehicle_distances.items():
                    st.write(f" - {vehicle_id}: {distance:.2f}")
                st.session_state['best_routes'] = best_routes
                st.session_state['optimization_performed'] = True
            else:
                st.error("No feasible solution found with the given constraints.")
                st.session_state['best_routes'] = None
                st.session_state['optimization_performed'] = False

    if st.session_state.get('optimization_performed') and st.session_state.get('best_routes'):
        parcel_ids = {parcel['id'] for parcel in st.session_state['parcels']}
        valid = True
        for route in st.session_state['best_routes'].values():
            if not all(parcel_id in parcel_ids for parcel_id in route):
                valid = False
                break
        if valid:
            plot_routes(st.session_state['parcels'], st.session_state['best_routes'], st.session_state['vehicles'])
        else:
            st.warning("Best routes are outdated. Please optimize routes again.")
            st.session_state['best_routes'] = None
            st.session_state['optimization_performed'] = False

if __name__ == "__main__":
    main()