import folium
from streamlit_folium import st_folium



# Function to plot the routes on a Folium map
def plot_routes(parcels, best_routes, vehicles):
    australia_lat = -25.0
    australia_lon = 133.0
    m = folium.Map(location=[australia_lat, australia_lon], zoom_start=4)

    colors = [
        "red", "green", "blue", "purple", "orange", "darkred", "lightred",
        "beige", "darkblue", "darkgreen", "cadetblue", "darkpurple", "pink"
    ]

    # Plot the warehouse location
    folium.Marker(
        location=[australia_lat, australia_lon],
        popup="Warehouse",
        icon=folium.Icon(color="black", icon="home", prefix='fa')
    ).add_to(m)

    # Create a mapping from parcel IDs to their locations
    parcel_locations = {parcel['id']: parcel['location'] for parcel in parcels}

    # Plot each vehicle's route
    for i, (vehicle_id, route) in enumerate(best_routes.items()):
        route_color = colors[i % len(colors)]
        last_location = (australia_lat, australia_lon)  # Start from warehouse
        vehicle = next(v for v in vehicles if v['id'] == vehicle_id)
        for parcel_id in route:
            parcel_location = parcel_locations[parcel_id]
            # Add a marker for the parcel
            folium.Marker(
                location=parcel_location,
                popup=f"{parcel_id} (Vehicle {vehicle_id})",
                icon=folium.Icon(color=route_color, icon="truck", prefix='fa')
            ).add_to(m)
            # Draw a line from the last location to the current parcel
            folium.PolyLine([last_location, parcel_location], color=route_color).add_to(m)
            last_location = parcel_location
        # Draw a dashed line back to the warehouse
        folium.PolyLine(
            [last_location, (australia_lat, australia_lon)],
            color=route_color,
            dash_array="5, 5"
        ).add_to(m)
    # Display the map in Streamlit
    st_folium(m, width=700, height=500)
