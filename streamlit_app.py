import streamlit as st
import folium
from streamlit_folium import st_folium
import json
import os
import subprocess

# Streamlit App Layout
st.title("Random Coordinates Map around Australia")

# Button to trigger generation of coordinates
if st.button("Generate Coordinates"):
    # Run the external script to generate the coordinates
    result = subprocess.run(['python', 'generate_coordinates.py'], capture_output=True, text=True)

    # Display the output of the script execution
    if result.returncode == 0:
        st.success("Coordinates successfully generated!")
    else:
        st.error("Error generating coordinates!")
        st.write(result.stderr)

# Check if the JSON file exists
if os.path.exists("coordinates.json"):
    # Load coordinates from JSON file
    with open("coordinates.json", "r") as file:
        data = json.load(file)
        num_coords = data["num_coords"]
        coordinates = data["coordinates"]

    # Initialize Folium map centered on Australia
    australia_lat = -25.00
    australia_lon = 133.00
    m = folium.Map(location=[australia_lat, australia_lon], zoom_start=4)

    # Add the coordinates to the map
    for coord in coordinates:
        lat = coord["latitude"]
        lon = coord["longitude"]
        
        # Use the color field if it exists, otherwise default color
        color = coord.get("color", "blue")  # Default color for markers without specific color

        # Differentiate start and end points based on the 'point' key
        popup_text = f'({lat:.5f}, {lon:.5f})'
        if coord.get("point") == "start":
            popup_text += " - Start"
        elif coord.get("point") == "end":
            popup_text += " - End"
        
        # Add colored marker
        folium.Marker(
            location=[lat, lon],
            popup=popup_text,
            icon=folium.Icon(color=color)
        ).add_to(m)

    # Display the number of coordinates
    st.write(f"Number of coordinates displayed: {num_coords}")

    # Display the map in Streamlit
    st_data = st_folium(m, width=700, height=500)
else:
    st.write("No coordinates found. Please generate the coordinates using the button above.")
