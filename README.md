# Vehicle Routing Optimization Project

## Project Overview
This project is a Vehicle Routing Problem (VRP) solution using optimization algorithms to generate and refine delivery routes. The project uses a **Greedy Algorithm** to create initial routes and **Simulated Annealing** with **3-Opt Local Search** to improve them, maximizing item delivery while minimizing travel distance. The project also includes a Streamlit-based GUI to visualize routes and interact with the algorithm settings.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Setup and Installation](#setup-and-installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

---

## Features
- **Greedy Algorithm** for initial route generation.
- **Simulated Annealing Optimization** for refining routes.
- **3-Opt Local Search** for improved path adjustments.
- **Time Window Constraints** for realistic scheduling.
- **Streamlit GUI** for easy parameter adjustment and route visualization.
- **Interactive Map** for viewing delivery locations and vehicle routes.

---

## Setup and Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8 or later
- Git (for cloning the repository)

### 1. Clone the Repository
First, clone the repository to your local machine:
```bash
git clone https://github.com/mystdreamer/intelligentroutes.git
cd vehicle-routing-optimization
```

### 2. Environment Setup
Note: If using Visual studio code on mac, ensure your terminal is a command interpreter shell like bash and not zsh.

Create and activate virtual environment:
```bash
python -m venv venv
```

### 3. Install Required Packages
Install all required packages using pip:
```bash
pip install -r requirements.txt
```

Requirements File: The requirements.txt file should contain the following libraries:
```bash
streamlit
folium
numpy
matplotlib
random
json
math
```

---

## Configuration

### JSON Data Files
The project relies on JSON files for coordinate data and vehicle parameters. You may need the following files:

- set_coordinates.json: Contains preset customer locations, demands, time windows, and customer group information.
- vehicle_data.json: Specifies each vehicle’s capacity and maximum travel distance.

---

## Usage
To start the Streamlit application:
```bash
streamlit run main.py
```

### Main Interface Options
1.Coordinate Input:

- Use Preset Coordinates: Uses the set_coordinates.json file.
- Generate Random Coordinates: Generates random customer locations based on user input.
- Upload Custom Coordinates: Upload your own JSON file with custom coordinates.

2.Generate Initial Solution:

- Run the Greedy Algorithm to quickly create initial delivery routes.
- The initial routes are displayed on the map.

3.Optimize Solution:

- Choose the Simulated Annealing optimization option.
- The optimized routes are displayed below the initial routes for comparison.

### Viewing Additional Information
- You can select options to display:

- Total customers and items delivered.
- Detailed vehicle information, including distance traveled and item capacity used.

---

## Technologies Used
- Python: Core programming language.
- Streamlit: For building the graphical user interface.
- Folium: For interactive map visualizations.
- Numpy: For efficient array and matrix calculations.
- Matplotlib: For plotting customer locations and routes.
- Math: For performing calculations like distances.
- Random: For generating random test data.
- Json: For handling JSON input and output.

---