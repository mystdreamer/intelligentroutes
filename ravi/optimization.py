# optimization.py

import itertools
from turtle import st
from ortools.linear_solver import pywraplp  # For optimization with OR-Tools
from coordinates import calculate_distance

# Class responsible for optimizing routes for delivery vehicles
class MasterRoutingAgent:
    def __init__(self):
        self.capacity_constraints = {}  # Vehicle capacity constraints
        self.max_distances = {}  # Vehicle maximum distance constraints
        self.parcels = {}  # Parcel locations
        self.vehicles = []  # List of vehicles

    def receive_capacity(self, vehicle_id, capacity):
        self.capacity_constraints[vehicle_id] = capacity

    def receive_max_distance(self, vehicle_id, max_distance):
        self.max_distances[vehicle_id] = max_distance

    def set_parcel_list(self, parcels):
        self.parcels = {parcel['id']: parcel['location'] for parcel in parcels}

    def set_vehicle_list(self, vehicles):
        self.vehicles = vehicles

    def calculate_total_route_distance(self, route):
        if not route:
            return 0
        total_distance = 0
        warehouse_location = (-25.0, 133.0)  # Central warehouse location in Australia
        last_location = warehouse_location
        for parcel_id in route:
            parcel_location = self.parcels[parcel_id]
            total_distance += calculate_distance(last_location, parcel_location)
            last_location = parcel_location
        # Add distance from last parcel back to warehouse
        total_distance += calculate_distance(last_location, warehouse_location)
        return total_distance

    def baseline_optimize_routes(self):
        best_solution = None
        min_total_distance = float('inf')
        max_items_delivered = 0
        all_parcels = list(self.parcels.keys())
        num_parcels = len(all_parcels)
        num_vehicles = len(self.vehicles)

        # Get vehicle IDs
        vehicle_ids = [vehicle['id'] for vehicle in self.vehicles]
        capacities = [self.capacity_constraints[vid] for vid in vehicle_ids]
        max_total_capacity = sum(capacities)

        # Generate all possible subsets of parcels
        for k in range(min(num_parcels, max_total_capacity), 0, -1):
            parcel_subsets = itertools.combinations(all_parcels, k)
            for parcel_subset in parcel_subsets:
                # Now assign parcels in the subset to vehicles
                # Generate all possible assignments of parcels to vehicles
                total_assignments = len(vehicle_ids) ** k
                max_combinations = 100000  # limitting max combinations
                if total_assignments > max_combinations:
                    continue  # Skip this number of parcels due to computational limits
                for assignments in itertools.product(vehicle_ids, repeat=k):
                    routes = {vehicle_id: [] for vehicle_id in vehicle_ids}
                    for parcel_id, vehicle_id in zip(parcel_subset, assignments):
                        routes[vehicle_id].append(parcel_id)
                    feasible = True
                    total_distance = 0
                    # Check each vehicle's route for feasibility
                    for vehicle_id in vehicle_ids:
                        route = routes[vehicle_id]
                        vehicle = next(v for v in self.vehicles if v['id'] == vehicle_id)
                        # Check capacity constraint
                        if len(route) > self.capacity_constraints[vehicle_id]:
                            feasible = False
                            break
                        # Check distance constraint
                        route_distance = self.calculate_total_route_distance(route)
                        if route_distance > self.max_distances[vehicle_id]:
                            feasible = False
                            break
                        total_distance += route_distance
                    # Update the best solution if a better feasible solution is found
                    total_items_delivered = k
                    if feasible:
                        if total_items_delivered > max_items_delivered:
                            best_solution = routes
                            min_total_distance = total_distance
                            max_items_delivered = total_items_delivered
                        elif total_items_delivered == max_items_delivered and total_distance < min_total_distance:
                            best_solution = routes
                            min_total_distance = total_distance
                            max_items_delivered = total_items_delivered
            # If we found a solution delivering k items, break early
            if best_solution is not None:
                break
        return best_solution

    def ortools_optimize_routes(self):

        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            st.error("Solver not found. Please ensure that OR-Tools is installed correctly.")
            return None

        parcels = list(self.parcels.keys())
        vehicles = [v['id'] for v in self.vehicles]
        num_parcels = len(parcels)
        num_vehicles = len(vehicles)

        # Decision variables: x[p][v] = 1 if parcel p is assigned to vehicle v
        x = {}
        for p in parcels:
            for v in vehicles:
                x[p, v] = solver.IntVar(0, 1, f'x_{p}_{v}')

        # Objective: Maximize the total number of parcels delivered
        total_parcels_delivered = solver.Sum([x[p, v] for p in parcels for v in vehicles])
        solver.Maximize(total_parcels_delivered)

        # Constraint: Each parcel can be assigned to at most one vehicle
        for p in parcels:
            solver.Add(solver.Sum([x[p, v] for v in vehicles]) <= 1)

        # Capacity constraints for each vehicle
        for v in vehicles:
            solver.Add(solver.Sum([x[p, v] for p in parcels]) <= self.capacity_constraints[v])

        # Distance constraints for each vehicle
        for v in vehicles:
            total_distance_expr = solver.Sum([
                x[p, v] * self.calculate_distance_contribution(v, p)
                for p in parcels
            ])
            solver.Add(total_distance_expr <= self.max_distances[v])

        # First solve to maximize the number of parcels delivered
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            max_parcels_delivered = total_parcels_delivered.solution_value()

            # Now, add a constraint to fix the total parcels delivered
            solver.Add(total_parcels_delivered == max_parcels_delivered)

            # Create a new objective to minimize total travel distance
            total_distance = solver.Sum([
                x[p, v] * self.calculate_distance_contribution(v, p)
                for p in parcels for v in vehicles
            ])
            solver.Minimize(total_distance)

            # Solve again
            status = solver.Solve()

            if status == pywraplp.Solver.OPTIMAL:
                best_solution = {v: [] for v in vehicles}
                for p in parcels:
                    for v in vehicles:
                        if x[p, v].solution_value() > 0.5:
                            best_solution[v].append(p)
                return best_solution
            else:
                st.error("No feasible solution found when minimizing total distance.")
                return None
        else:
            st.error("No feasible solution found with the given constraints.")
            return None

    def calculate_distance_contribution(self, vehicle_id, parcel_id):

        warehouse_location = (-25.0, 133.0)
        parcel_location = self.parcels[parcel_id]
        distance_to_parcel = calculate_distance(warehouse_location, parcel_location)
        distance_back = calculate_distance(parcel_location, warehouse_location)
        return distance_to_parcel + distance_back