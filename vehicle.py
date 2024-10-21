
class Vehicle:
    def __init__(self, vehicle_id, capacity, max_distance):
        self.vehicle_id = vehicle_id
        self.capacity = capacity
        self.max_distance = max_distance
        self.route = []
        self.distance_traveled = 0

    def add_to_route(self, customer, distance):
        if self.capacity >= customer.demand and self.distance_traveled + distance <= self.max_distance:
            self.capacity -= customer.demand
            self.distance_traveled += distance
            self.route.append(customer)
            return True
        return False

    def __repr__(self):
        return f"Vehicle {self.vehicle_id}: Distance traveled = {self.distance_traveled}, Remaining capacity = {self.capacity}"
