import random

def generate_service_times(n_customers, min_time=0, max_time=1):
    return [random.uniform(min_time, max_time) for _ in range(n_customers)]


print(generate_service_times(4))