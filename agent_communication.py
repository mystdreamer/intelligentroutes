from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import json
import uuid

@dataclass
class ACLMessage:
    sender: str
    receiver: str
    performative: str  # request, inform, propose, accept-proposal, reject-proposal, refuse
    conversation_id: str
    content: Dict[str, Any]
    in_reply_to: Optional[str] = None
    reply_with: Optional[str] = None

    @staticmethod
    def create_conversation_id() -> str:
        return str(uuid.uuid4())

    def to_json(self) -> str:
        return json.dumps({
            "sender": self.sender,
            "receiver": self.receiver,
            "performative": self.performative,
            "conversation_id": self.conversation_id,
            "content": self.content,
            "in_reply_to": self.in_reply_to,
            "reply_with": self.reply_with
        })

    @staticmethod
    def from_json(json_str: str) -> 'ACLMessage':
        data = json.loads(json_str)
        return ACLMessage(**data)

class DeliveryAgent:
    def __init__(self, agent_id: str, vehicle_id: int):
        self.agent_id = agent_id
        self.vehicle_id = vehicle_id
        self.current_route: Optional[List[int]] = None
        self.capacity: Optional[int] = None
        self.max_distance: Optional[float] = None
        self.current_load = 0
        self.distance_traveled = 0
        self.current_time = 0
        self.current_location = 0

    def handle_request(self, message: ACLMessage) -> Optional[ACLMessage]:
        if message.performative == "request":
            if message.content["action"] == "evaluate_route":
                return self._evaluate_route(message)
            elif message.content["action"] == "accept_route":
                return self._accept_route(message)
            elif message.content["action"] == "update_state":
                return self._update_state(message)
        return None

    def _evaluate_route(self, message: ACLMessage) -> ACLMessage:
        proposed_route = message.content.get("route", [])
        constraints = message.content.get("constraints", {})
        is_feasible = self._check_route_feasibility(proposed_route, constraints)
        
        return ACLMessage(
            sender=self.agent_id,
            receiver=message.sender,
            performative="inform" if is_feasible else "refuse",
            conversation_id=message.conversation_id,
            in_reply_to=message.reply_with,
            content={
                "route_feasible": is_feasible,
                "vehicle_id": self.vehicle_id,
                "proposed_route": proposed_route
            }
        )

    def _accept_route(self, message: ACLMessage) -> ACLMessage:
        self.current_route = message.content.get("route", [])
        return ACLMessage(
            sender=self.agent_id,
            receiver=message.sender,
            performative="inform",
            conversation_id=message.conversation_id,
            in_reply_to=message.reply_with,
            content={
                "route_accepted": True,
                "vehicle_id": self.vehicle_id,
                "assigned_route": self.current_route
            }
        )

    def _update_state(self, message: ACLMessage) -> ACLMessage:
        state_updates = message.content.get("state_updates", {})
        self.current_load = state_updates.get("current_load", self.current_load)
        self.distance_traveled = state_updates.get("distance_traveled", self.distance_traveled)
        self.current_time = state_updates.get("current_time", self.current_time)
        self.current_location = state_updates.get("current_location", self.current_location)
        
        return ACLMessage(
            sender=self.agent_id,
            receiver=message.sender,
            performative="inform",
            conversation_id=message.conversation_id,
            content={
                "state_updated": True,
                "vehicle_id": self.vehicle_id
            }
        )

    def _check_route_feasibility(self, route: List[int], constraints: Dict) -> bool:
        service_time = constraints.get("service_time", 0.2)
        capacity = constraints.get("capacity", self.capacity)
        max_distance = constraints.get("max_distance", self.max_distance)
        demands = constraints.get("demands", [])
        time_windows = constraints.get("time_windows", [])
        dist_matrix = constraints.get("dist_matrix", [])

        if not route or len(route) < 2:
            return False

        current_load = 0
        current_time = 0
        total_distance = 0
        current = route[0]

        for next_customer in route[1:]:
            if next_customer >= len(dist_matrix):
                return False

            distance = dist_matrix[current][next_customer]
            total_distance += distance

            if total_distance > max_distance:
                return False

            if next_customer != 0:  # Skip depot
                current_load += demands[next_customer]
                if current_load > capacity:
                    return False

                arrival_time = current_time + distance
                start_time, end_time = time_windows[next_customer]

                if arrival_time > end_time:
                    return False

                current_time = max(arrival_time, start_time) + service_time

            current = next_customer

        return True

class MasterRoutingAgent:
    def __init__(self):
        self.conversation_id: Optional[str] = None
        self.delivery_agents: Dict[str, DeliveryAgent] = {}

    def register_delivery_agent(self, da: DeliveryAgent):
        self.delivery_agents[da.agent_id] = da

    def propose_routes(self, routes: List[List[int]], constraints: Dict) -> Dict[str, List[int]]:
        self.conversation_id = ACLMessage.create_conversation_id()
        assignments = {}

        for da_id, route in zip(self.delivery_agents.keys(), routes):
            request = ACLMessage(
                sender="MRA",
                receiver=da_id,
                performative="request",
                conversation_id=self.conversation_id,
                reply_with=str(uuid.uuid4()),
                content={
                    "action": "evaluate_route",
                    "route": route,
                    "constraints": constraints
                }
            )

            response = self.delivery_agents[da_id].handle_request(request)
            
            if response and response.performative == "inform":
                assign_request = ACLMessage(
                    sender="MRA",
                    receiver=da_id,
                    performative="request",
                    conversation_id=self.conversation_id,
                    reply_with=str(uuid.uuid4()),
                    content={
                        "action": "accept_route",
                        "route": route
                    }
                )
                
                confirm = self.delivery_agents[da_id].handle_request(assign_request)
                if confirm and confirm.performative == "inform":
                    assignments[da_id] = route

        return assignments

    def update_agent_state(self, da_id: str, state_updates: Dict):
        if da_id in self.delivery_agents:
            message = ACLMessage(
                sender="MRA",
                receiver=da_id,
                performative="request",
                conversation_id=self.conversation_id or ACLMessage.create_conversation_id(),
                content={
                    "action": "update_state",
                    "state_updates": state_updates
                }
            )
            return self.delivery_agents[da_id].handle_request(message)
        return None