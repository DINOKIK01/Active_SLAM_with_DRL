import numpy as np
from enum import IntEnum
from collections import deque
from pic4rl.utils.env_utils import *


class EdgeState(IntEnum):
    UNKNOWN = -1
    FREE = 0
    BLOCKED = 1


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.visited = False
        self.reentry_count = 0
        self.steps_since_entry = 0

        # edges: N, E, S, W
        self.edges = {
            "N": EdgeState.UNKNOWN,
            "E": EdgeState.UNKNOWN,
            "S": EdgeState.UNKNOWN,
            "W": EdgeState.UNKNOWN
        }

    def __repr__(self):
        return f"Node({self.x},{self.y}, visited={self.visited})"


class GraphMap:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.last_position = (-1, -1)

        self.nodes = [
            [Node(x, y) for y in range(height)]
            for x in range(width)
        ]

    # -----------------------------
    # Node Zugriff
    # -----------------------------
    def get_node(self, x, y):
        if not self.in_bounds(x, y):
            return None
        return self.nodes[x][y]

    def in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    # -----------------------------
    # Nachbarn
    # -----------------------------
    def get_neighbor(self, node, direction):
        dx, dy = self.direction_to_delta(direction)
        nx, ny = node.x + dx, node.y + dy
        return self.get_node(nx, ny)

    def direction_to_delta(self, direction):
        return {
            "N": (0, -1),   
            "E": (1, 0),    
            "S": (0, 1),    
            "W": (-1, 0)    
        }[direction]

    def opposite_direction(self, direction):
        return {
            "N": "S",
            "E": "W",
            "S": "N",
            "W": "E"
        }[direction]

    # -----------------------------
    # Edge setzen (beidseitig!)
    # -----------------------------
    def set_edge(self, node, direction, state):
        node.edges[direction] = state

        neighbor = self.get_neighbor(node, direction)
        if neighbor is not None:
            opp = self.opposite_direction(direction)
            neighbor.edges[opp] = state

    # -----------------------------
    # Besuch markieren
    # -----------------------------
    def visit_node(self, x, y):
        node = self.get_node(x, y)
        if node:
            node.visited = True
            node.visit_count += 1

    def update_visit_state(self, i, j):
        node = self.get_node(i, j)
        if node is None:
            return

        if not node.visited:
            node.visited = True
            node.reentry_count = 0
            node.steps_since_entry = 0
        else:
            node.reentry_count += 1
            node.steps_since_entry = 0            

    def increment_step_counter(self, i, j):
        node = self.get_node(i, j)

        if node is not None:
            node.steps_since_entry += 1

    # -----------------------------
    # Graph Features
    # -----------------------------
    def get_edge_status(self, robot_pose, radius=0):

        mapping = {
            EdgeState.BLOCKED: -1,
            EdgeState.UNKNOWN: 0,
            EdgeState.FREE: 1,
        }

        if radius == 0:
            i, j = get_coordinates(robot_pose)
            node = self.get_node(i, j)
            if node is None:
                return [0, 0, 0, 0]

            return [
                mapping[node.edges["N"]],
                mapping[node.edges["E"]],
                mapping[node.edges["S"]],
                mapping[node.edges["W"]],
            ]
        elif radius == 1:
            i, j = get_coordinates(robot_pose)
            node = self.get_node(i, j)
            if node is None:
                return [0 for _ in range(12)]
            node_n = self.get_neighbor(node, "N")
            node_e = self.get_neighbor(node, "E")
            node_s = self.get_neighbor(node, "S")
            node_w = self.get_neighbor(node, "W")

            edges_n = []
            edges_e = []
            edges_s = []
            edges_w = []

            if node_n is None:
                edges_n = [
                    0,
                    mapping[node.edges["N"]],
                    0,
                ]
            else:
                edges_n = [
                    mapping[node_n.edges["E"]],
                    mapping[node_n.edges["S"]],
                    mapping[node_n.edges["W"]],
                ]
            if node_e is None:
                edges_e = [
                    0,
                    mapping[node.edges["E"]],
                    0,
                ]
            else:
                edges_e = [
                    mapping[node_e.edges["N"]],
                    mapping[node_e.edges["S"]],
                    mapping[node_e.edges["W"]],
                ]
            if node_s is None:
                edges_s = [
                    0,
                    mapping[node.edges["S"]],
                    0,
                ]
            else:
                edges_s = [
                    mapping[node_s.edges["N"]],
                    mapping[node_s.edges["E"]],
                    mapping[node_s.edges["W"]],
                ]
            if node_w is None:
                edges_w = [
                    0,
                    mapping[node.edges["W"]],
                    0,
                ]
            else:
                edges_w = [
                    mapping[node_w.edges["N"]],
                    mapping[node_w.edges["E"]],
                    mapping[node_w.edges["S"]],
                ]


            return edges_n + edges_e + edges_s + edges_w
            
    
    def get_neighbor_status(self, robot_pose, radius=0):

        i, j = get_coordinates(robot_pose)
        node = self.get_node(i, j)
        
        if radius == 0:
            if node is None:
                return [0, 0, 0, 0]

            status = []

            for direction in ["N", "E", "S", "W"]:
                neighbor = self.get_neighbor(node, direction)

                if neighbor is not None and neighbor.visited:
                    status.append(1)
                else:
                    status.append(0)

            return status
        elif radius == 1:
            if node is None:
                return [0 for _ in range(9)]
            node_n = self.get_neighbor(node, "N")
            node_e = self.get_neighbor(node, "E")
            node_s = self.get_neighbor(node, "S")
            node_w = self.get_neighbor(node, "W")

            status = []
            neighbors_n = []
            neighbors_m = []
            neighbors_s = []

            if node_n is None:
                neighbors_n = [None, None, None,]
            else:
                neighbors_n = [
                    self.get_neighbor(node_n,"W"),
                    node_n,
                    self.get_neighbor(node_n,"E"),
                ]
            neighbors_m = [
                self.get_neighbor(node,"W"),
                node,
                self.get_neighbor(node,"E"),
            ]
            if node_s is None:
                neighbors_s = [None, None, None,]
            else:
                neighbors_s = [
                    self.get_neighbor(node_s,"W"),
                    node_s,
                    self.get_neighbor(node_s,"E"),
                ]

            neighbors = neighbors_n + neighbors_m + neighbors_s
            for neighbor in neighbors:
                if neighbor is not None and neighbor.visited:
                    status.append(1)
                else:
                    status.append(0)
            return status

    def get_frontier_vector(self, robot_pose):
        free_frontiers, unknown_unvisited_frontiers, unknown_visited_frontiers =  self.calculate_frontier_sets()
        target_node, distance, direction, frontier_type = self.choose_frontier_node(
            robot_pose,
            free_frontiers,
            unknown_unvisited_frontiers,
            unknown_visited_frontiers,
        )

        print(f"target node: {target_node} ; distance: {distance} ; direction: {direction} ; frontier type: {frontier_type}")

        if target_node is None:
            print("!!!THIS SHOULD NOT HAPPEN")
            return [0,0,0,0,0]
            #return [0,0,0,0,0,0,0,0]
        
        if distance == 0:   # if current node is frontier node
            return [0,0,0,0,0]
            #return [0,0,0,0,0,0,0,0]
        
        distance_normed = min(distance / 4.0, 1.0)

        match direction:
            case "N":
                direction_vector = [1,0,0,0]
            case "E":
                direction_vector = [0,1,0,0]
            case "S":
                direction_vector = [0,0,1,0]
            case "W":
                direction_vector = [0,0,0,1]
            case _:
                direction_vector = [0,0,0,0]

        match frontier_type:
            case 0:
                frontier_type_vector = [1,0,0]
            case 1:
                frontier_type_vector = [0,1,0]
            case 2:
                frontier_type_vector = [0,0,1]
            case _:
                frontier_type_vector = [0,0,0]

        return [distance_normed] + direction_vector #+ frontier_type_vector
    
    def choose_frontier_node(
        self,
        robot_pose,
        free_frontiers,
        unknown_unvisited_frontiers,
        unknown_visited_frontiers,
    ):
        i, j = get_coordinates(robot_pose)

        start_node = self.get_node(i, j)

        if start_node is None:
            return None, -1, None, None
        
        if start_node in free_frontiers:
            print("free_frontiers0")
            direction = self.get_frontier_direction(start_node, 0)
            return start_node, 0, direction, 0

        if start_node in unknown_unvisited_frontiers:
            print("free_frontiers1")
            direction = self.get_frontier_direction(start_node, 1)
            return start_node, 0, direction, 1

        if start_node in unknown_visited_frontiers:
            print("free_frontiers2")
            direction = self.get_frontier_direction(start_node, 2)
            return start_node, 0, direction, 2

        if free_frontiers:
            candidates = free_frontiers
            frontier_type = 0
        elif unknown_unvisited_frontiers:
            candidates = unknown_unvisited_frontiers
            frontier_type = 1
        elif unknown_visited_frontiers:
            candidates = unknown_visited_frontiers
            frontier_type = 2
        else:
            return None, -1, None, None

        best_node = None
        best_distance = float("inf")
        best_direction = None

        for candidate in candidates:

            distance, direction = self.shortest_path_info(
                start_node,
                candidate
            )

            if distance >= 0 and distance < best_distance:
                best_node = candidate
                best_distance = distance
                best_direction = direction

        return best_node, best_distance, best_direction, frontier_type
    
    def get_frontier_direction(self, node, frontier_type):
        for direction, edge_state in node.edges.items():

            neighbor = self.get_neighbor(node, direction)

            if neighbor is None:
                continue

            if frontier_type == 0:
                if (
                    edge_state == EdgeState.FREE
                    and not neighbor.visited
                ):
                    return direction

            elif frontier_type == 1:
                if (
                    edge_state == EdgeState.UNKNOWN
                    and not neighbor.visited
                ):
                    return direction

            elif frontier_type == 2:
                if (
                    edge_state == EdgeState.UNKNOWN
                    and neighbor.visited
                ):
                    return direction

        return None
    
    def calculate_frontier_sets(self):
        free_frontiers = []
        unknown_unvisited_frontiers = []
        unknown_visited_frontiers = []

        for row in self.nodes:
            for node in row:

                has_free_unvisited = False
                has_unknown_unvisited = False
                has_unknown_visited = False

                for direction, edge_state in node.edges.items():

                    neighbor = self.get_neighbor(node, direction)

                    if neighbor is None:
                        continue

                    if (
                        edge_state == EdgeState.FREE
                        and not neighbor.visited
                    ):
                        has_free_unvisited = True

                    elif (
                        edge_state == EdgeState.UNKNOWN
                        and not neighbor.visited
                    ):
                        has_unknown_unvisited = True

                    elif (
                        edge_state == EdgeState.UNKNOWN
                        and neighbor.visited
                    ):
                        has_unknown_visited = True

                if has_free_unvisited:
                    free_frontiers.append(node)

                if has_unknown_unvisited:
                    unknown_unvisited_frontiers.append(node)

                if has_unknown_visited:
                    unknown_visited_frontiers.append(node)

        return (
            free_frontiers,
            unknown_unvisited_frontiers,
            unknown_visited_frontiers,
        )
    
    def total_edges(self):
        return self.width * self.height * 4
    
    def discovered_edges(self):
        count = 0

        for row in self.nodes:
            for node in row:
                for state in node.edges.values():
                    if state != EdgeState.UNKNOWN:
                        count += 1

        return count
    
    def get_current_visit_count(self, robot_pose):
        i, j = get_coordinates(robot_pose)

        node = self.get_node(i, j)

        if node is None:
            return 0
        
        return node.visit_count

    def get_step_count(self, robot_pose):
        i, j = get_coordinates(robot_pose)
        node = self.get_node(i, j)

        if node is None:
            return 0

        return node.steps_since_entry
    
    def get_reentries(self, robot_pose):
        i, j = get_coordinates(robot_pose)
        node = self.get_node(i, j)

        if node is None:
            return 0

        return node.reentry_count
    

    def get_unvisited_neighbors(self, node):
        count = 0
        for d, state in node.edges.items():
            if state == EdgeState.FREE:
                n = self.get_neighbor(node, d)
                if n and not n.visited:
                    count += 1
        return count

    def get_free_neighbors(self, node):
        return sum(1 for s in node.edges.values() if s == EdgeState.FREE)

    def is_dead_end(self, node):
        return self.get_free_neighbors(node) <= 1

    def total_nodes(self):
        return self.width * self.height

    def visited_nodes(self):
        return sum(
            1 for row in self.nodes for n in row if n.visited
        )

    def remaining_nodes(self):
        return self.total_nodes() - self.visited_nodes()

    # -----------------------------
    # BFS: nächster unvisited Node
    # -----------------------------
    

    def shortest_path_info(self, start_node, goal_node):

        if start_node == goal_node:
            return 0, None

        visited = set()
        queue = deque()

        visited.add((start_node.x, start_node.y))

        for direction, state in start_node.edges.items():

            if state != EdgeState.FREE:
                continue

            neighbor = self.get_neighbor(start_node, direction)

            if neighbor is None:
                continue

            queue.append((neighbor, 1, direction))

        while queue:

            current, distance, first_direction = queue.popleft()

            key = (current.x, current.y)

            if key in visited:
                continue

            visited.add(key)

            if current == goal_node:
                return distance, first_direction

            for direction, state in current.edges.items():

                if state != EdgeState.FREE:
                    continue

                neighbor = self.get_neighbor(current, direction)

                if neighbor is not None:
                    queue.append(
                        (
                            neighbor,
                            distance + 1,
                            first_direction
                        )
                    )

        return -1, None

    # -----------------------------
    # Debug / Visualisierung
    # -----------------------------

    def print_graph(self, robot_pose):
        GREEN = "\033[92m"
        RED = "\033[91m"
        GRAY = "\033[90m"
        RESET = "\033[0m"
        BLUE = "\033[94m"

        i, j = get_coordinates(robot_pose)
        current = self.get_node(i, j)

        def node_str(node, current=None):
            if current and node.x == current.x and node.y == current.y:
                return f"{BLUE}●{RESET}"
            elif node.visited:
                return f"{GREEN}●{RESET}"
            else:
                return f"{GRAY}●{RESET}"

        def edge_str(state, horizontal=True):
            if state == EdgeState.FREE:
                color = GREEN
            elif state == EdgeState.BLOCKED:
                color = RED
            else:
                color = GRAY

            return f"{color}{'──' if horizontal else '│'}{RESET}"

        for y in range(self.height):

            # -----------------------------
            # 1. NORD KANTEN
            # -----------------------------
            line_north = ""
            for x in range(self.width):
                node = self.nodes[x][y]

                # Außenkante oben
                state = node.edges["N"]

                line_north += " " + edge_str(state, horizontal=True) + " "

            print(line_north)

            # -----------------------------
            # 2. WEST | NODE | OST
            # -----------------------------
            line_mid = ""
            for x in range(self.width):
                node = self.nodes[x][y]

                # WEST
                state_w = node.edges["W"]

                # OST
                state_e = node.edges["E"]

                line_mid += edge_str(state_w, horizontal=False)
                line_mid += node_str(node, current) + " "
                line_mid += edge_str(state_e, horizontal=False)

            print(line_mid)

            # -----------------------------
            # 3. SÜD KANTEN
            # -----------------------------
            line_south = ""
            for x in range(self.width):
                node = self.nodes[x][y]

                # Außenkante unten
                state = node.edges["S"]

                line_south += " " + edge_str(state, horizontal=True) + " "

            print(line_south)

def update_graph_from_lidar(
    graph,
    robot_pose,
    lidar_measurements,
    cell_size=3.0,
    offset=7.5,
    center_tol=0.5,
    angle_tol=0.35,  
    wall_threshold=2.0,
    free_threshold=2.9,
):
    """
    Update GraphMap basierend auf Lidar + Pose

    Args:
        graph: GraphMap
        robot_x, robot_y: Position in Metern
        robot_yaw: Orientierung in Radiant
        lidar_measurements: array mit 36 Werten
    """
    robot_x = robot_pose[0]
    robot_y = robot_pose[1]
    robot_yaw = robot_pose[2]

    # -----------------------------
    # 1. Zellindex bestimmen
    # -----------------------------
    i, j = get_coordinates(robot_pose)

    node = graph.get_node(i, j)
    if node is None:
        return
    
    # -----------------------------
    # 2. Node als besucht markieren
    # -----------------------------
    last_x, last_y = graph.last_position
    if i != last_x or j != last_y:
        graph.update_visit_state(i, j)
    graph.increment_step_counter(i, j)

    # -----------------------------
    # 3. Node Transition updaten
    # -----------------------------
    def update_node_transition(graph, current_node):
        last_x, last_y = graph.last_position

        # erster Schritt → nichts zu tun
        if last_x == -1 and last_y == -1:
            graph.last_position = (current_node.x, current_node.y)
            return

        # gleiche Position → kein Übergang
        if (last_x, last_y) == (current_node.x, current_node.y):
            return

        last_node = graph.get_node(last_x, last_y)
        if last_node is None:
            graph.last_position = (current_node.x, current_node.y)
            return

        dx = current_node.x - last_node.x
        dy = current_node.y - last_node.y

        def delta_to_direction(delta):
            return{
                (0, -1): "N",
                (1, 0): "E",
                (0, 1): "S",
                (-1, 0): "W"
            }[delta]

        direction = delta_to_direction((dx, dy))
        

        # Nur UNKNOWN → FREE setzen
        if last_node.edges[direction] == EdgeState.UNKNOWN:
            graph.set_edge(last_node, direction, EdgeState.FREE)

        # neue Position speichern
        graph.last_position = (current_node.x, current_node.y)

    update_node_transition(graph, node)

    # -----------------------------
    # 4. Zentrum prüfen
    # -----------------------------
    cx = i * cell_size - offset
    cy = offset - j * cell_size

    # -----------------------------
    # 5. Lidar in 4 Sektoren teilen
    # -----------------------------
    front, left, back, right = compute_lidar_groups(lidar_measurements)
    sectors = {
        "front": front,
        "left": left,
        "back": back,
        "right": right,
    }

    # robuste Distanz: Median
    sector_dist = {
        k: np.median(v) for k, v in sectors.items()
    }

    # Position Überprüfen
    dx = robot_x - cx
    dy = robot_y - cy

    at_center = abs(dx) < center_tol and abs(dy) < center_tol

    edge_trigger = 0.6
    axis_tol = 0.3

    at_edge_middle = (
        (abs(dx) > edge_trigger and abs(dy) < axis_tol) or
        (abs(dy) > edge_trigger and abs(dx) < axis_tol)
    )

    if at_edge_middle:
        update_edges_near_walls_with_orientation(
            graph,
            node,
            robot_x,
            robot_y,
            robot_yaw,
            cx,
            cy,
            sector_dist
        )
        return

    if not at_center:
        return
    
    # -----------------------------
    # 6. Winkel prüfen
    # -----------------------------

    # Welt-Richtungen
    directions = []

    if is_facing(robot_yaw, 0, angle_tol):  # Osten
        directions.append(("front", "E"))
        directions.append(("left", "N"))
        directions.append(("back", "W"))
        directions.append(("right", "S"))
    if is_facing(robot_yaw, np.pi / 2, angle_tol):  # Norden
        directions.append(("front", "N"))
        directions.append(("left", "W"))
        directions.append(("back", "S"))
        directions.append(("right", "E"))
    if is_facing(robot_yaw, np.pi, angle_tol):  # Westen
        directions.append(("front", "W"))
        directions.append(("left", "S"))
        directions.append(("back", "E"))
        directions.append(("right", "N"))
    if is_facing(robot_yaw, -np.pi / 2, angle_tol):  # Süden
        directions.append(("front", "S"))
        directions.append(("left", "E"))
        directions.append(("back", "N"))
        directions.append(("right", "W"))

    # -----------------------------
    # 7. Updates durchführen
    # -----------------------------
    def safe_update(node, direction, new_state):
        current = node.edges[direction]

        # niemals sichere Infos überschreiben
        if current == EdgeState.BLOCKED:
            return
        if current == EdgeState.FREE and new_state == EdgeState.BLOCKED:
            return

        if new_state != EdgeState.UNKNOWN:
            graph.set_edge(node, direction, new_state)

    for sector_name, graph_dir in directions:
        dist = sector_dist[sector_name]
        state = classify(dist, wall_threshold, free_threshold)

        safe_update(node, graph_dir, state)

def update_edges_near_walls_with_orientation(
    graph,
    node,
    robot_x,
    robot_y,
    robot_yaw,
    cx,
    cy,
    sector_dist,
    proximity_threshold=0.5,
    angle_tol=0.35,
    wall_threshold=0.8,
    free_threshold=2.9,
):
    dx = robot_x - cx
    dy = cy - robot_y

    def safe_update(direction, dist):
        state = classify(dist, wall_threshold, free_threshold)
        current = node.edges[direction]

        if current == EdgeState.BLOCKED:
            return
        if current == EdgeState.FREE and state == EdgeState.BLOCKED:
            return

        if state != EdgeState.UNKNOWN:
            graph.set_edge(node, direction, state)

    # -----------------------------
    # Mapping: Robot → Welt
    # -----------------------------
    # Ergebnis: welcher Sektor entspricht welcher Welt-Richtung?
    world_mapping = {}

    if is_facing(robot_yaw, 0, angle_tol):  # Osten
        world_mapping = {
            "E": "front",
            "N": "left",
            "W": "back",
            "S": "right"
        }

    elif is_facing(robot_yaw, np.pi / 2, angle_tol):  # Norden
        world_mapping = {
            "N": "front",
            "W": "left",
            "S": "back",
            "E": "right"
        }

    elif is_facing(robot_yaw, np.pi, angle_tol):  # Westen
        world_mapping = {
            "W": "front",
            "S": "left",
            "E": "back",
            "N": "right"
        }

    elif is_facing(robot_yaw, -np.pi / 2, angle_tol):  # Süden
        world_mapping = {
            "S": "front",
            "E": "left",
            "N": "back",
            "W": "right"
        }

    else:
        # NICHT sauber ausgerichtet → KEIN UPDATE
        return

    # -----------------------------
    # EDGE UPDATES (nur wenn nah an Wand)
    # -----------------------------

    # EAST / WEST
    if dx > proximity_threshold:
        sector = world_mapping["E"]
        safe_update("E", sector_dist[sector])

    elif dx < -proximity_threshold:
        sector = world_mapping["W"]
        safe_update("W", sector_dist[sector])

    # NORTH / SOUTH
    if dy > proximity_threshold:
        sector = world_mapping["S"]
        safe_update("S", sector_dist[sector])

    elif dy < -proximity_threshold:
        sector = world_mapping["N"]
        safe_update("N", sector_dist[sector])

def angle_diff(a, b):
    return np.arctan2(np.sin(a - b), np.cos(a - b))

def is_facing(yaw, target, angle_tol):
    return abs(angle_diff(yaw, target)) < angle_tol

def classify(dist, wall_threshold, free_threshold):
    if dist < wall_threshold:
        return EdgeState.BLOCKED
    elif dist > free_threshold:
        return EdgeState.FREE
    else:
        return EdgeState.UNKNOWN

def step_count_sigmoid(step_count):
    return 1 / (1 + np.exp(-0.9 * (step_count - 20)))

def reentry_sigmoid(reentries):
    return 1 / (1 + np.exp(-1.5 * (reentries - 2.5)))