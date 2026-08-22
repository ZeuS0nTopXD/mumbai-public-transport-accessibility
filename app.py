from flask import Flask, render_template
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import os
import time
import heapq
from collections import deque

app = Flask(__name__)


# ============================================================
# 1. HEURISTIC / DISTANCE FUNCTION
# ============================================================

def euclidean_distance(node1, node2, nodes):
    lat1, lon1 = nodes[node1]
    lat2, lon2 = nodes[node2]

    # Approximate conversion of latitude/longitude to km
    lat_diff = (lat1 - lat2) * 111
    lon_diff = (lon1 - lon2) * 111 * np.cos(np.radians((lat1 + lat2) / 2))

    return np.sqrt(lat_diff ** 2 + lon_diff ** 2)


# ============================================================
# 2. BFS
# ============================================================

def bfs(graph, start, goals, nodes):
    queue = deque([(start, [start], 0)])
    visited = {start}
    expanded = 0

    while queue:
        current, path, cost = queue.popleft()
        expanded += 1

        if current in goals:
            return path, cost, expanded

        for neighbour, edge_cost in graph.get(current, []):
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(
                    (neighbour, path + [neighbour], cost + edge_cost)
                )

    return None, float('inf'), expanded


# ============================================================
# 3. DFS
# ============================================================

def dfs(graph, start, goals, nodes):
    stack = [(start, [start], 0)]
    visited = set()
    expanded = 0

    while stack:
        current, path, cost = stack.pop()

        if current in visited:
            continue

        visited.add(current)
        expanded += 1

        if current in goals:
            return path, cost, expanded

        for neighbour, edge_cost in reversed(graph.get(current, [])):
            if neighbour not in visited:
                stack.append(
                    (neighbour, path + [neighbour], cost + edge_cost)
                )

    return None, float('inf'), expanded


# ============================================================
# 4. UNIFORM COST SEARCH
# ============================================================

def uniform_cost_search(graph, start, goals, nodes):
    priority_queue = [(0, start, [start])]
    visited_cost = {}
    expanded = 0

    while priority_queue:
        cost, current, path = heapq.heappop(priority_queue)

        if current in visited_cost and visited_cost[current] <= cost:
            continue

        visited_cost[current] = cost
        expanded += 1

        if current in goals:
            return path, cost, expanded

        for neighbour, edge_cost in graph.get(current, []):
            new_cost = cost + edge_cost

            if neighbour not in visited_cost or new_cost < visited_cost[neighbour]:
                heapq.heappush(
                    priority_queue,
                    (new_cost, neighbour, path + [neighbour])
                )

    return None, float('inf'), expanded


# ============================================================
# 5. GREEDY BEST FIRST SEARCH
# ============================================================

def greedy_best_first_search(graph, start, goals, nodes):
    # Heuristic = distance to closest transport station
    def heuristic(node):
        if not goals:
            return float('inf')

        return min(
            euclidean_distance(node, goal, nodes)
            for goal in goals
        )

    priority_queue = [(heuristic(start), start, [start], 0)]
    visited = set()
    expanded = 0

    while priority_queue:
        _, current, path, cost = heapq.heappop(priority_queue)

        if current in visited:
            continue

        visited.add(current)
        expanded += 1

        if current in goals:
            return path, cost, expanded

        for neighbour, edge_cost in graph.get(current, []):
            if neighbour not in visited:
                heapq.heappush(
                    priority_queue,
                    (
                        heuristic(neighbour),
                        neighbour,
                        path + [neighbour],
                        cost + edge_cost
                    )
                )

    return None, float('inf'), expanded


# ============================================================
# 6. A* SEARCH
# ============================================================

def a_star_search(graph, start, goals, nodes):

    def heuristic(node):
        if not goals:
            return float('inf')

        return min(
            euclidean_distance(node, goal, nodes)
            for goal in goals
        )

    priority_queue = [
        (heuristic(start), 0, start, [start])
    ]

    best_cost = {start: 0}
    expanded = 0

    while priority_queue:

        f_cost, current_cost, current, path = heapq.heappop(
            priority_queue
        )

        if current_cost > best_cost.get(current, float('inf')):
            continue

        expanded += 1

        if current in goals:
            return path, current_cost, expanded

        for neighbour, edge_cost in graph.get(current, []):

            new_cost = current_cost + edge_cost

            if new_cost < best_cost.get(neighbour, float('inf')):

                best_cost[neighbour] = new_cost

                f_value = new_cost + heuristic(neighbour)

                heapq.heappush(
                    priority_queue,
                    (
                        f_value,
                        new_cost,
                        neighbour,
                        path + [neighbour]
                    )
                )

    return None, float('inf'), expanded


# ============================================================
# 7. BUILD SEARCH GRAPH
# ============================================================

def build_graph(df_wards, df_transit):

    nodes = {}
    graph = {}

    # --------------------------------------------------------
    # Add ward nodes
    # --------------------------------------------------------

    for _, row in df_wards.iterrows():

        ward = str(row['Ward_Alphabet'])

        nodes[f"W_{ward}"] = (
            row['latitude'],
            row['longitude']
        )

        graph[f"W_{ward}"] = []

    # --------------------------------------------------------
    # Add transport nodes
    # --------------------------------------------------------

    for index, row in df_transit.iterrows():

        transport_id = f"T_{index}"

        nodes[transport_id] = (
            row['latitude'],
            row['longitude']
        )

        graph[transport_id] = []

    ward_nodes = [
        node for node in nodes if node.startswith("W_")
    ]

    transport_nodes = [
        node for node in nodes if node.startswith("T_")
    ]

    # --------------------------------------------------------
    # Connect wards to nearby wards
    # --------------------------------------------------------

    ward_coordinates = np.array(
        [nodes[node] for node in ward_nodes]
    )

    tree = cKDTree(ward_coordinates)

    for i, ward_node in enumerate(ward_nodes):

        distances, indices = tree.query(
            ward_coordinates[i],
            k=min(4, len(ward_nodes))
        )

        for distance, neighbour_index in zip(
            np.atleast_1d(distances),
            np.atleast_1d(indices)
        ):

            neighbour = ward_nodes[neighbour_index]

            if neighbour == ward_node:
                continue

            distance_km = distance * 111

            graph[ward_node].append(
                (neighbour, distance_km)
            )

    # --------------------------------------------------------
    # Connect wards to nearby transport stations
    # --------------------------------------------------------

    transport_coordinates = np.array(
        [nodes[node] for node in transport_nodes]
    )

    transport_tree = cKDTree(transport_coordinates)

    for ward_node in ward_nodes:

        ward_coord = np.array(nodes[ward_node])

        # Connect each ward to its nearest 5 transport points
        k = min(5, len(transport_nodes))

        distances, indices = transport_tree.query(
            ward_coord,
            k=k
        )

        for distance, transport_index in zip(
            np.atleast_1d(distances),
            np.atleast_1d(indices)
        ):

            transport_node = transport_nodes[transport_index]

            distance_km = distance * 111

            graph[ward_node].append(
                (transport_node, distance_km)
            )

            graph[transport_node].append(
                (ward_node, distance_km)
            )

    return graph, nodes, transport_nodes


# ============================================================
# 8. RUN ALL FIVE ALGORITHMS
# ============================================================

def compare_algorithms(graph, nodes, transport_nodes, ward_nodes):

    algorithms = {
        "BFS": bfs,
        "DFS": dfs,
        "Uniform Cost Search": uniform_cost_search,
        "Greedy Best First Search": greedy_best_first_search,
        "A* Search": a_star_search
    }

    results = []

    # Run every algorithm for every ward
    for algorithm_name, algorithm in algorithms.items():

        total_cost = 0
        total_expanded = 0
        total_time = 0
        successful_searches = 0

        for ward in ward_nodes:

            start_time = time.perf_counter()

            path, cost, expanded = algorithm(
                graph,
                ward,
                set(transport_nodes),
                nodes
            )

            end_time = time.perf_counter()

            execution_time = end_time - start_time

            if path is not None:

                total_cost += cost
                total_expanded += expanded
                successful_searches += 1

            total_time += execution_time

        if successful_searches > 0:

            avg_cost = total_cost / successful_searches
            avg_expanded = total_expanded / successful_searches
            avg_time = total_time / successful_searches

        else:

            avg_cost = float('inf')
            avg_expanded = float('inf')
            avg_time = float('inf')

        results.append({
            "algorithm": algorithm_name,
            "average_path_cost": avg_cost,
            "average_nodes_expanded": avg_expanded,
            "average_execution_time": avg_time,
            "successful_searches": successful_searches
        })

    results_df = pd.DataFrame(results)

    return results_df


# ============================================================
# 9. SELECT BEST TWO ALGORITHMS
# ============================================================

def select_best_two(results_df):

    # Lower is better for all three metrics

    for column in [
        "average_path_cost",
        "average_nodes_expanded",
        "average_execution_time"
    ]:

        minimum = results_df[column].min()
        maximum = results_df[column].max()

        if maximum == minimum:

            results_df[column + "_score"] = 1

        else:

            # Normalized score:
            # 1 = best
            # 0 = worst

            results_df[column + "_score"] = (
                (maximum - results_df[column]) /
                (maximum - minimum)
            )

    # Give highest importance to path cost
    results_df["overall_score"] = (
        results_df["average_path_cost_score"] * 0.50
        +
        results_df["average_nodes_expanded_score"] * 0.30
        +
        results_df["average_execution_time_score"] * 0.20
    )

    results_df = results_df.sort_values(
        by="overall_score",
        ascending=False
    )

    best_two = results_df.head(2)["algorithm"].tolist()

    return results_df, best_two


# ============================================================
# 10. FINAL CONNECTIVITY ANALYSIS
# ============================================================

def calculate_accessibility(
    df_wards,
    df_transit,
    graph,
    nodes,
    transport_nodes,
    best_algorithms
):

    # --------------------------------------------------------
    # Calculate direct nearest-transit distance
    # --------------------------------------------------------

    transit_coords = df_transit[
        ['latitude', 'longitude']
    ].values

    ward_coords = df_wards[
        ['latitude', 'longitude']
    ].values

    tree = cKDTree(transit_coords)

    distances, _ = tree.query(ward_coords)

    df_wards['distance_km'] = distances * 111

    # --------------------------------------------------------
    # Accessibility gap score
    # --------------------------------------------------------

    df_wards['Accessibility_Gap_Score'] = (
        df_wards['distance_km']
        *
        df_wards['TOT_P_DEN']
    )

    # --------------------------------------------------------
    # Poor access areas
    # --------------------------------------------------------

    df_poor = df_wards[
        df_wards['distance_km'] > 0.30
    ].sort_values(
        by='distance_km',
        ascending=False
    )

    poor_access_list = []

    for _, row in df_poor.iterrows():

        poor_access_list.append({
            'ward': row['Ward_Alphabet'],
            'name': row['Ward_Names'],
            'distance': round(
                row['distance_km'], 2
            )
        })

    # --------------------------------------------------------
    # Run BEST TWO algorithms for final analysis
    # --------------------------------------------------------

    algorithm_functions = {
        "BFS": bfs,
        "DFS": dfs,
        "Uniform Cost Search": uniform_cost_search,
        "Greedy Best First Search": greedy_best_first_search,
        "A* Search": a_star_search
    }

    algorithm_scores = {}

    for algorithm_name in best_algorithms:

        algorithm = algorithm_functions[
            algorithm_name
        ]

        ward_results = []

        for _, row in df_wards.iterrows():

            ward_node = f"W_{row['Ward_Alphabet']}"

            if ward_node not in graph:
                continue

            path, cost, expanded = algorithm(
                graph,
                ward_node,
                set(transport_nodes),
                nodes
            )

            if path is not None:

                ward_results.append({
                    'ward': row['Ward_Alphabet'],
                    'name': row['Ward_Names'],
                    'path_cost': cost,
                    'nodes_expanded': expanded
                })

        algorithm_scores[algorithm_name] = pd.DataFrame(
            ward_results
        )

    # --------------------------------------------------------
    # Combine results from the BEST TWO algorithms
    # --------------------------------------------------------

    df_final = df_wards.copy()

    for algorithm_name in best_algorithms:

        result = algorithm_scores[algorithm_name]

        df_final = df_final.merge(
            result[
                ['ward', 'path_cost', 'nodes_expanded']
            ].rename(
                columns={
                    'path_cost':
                        f'{algorithm_name}_cost',
                    'nodes_expanded':
                        f'{algorithm_name}_expanded'
                }
            ),
            left_on='Ward_Alphabet',
            right_on='ward',
            how='left'
        )

        if 'ward' in df_final.columns:
            df_final.drop(
                columns=['ward'],
                inplace=True
            )

    # --------------------------------------------------------
    # Final priority score
    # --------------------------------------------------------

    cost_columns = [
        f'{algorithm}_cost'
        for algorithm in best_algorithms
    ]

    df_final['Search_Average_Cost'] = (
        df_final[cost_columns]
        .mean(axis=1)
    )

    # Combine geographical gap + search result
    df_final['Final_Priority_Score'] = (
        df_final['Accessibility_Gap_Score']
        *
        (1 + df_final['Search_Average_Cost'])
    )

    # --------------------------------------------------------
    # TOP 5 areas requiring better connectivity
    # --------------------------------------------------------

    df_priority = df_final.sort_values(
        by='Final_Priority_Score',
        ascending=False
    ).head(5)

    prioritized_list = []

    for _, row in df_priority.iterrows():

        prioritized_list.append({
            'ward': row['Ward_Alphabet'],
            'name': row['Ward_Names'],
            'distance': round(
                row['distance_km'], 2
            ),
            'score': round(
                row['Final_Priority_Score'], 2
            )
        })

    return poor_access_list, prioritized_list


# ============================================================
# 11. MAIN PROCESSING FUNCTION
# ============================================================

def process_transit_data():

    base_path = os.path.dirname(
        os.path.abspath(__file__)
    )

    wards_path = os.path.join(
        base_path,
        'data',
        'ward_level_collated.csv'
    )

    transit_path = os.path.join(
        base_path,
        'data',
        'unified_mumbai_transport.csv'
    )

    # --------------------------------------------------------
    # Load REAL-WORLD datasets
    # --------------------------------------------------------

    df_wards = pd.read_csv(wards_path)
    df_transit = pd.read_csv(transit_path)

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    df_transit = df_transit.rename(
        columns={
            'Latitude': 'latitude',
            'Longitude': 'longitude'
        }
    )

    df_transit = df_transit.dropna(
        subset=['latitude', 'longitude']
    )

    df_wards = df_wards.dropna(
        subset=[
            'Ward_Alphabet',
            'TOT_P_DEN'
        ]
    )

    # --------------------------------------------------------
    # Mumbai ward coordinates
    # --------------------------------------------------------

    mumbai_ward_coords = {

        'A': (18.9220, 72.8347),
        'B': (18.9548, 72.8377),
        'C': (18.9449, 72.8259),
        'D': (18.9647, 72.8130),
        'E': (18.9696, 72.8423),

        'F/N': (19.0238, 72.8550),
        'F/S': (19.0014, 72.8452),

        'G/N': (19.0330, 72.8475),
        'G/S': (19.0103, 72.8262),

        'H/E': (19.0700, 72.8468),
        'H/W': (19.0657, 72.8310),

        'K/E': (19.1136, 72.8697),
        'K/W': (19.1197, 72.8464),

        'L': (19.0759, 72.8877),

        'M/E': (19.0473, 72.9158),
        'M/W': (19.0596, 72.8958),

        'N': (19.1417, 72.9331),

        'P/N': (19.1874, 72.8484),
        'P/S': (19.1551, 72.8464),

        'R/C': (19.2215, 72.8556),
        'R/N': (19.2804, 72.8597),
        'R/S': (19.2094, 72.8126),

        'S': (19.1306, 72.9375),
        'T': (19.1735, 72.9495)
    }

    df_wards['latitude'] = (
        df_wards['Ward_Alphabet']
        .map(
            lambda x:
            mumbai_ward_coords.get(
                x,
                (19.0760, 72.8777)
            )[0]
        )
    )

    df_wards['longitude'] = (
        df_wards['Ward_Alphabet']
        .map(
            lambda x:
            mumbai_ward_coords.get(
                x,
                (19.0760, 72.8777)
            )[1]
        )
    )

    # --------------------------------------------------------
    # BUILD SEARCH GRAPH
    # --------------------------------------------------------

    graph, nodes, transport_nodes = build_graph(
        df_wards,
        df_transit
    )

    ward_nodes = [
        node for node in nodes
        if node.startswith("W_")
    ]

    # --------------------------------------------------------
    # RUN ALL 5 ALGORITHMS
    # --------------------------------------------------------

    results_df = compare_algorithms(
        graph,
        nodes,
        transport_nodes,
        ward_nodes
    )

    # --------------------------------------------------------
    # SELECT BEST TWO
    # --------------------------------------------------------

    results_df, best_two = select_best_two(
        results_df
    )

    # --------------------------------------------------------
    # FINAL ANALYSIS USING BEST TWO
    # --------------------------------------------------------

    poor_access, prioritized = calculate_accessibility(
        df_wards,
        df_transit,
        graph,
        nodes,
        transport_nodes,
        best_two
    )

    # Convert algorithm comparison into dictionaries
    algorithm_results = []

    for _, row in results_df.iterrows():

        algorithm_results.append({
            'algorithm': row['algorithm'],
            'path_cost': round(
                row['average_path_cost'], 3
            ),
            'nodes': round(
                row['average_nodes_expanded'], 2
            ),
            'time': round(
                row['average_execution_time'], 6
            ),
            'score': round(
                row['overall_score'], 3
            )
        })

    return (
        poor_access,
        prioritized,
        algorithm_results,
        best_two
    )


# ============================================================
# 12. RUN PROCESSING ON STARTUP (CACHED)
# ============================================================

print("Processing transit data on startup...")
(
    cached_poor_access,
    cached_prioritized,
    cached_algorithm_results,
    cached_best_two
) = process_transit_data()
print("Data processing complete! Starting web server.")


# ============================================================
# 13. FLASK ROUTE
# ============================================================

@app.route('/')
def home():
    # Instantly return the pre-calculated results
    return render_template(
        'index.html',
        poor_access=cached_poor_access,
        prioritized=cached_prioritized,
        algorithm_results=cached_algorithm_results,
        best_two=cached_best_two
    )


# ============================================================
# 14. RUN APPLICATION
# ============================================================

if __name__ == '__main__':
    app.run(debug=True)
