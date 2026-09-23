from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import os
import time
import heapq
import random
import pickle
import hashlib
import math
from collections import deque, OrderedDict
from difflib import SequenceMatcher

app = Flask(__name__)

# ============================================================
# OFFICIAL DATA / CACHE CONFIGURATION
# ============================================================
CACHE_VERSION = 23
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_CACHE_PATH = os.path.join(DATA_DIR, 'processed_official_cache.pkl')
ROUTE_CACHE_MAX_SIZE = 1024
_route_cache = OrderedDict()
BENCHMARK_ROUTE_COUNT = 300
BENCHMARK_SCORE_DESCRIPTION = (
    'Score = success rate × (55% route optimality + 30% search '
    'efficiency + 15% runtime efficiency)'
)
ROUTE_REGRESSION_DESCRIPTION = (
    'Regression-style route-cost metrics compare each algorithm\'s measured '
    'scheduled route cost with the best measured route cost on the same routes.'
)

OFFICIAL_FILES = {
    'wards': 'bmc_ward_population.csv',
    'stations': 'mumbai_local_train_stations_ALL_OFFICIAL.csv',
    'services': 'mumbai_local_train_services_ALL_OFFICIAL.csv',
    'stop_times': 'mumbai_local_train_stop_times_ALL_OFFICIAL.csv',
    'best_routes': 'best_official_verified_routes.csv',
    'best_summary': 'best_official_network_summary.csv',
}

def _file_signature(path):
    stat = os.stat(path)
    return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}

def _source_signature():
    try:
        with open(__file__, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None

def _cache_metadata(paths):
    return {
        'cache_version': CACHE_VERSION,
        'source_hash': _source_signature(),
        'files': {key: _file_signature(path) for key, path in paths.items()}
    }

def _load_processed_cache(paths):
    if not os.path.exists(PROCESSED_CACHE_PATH):
        return None
    try:
        with open(PROCESSED_CACHE_PATH, 'rb') as f:
            payload = pickle.load(f)
        if payload.get('metadata') != _cache_metadata(paths):
            return None
        data = payload.get('data')
        if not isinstance(data, tuple) or len(data) != 12:
            return None
        print('Loaded official processed data from cache.')
        return data
    except Exception as exc:
        print(f'Could not load official cache ({exc}); rebuilding...')
        return None

def _save_processed_cache(data, paths):
    os.makedirs(DATA_DIR, exist_ok=True)
    payload = {'metadata': _cache_metadata(paths), 'data': data}
    temp_path = PROCESSED_CACHE_PATH + '.tmp'
    try:
        with open(temp_path, 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_path, PROCESSED_CACHE_PATH)
        print('Saved official processed data cache.')
    except Exception as exc:
        print(f'Could not save processed cache: {exc}')
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass

# ============================================================
# 1. BASIC DISTANCE HELPERS
# ============================================================

def euclidean_distance(node1, node2, nodes):
    lat1, lon1 = nodes[node1]
    lat2, lon2 = nodes[node2]

    lat_diff = (lat1 - lat2) * 111.0
    mean_lat = (lat1 + lat2) * 0.5
    lon_diff = (
        (lon1 - lon2)
        * 111.0
        * math.cos(math.radians(mean_lat))
    )

    return math.hypot(lat_diff, lon_diff)


def point_distance(a, b):
    lat1, lon1 = a
    lat2, lon2 = b

    lat_diff = (lat1 - lat2) * 111.0
    mean_lat = (lat1 + lat2) * 0.5
    lon_diff = (
        (lon1 - lon2)
        * 111.0
        * math.cos(math.radians(mean_lat))
    )

    return math.hypot(lat_diff, lon_diff)


# ============================================================
# 2. SEARCH ALGORITHMS
# ============================================================

def _reconstruct_path(parent, goal):
    path = []
    current = goal

    while current is not None:
        path.append(current)
        current = parent.get(current)

    path.reverse()
    return path


def _path_cost_from_parents(path, parent_cost):
    return sum(
        parent_cost.get(node, 0.0)
        for node in path[1:]
    )



def bfs(graph, start, goals, nodes):
    """Breadth-first search with O(V) parent storage instead of copying paths."""
    queue = deque([start])
    visited = {start}
    parent = {start: None}
    parent_cost = {}
    expanded = 0

    while queue:
        current = queue.popleft()
        expanded += 1

        if current in goals:
            path = _reconstruct_path(parent, current)
            cost = _path_cost_from_parents(path, parent_cost)
            return path, cost, expanded

        for neighbour, edge_cost in graph.get(current, ()):
            if neighbour in visited:
                continue

            visited.add(neighbour)
            parent[neighbour] = current
            parent_cost[neighbour] = edge_cost
            queue.append(neighbour)

    return None, float('inf'), expanded



def dfs(graph, start, goals, nodes):
    """Depth-first search using parent pointers rather than path copies."""
    stack = [start]
    visited = set()
    parent = {start: None}
    parent_cost = {}
    expanded = 0

    while stack:
        current = stack.pop()

        if current in visited:
            continue

        visited.add(current)
        expanded += 1

        if current in goals:
            path = _reconstruct_path(parent, current)
            cost = _path_cost_from_parents(path, parent_cost)
            return path, cost, expanded

        neighbours = graph.get(current, ())
        for neighbour, edge_cost in reversed(neighbours):
            if neighbour not in visited:
                parent[neighbour] = current
                parent_cost[neighbour] = edge_cost
                stack.append(neighbour)

    return None, float('inf'), expanded



def uniform_cost_search(graph, start, goals, nodes):
    """UCS with parent pointers; avoids O(path length) list copies."""
    counter = 0
    priority_queue = [(0.0, counter, start)]
    best_cost = {start: 0.0}
    parent = {start: None}
    parent_cost = {}
    expanded = 0

    while priority_queue:
        cost, _, current = heapq.heappop(priority_queue)

        if cost != best_cost.get(current):
            continue

        expanded += 1

        if current in goals:
            path = _reconstruct_path(parent, current)
            return path, cost, expanded

        for neighbour, edge_cost in graph.get(current, ()):
            new_cost = cost + edge_cost
            if new_cost < best_cost.get(neighbour, float('inf')):
                best_cost[neighbour] = new_cost
                parent[neighbour] = current
                parent_cost[neighbour] = edge_cost
                counter += 1
                heapq.heappush(
                    priority_queue,
                    (new_cost, counter, neighbour)
                )

    return None, float('inf'), expanded



def _topology_heuristic_factory(graph, goals):
    """Build a data-derived heuristic without geographic coordinates.

    h(n) = minimum number of remaining edges * minimum observed edge cost.
    Every edge costs at least the minimum edge cost, so this is admissible
    for A* and uses only the official timetable graph.
    """
    goals = set(goals)
    reverse = {node: [] for node in graph}
    min_edge_cost = float('inf')

    for node, edges in graph.items():
        for neighbour, cost in edges:
            reverse.setdefault(neighbour, []).append(node)
            if cost > 0:
                min_edge_cost = min(min_edge_cost, float(cost))

    if not np.isfinite(min_edge_cost):
        min_edge_cost = 0.0

    hop_distance = {}
    queue = deque()

    for goal in goals:
        hop_distance[goal] = 0
        queue.append(goal)

    while queue:
        current = queue.popleft()
        for previous in reverse.get(current, ()):
            if previous not in hop_distance:
                hop_distance[previous] = hop_distance[current] + 1
                queue.append(previous)

    def heuristic(node):
        hops = hop_distance.get(node)
        if hops is None:
            return 0.0
        return float(hops * min_edge_cost)

    return heuristic


def greedy_best_first_search(graph, start, goals, nodes):
    """Greedy search using a topology-derived remaining-time estimate."""
    heuristic = _topology_heuristic_factory(graph, goals)
    counter = 0
    priority_queue = [(heuristic(start), counter, start, 0.0)]
    parent = {start: None}
    parent_cost = {}
    visited = set()
    expanded = 0

    while priority_queue:
        _, _, current, cost = heapq.heappop(priority_queue)
        if current in visited:
            continue

        visited.add(current)
        expanded += 1

        if current in goals:
            path = _reconstruct_path(parent, current)
            return path, cost, expanded

        for neighbour, edge_cost in graph.get(current, ()):
            if neighbour in visited:
                continue
            parent[neighbour] = current
            parent_cost[neighbour] = edge_cost
            counter += 1
            heapq.heappush(
                priority_queue,
                (heuristic(neighbour), counter, neighbour, cost + edge_cost)
            )

    return None, float('inf'), expanded


def a_star_search(graph, start, goals, nodes):
    """A* using an admissible timetable-topology heuristic."""
    heuristic = _topology_heuristic_factory(graph, goals)
    counter = 0
    priority_queue = [(heuristic(start), 0.0, counter, start)]
    best_cost = {start: 0.0}
    parent = {start: None}
    parent_cost = {}
    expanded = 0

    while priority_queue:
        f_cost, current_cost, _, current = heapq.heappop(priority_queue)

        if current_cost != best_cost.get(current):
            continue

        expanded += 1

        if current in goals:
            path = _reconstruct_path(parent, current)
            return path, current_cost, expanded

        for neighbour, edge_cost in graph.get(current, ()):
            new_cost = current_cost + edge_cost

            if new_cost < best_cost.get(neighbour, float('inf')):
                best_cost[neighbour] = new_cost
                parent[neighbour] = current
                parent_cost[neighbour] = edge_cost
                counter += 1
                heapq.heappush(
                    priority_queue,
                    (
                        new_cost + heuristic(neighbour),
                        new_cost,
                        counter,
                        neighbour
                    )
                )

    return None, float('inf'), expanded


ALGORITHMS = {
    'BFS': bfs,
    'DFS': dfs,
    'Uniform Cost Search': uniform_cost_search,
    'Greedy Best First Search': greedy_best_first_search,
    'A* Search': a_star_search
}


# ============================================================
# 3. OFFICIAL LOCAL-TRAIN GRAPH HELPERS
# ============================================================

def norm_text(value):
    return ''.join(
        ch.lower()
        for ch in str(value)
        if ch.isalnum()
    )


def _canonical_station_key(value):
    return norm_text(value)


def _parse_schedule_minutes(value):
    """Convert an official HH:MM:SS timetable value to minutes."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    try:
        td = pd.to_timedelta(text)
        return float(td.total_seconds() / 60.0)
    except (ValueError, TypeError):
        return None


def _station_display_map(stations_df, stop_times_df):
    """Use official station names; no coordinates are inferred."""
    result = {}

    for _, row in stations_df.iterrows():
        station = str(row.get('station', '')).strip()
        if not station:
            continue
        key = _canonical_station_key(station)
        if key and key not in result:
            result[key] = station

    for _, row in stop_times_df[['station']].dropna().iterrows():
        station = str(row['station']).strip()
        if not station:
            continue
        key = _canonical_station_key(station)
        if key and key not in result:
            result[key] = station

    return result


def _build_official_railway_graph(stations_df, stop_times_df):
    """Build the railway graph exclusively from official stop sequences.

    Each directed edge is created only when two stations are consecutive
    stops of the same official train service. Edge cost is the scheduled
    time difference between those two stops.
    """
    display_map = _station_display_map(stations_df, stop_times_df)

    nodes = {}
    graph = {}
    metadata = {}
    station_lines = {}

    for key, display in display_map.items():
        node = f'S_{key}'
        nodes[node] = (None, None)
        graph[node] = []
        metadata[node] = {
            'label': display,
            'mode': 'Local Train',
            'line': '',
            'coordinates_available': False
        }

    # Official station master tells us which line(s) a station belongs to.
    for _, row in stations_df.iterrows():
        station = str(row.get('station', '')).strip()
        line = str(row.get('line', '')).strip()
        key = _canonical_station_key(station)
        if key and line:
            station_lines.setdefault(key, set()).add(line)

    for key, lines_for_station in station_lines.items():
        node = f'S_{key}'
        if node in metadata:
            ordered = sorted(lines_for_station)
            metadata[node]['line'] = ' • '.join(ordered)

    edge_costs = {}
    service_count = 0
    edge_records = 0

    work = stop_times_df.copy()
    work['station'] = work['station'].astype(str).str.strip()
    work['line'] = work['line'].astype(str).str.strip()
    work['train_id'] = work['train_id'].astype(str).str.strip()
    work['stop_sequence'] = pd.to_numeric(
        work['stop_sequence'], errors='coerce'
    )
    work['_minutes'] = work['scheduled_time'].map(_parse_schedule_minutes)

    work = work.dropna(
        subset=['station', 'train_id', 'stop_sequence', '_minutes']
    )

    # Train IDs are interpreted together with line, matching the official
    # service records and preventing unrelated line records from mixing.
    for (line_name, train_id), group in work.groupby(
        ['line', 'train_id'], sort=False
    ):
        group = group.sort_values('stop_sequence')
        previous = None
        previous_time = None

        for _, row in group.iterrows():
            key = _canonical_station_key(row['station'])
            node = f'S_{key}' if key else None
            current_time = float(row['_minutes'])

            if node not in graph:
                previous = node
                previous_time = current_time
                continue

            if previous is not None and previous != node and previous_time is not None:
                duration = current_time - previous_time

                # Overnight services can legitimately cross midnight.
                if duration < 0:
                    duration += 24.0 * 60.0

                # Never invent a duration. If the official timetable cannot
                # provide a positive interval, that connection is skipped.
                if duration > 0:
                    pair = (previous, node)
                    existing = edge_costs.get(pair)
                    if existing is None or duration < existing:
                        edge_costs[pair] = duration
                    edge_records += 1

            previous = node
            previous_time = current_time

        service_count += 1

    for (a, b), cost in edge_costs.items():
        graph[a].append((b, float(cost)))

    for node in graph:
        graph[node].sort(key=lambda item: (item[0], item[1]))

    # Add line metadata discovered directly from official stop-time records.
    stop_line_map = {}
    for _, row in work[['station', 'line']].drop_duplicates().iterrows():
        key = _canonical_station_key(row['station'])
        line = str(row['line']).strip()
        if key and line:
            stop_line_map.setdefault(key, set()).add(line)

    for key, lines_for_station in stop_line_map.items():
        node = f'S_{key}'
        if node in metadata:
            all_lines = set(
                filter(None, metadata[node].get('line', '').split(' • '))
            )
            all_lines.update(lines_for_station)
            metadata[node]['line'] = ' • '.join(sorted(all_lines))

    print(
        f'Official railway graph: {len(graph)} stations, '
        f'{len(edge_costs)} directed connections from '
        f'{service_count} official services.'
    )

    return graph, nodes, metadata, edge_records


# ============================================================
# 5. STATION LOOKUP
# ============================================================

def build_station_lookup(stations_df, stop_times_df, nodes, metadata):
    """Build searchable official railway stations only."""
    lookup = {}
    for node, info in metadata.items():
        if info.get('mode') != 'Local Train':
            continue
        name = str(info.get('label', '')).strip()
        if not name:
            continue
        lookup[name] = {
            'node': node,
            'latitude': None,
            'longitude': None,
            'mode': 'Local Train',
            'line': info.get('line', ''),
            'coordinates_available': False
        }
    return dict(sorted(lookup.items(), key=lambda kv: kv[0].lower()))


def node_label(node, node_to_station, metadata):
    if node in node_to_station:
        return node_to_station[node]
    return metadata.get(node, {}).get('label', str(node))


# ============================================================
# 6. ROUTE TYPE INFORMATION
# ============================================================

def route_type_info(points):

    modes = []
    lines = []

    for point in points:

        mode = str(
            point.get(
                'mode',
                ''
            )
        ).strip()

        line = str(
            point.get(
                'line',
                ''
            )
        ).strip()

        if (
            mode
            and mode not in (
                'Ward',
                'Transit'
            )
            and mode not in modes
        ):

            modes.append(mode)

        if (
            line
            and line.lower() != 'nan'
            and line not in lines
        ):

            lines.append(line)

    if len(modes) == 1:

        mode = modes[0]

        if mode == 'Local Train':
            title = 'Local Train Route'
            icon = '🚆'

        elif mode == 'Metro':
            title = 'Metro Route'
            icon = '🚇'

        elif mode == 'Monorail':
            title = 'Monorail Route'
            icon = '🚝'

        elif mode == 'Bus':
            title = 'Bus Route'
            icon = '🚌'

        else:
            title = f'{mode} Route'
            icon = '🛣️'

    elif len(modes) > 1:

        title = 'Mixed Transit Route'
        icon = '🔄'

    else:

        title = 'Transit Route'
        icon = '🛣️'

    return {
        'title': title,
        'icon': icon,
        'modes': modes,
        'lines': lines,
        'is_mixed': len(modes) > 1
    }


def build_mode_summary(points):

    modes = []

    for p in points:

        mode = p.get(
            'mode',
            ''
        )

        if (
            mode
            and mode != 'Ward'
            and mode not in modes
        ):

            modes.append(mode)

    return modes


# ============================================================
# 7. STATION GROUPS
# ============================================================

def build_station_groups(
    station_lookup
):

    groups = {
        'Local Train': [],
        'Metro': [],
        'Monorail': [],
        'Bus': [],
        'Other Transit': []
    }

    for name, info in station_lookup.items():

        mode = str(
            info.get(
                'mode',
                ''
            )
        ).strip()

        if mode not in groups:
            mode = 'Other Transit'

        groups[mode].append(
            {
                'name': name,
                'line': str(
                    info.get(
                        'line',
                        ''
                    )
                ).strip()
            }
        )

    for group in groups.values():

        group.sort(
            key=lambda x:
            x['name'].lower()
        )

    return groups


# ============================================================
# 8. USER SEARCH
# ============================================================

def resolve_station_name(
    value,
    station_lookup,
    normalized_index=None
):

    raw = str(
        value or ''
    ).strip()

    if not raw:
        return None

    if raw in station_lookup:
        return raw

    normalized = norm_text(raw)

    if not normalized:
        return None

    # Exact normalized match.
    # Use a precomputed index during live requests so we do not
    # normalize all ~2500 station names on every request.
    if normalized_index is not None:
        exact = normalized_index.get(normalized)
        if exact is not None:
            return exact
    else:
        for name in station_lookup:
            if norm_text(name) == normalized:
                return name

    candidates = []

    if normalized_index is not None:
        candidate_items = normalized_index.items()
    else:
        candidate_items = (
            (norm_text(name), name)
            for name in station_lookup
        )

    for target, name in candidate_items:

        if not target:
            continue

        ratio = SequenceMatcher(
            None,
            normalized,
            target
        ).ratio()

        if (
            normalized in target
            or target in normalized
        ):

            ratio += 0.20

        candidates.append(
            (
                ratio,
                -len(target),
                name
            )
        )

    candidates.sort(
        reverse=True
    )

    if (
        candidates
        and candidates[0][0] >= 0.72
    ):

        return candidates[0][2]

    return None


# ============================================================
# 9. NEW: ALGORITHM EVALUATION SYSTEM
# ============================================================

"""
HOW THE ALGORITHMS ARE NOW EVALUATED
-------------------------------------

For a fair comparison, every algorithm is tested on exactly the
same set of origin -> destination pairs.

UCS is used as the optimal-distance reference because Uniform
Cost Search finds the least-cost route when all edge costs are
non-negative.

For every test route:

1. Success
   Did the algorithm find a route?

2. Optimality
   How close was its route to the UCS optimal route?

       optimality = UCS_cost / algorithm_cost

   A perfect route = 1.0

3. Search efficiency
   How much searching did the algorithm perform compared with the
   most efficient successful algorithm on the same benchmark route?

       efficiency = minimum_expanded / algorithm_expanded

   Only the algorithm(s) with the fewest expanded nodes on that
   route receive 1.0.

4. Final score
   The weighted benchmark score combines route quality, search effort,
   measured runtime, and success:

       Final Score = Success Rate × (
           0.55 × Route Optimality
           + 0.30 × Search Efficiency
           + 0.15 × Runtime Efficiency
       )

Execution time is included as runtime efficiency in the final score.

The benchmark uses a fixed random seed so the ranking is
repeatable.
"""


def harmonic_mean(a, b):

    if a <= 0 or b <= 0:
        return 0.0

    return (
        2.0 * a * b
        / (a + b)
    )


def create_benchmark_pairs(graph, station_lookup, number_of_pairs=60):
    """Create 60 deterministic, actually reachable official timetable pairs."""
    names = sorted(station_lookup.keys(), key=str.lower)
    if len(names) < 2:
        return []

    rng = random.Random(42)
    candidates = []
    seen = set()

    # Build a lightweight directed reachability cache from each candidate
    # start. This uses only graph topology and prevents unreachable pairs
    # from being counted as benchmark failures.
    reachability = {}

    def reachable_from(start_node):
        if start_node in reachability:
            return reachability[start_node]

        seen_nodes = {start_node}
        queue = deque([start_node])

        while queue:
            current = queue.popleft()
            for neighbour, _ in graph.get(current, ()):
                if neighbour not in seen_nodes:
                    seen_nodes.add(neighbour)
                    queue.append(neighbour)

        reachability[start_node] = seen_nodes
        return seen_nodes

    # Prefer the largest reachable component without assuming any geography.
    for _ in range(max(number_of_pairs * 10, 600)):
        start_name, goal_name = rng.sample(names, 2)
        key = (start_name, goal_name)

        if key in seen:
            continue

        start_node = station_lookup[start_name]['node']
        goal_node = station_lookup[goal_name]['node']

        if goal_node not in reachable_from(start_node):
            continue

        seen.add(key)
        candidates.append(key)

        if len(candidates) >= number_of_pairs:
            break

    return sorted(candidates)


def _benchmark_score(success_rate, optimality, search_efficiency, runtime_efficiency):
    """Weighted evaluation for timetable routing.

    Final score =
      success × (0.55 × route optimality
                 + 0.30 × search efficiency
                 + 0.15 × runtime efficiency)

    Route optimality measures how close the algorithm's scheduled travel
    time is to the best successful result for the same route.
    Search efficiency measures expanded nodes relative to the best result.
    Runtime efficiency is based on the fastest measured successful result.
    """
    return success_rate * (
        0.55 * max(0.0, min(1.0, optimality))
        + 0.30 * max(0.0, min(1.0, search_efficiency))
        + 0.15 * max(0.0, min(1.0, runtime_efficiency))
    )


def _route_cost_regression_metrics(actual_costs, predicted_costs):
    """Measure route-cost error against the best measured route costs."""
    actual = np.asarray(list(actual_costs), dtype=float)
    predicted = np.asarray(list(predicted_costs), dtype=float)
    valid = (
        np.isfinite(actual)
        & np.isfinite(predicted)
        & (actual > 0)
    )
    actual = actual[valid]
    predicted = predicted[valid]

    if actual.size == 0:
        return {
            'samples': 0,
            'mae': None,
            'mse': None,
            'rmse': None,
            'mape': None,
            'r2': None
        }

    errors = predicted - actual
    absolute_errors = np.abs(errors)
    mae = float(np.mean(absolute_errors))
    mse = float(np.mean(np.square(errors)))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(absolute_errors / actual) * 100.0)
    total_variance = float(np.sum(np.square(actual - np.mean(actual))))
    residual_variance = float(np.sum(np.square(errors)))
    if total_variance > 0:
        r2 = 1.0 - (residual_variance / total_variance)
    else:
        r2 = 1.0 if residual_variance == 0 else 0.0

    return {
        'samples': int(actual.size),
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'r2': float(r2)
    }


def evaluate_algorithms(
    graph,
    nodes,
    station_lookup,
    number_of_pairs=BENCHMARK_ROUTE_COUNT
):
    print(f'Benchmarking algorithms on {number_of_pairs} fixed official routes...')

    pairs = create_benchmark_pairs(graph, station_lookup, number_of_pairs)
    metrics = {
        name: {
            'successes': 0,
            'optimality_values': [],
            'efficiency_values': [],
            'runtime_values': [],
            'runtime_efficiency_values': [],
            'route_costs': [],
            'expanded_nodes': [],
            'reference_costs': [],
            'predicted_costs': []
        }
        for name in ALGORITHMS
    }

    valid_benchmarks = 0
    benchmark_rows = []

    for start_name, goal_name in pairs:
        start = station_lookup[start_name]['node']
        goal = station_lookup[goal_name]['node']

        measurements = {}

        for algorithm_name, algorithm in ALGORITHMS.items():
            t0 = time.perf_counter()
            path, cost, expanded = algorithm(graph, start, {goal}, nodes)
            elapsed = time.perf_counter() - t0

            if path is not None and np.isfinite(cost) and cost > 0:
                measurements[algorithm_name] = {
                    'cost': float(cost),
                    'expanded': int(expanded),
                    'time': float(elapsed)
                }

        if not measurements:
            continue

        valid_benchmarks += 1
        best_cost = min(m['cost'] for m in measurements.values())
        best_expanded = min(m['expanded'] for m in measurements.values())
        best_time = min(m['time'] for m in measurements.values())

        benchmark_rows.append({
            'start': start_name,
            'destination': goal_name,
            'best_cost': best_cost
        })

        for algorithm_name, measurement in measurements.items():
            data = metrics[algorithm_name]
            data['successes'] += 1
            data['route_costs'].append(measurement['cost'])
            data['expanded_nodes'].append(measurement['expanded'])
            data['runtime_values'].append(measurement['time'])
            data['reference_costs'].append(best_cost)
            data['predicted_costs'].append(measurement['cost'])
            data['runtime_efficiency_values'].append(
                max(0.0, min(1.0, best_time / measurement['time']))
            )

            # Lower travel time is better.
            optimality = best_cost / measurement['cost'] if measurement['cost'] > 0 else 0.0

            # Fewer expanded nodes is better.
            search_efficiency = (
                best_expanded / measurement['expanded']
                if measurement['expanded'] > 0 else 0.0
            )

            # Faster measured execution is better.
            runtime_efficiency = (
                best_time / measurement['time']
                if measurement['time'] > 0 else 0.0
            )

            data['optimality_values'].append(
                max(0.0, min(1.0, optimality))
            )
            data['efficiency_values'].append(
                max(0.0, min(1.0, search_efficiency))
            )

    results = []

    for algorithm_name in ALGORITHMS:
        data = metrics[algorithm_name]

        if valid_benchmarks:
            success_rate = data['successes'] / valid_benchmarks
            optimality = (
                float(np.mean(data['optimality_values']))
                if data['optimality_values'] else 0.0
            )
            efficiency = (
                float(np.mean(data['efficiency_values']))
                if data['efficiency_values'] else 0.0
            )
            avg_time = (
                float(np.median(data['runtime_values']))
                if data['runtime_values'] else 0.0
            )
            runtime_efficiency = (
                float(np.mean(data['runtime_efficiency_values']))
                if data['runtime_efficiency_values'] else 0.0
            )
            final_score = _benchmark_score(
                success_rate, optimality, efficiency, runtime_efficiency
            )
        else:
            success_rate = optimality = efficiency = runtime_efficiency = final_score = 0.0
            avg_time = 0.0

        regression = _route_cost_regression_metrics(
            data['reference_costs'],
            data['predicted_costs']
        )

        results.append({
            'algorithm': algorithm_name,
            'success_rate': success_rate,
            'optimality': optimality,
            'efficiency': efficiency,
            'runtime_efficiency': runtime_efficiency,
            'final_score': final_score,
            'average_path_cost': (
                float(np.mean(data['route_costs']))
                if data['route_costs'] else float('inf')
            ),
            'average_nodes_expanded': (
                float(np.mean(data['expanded_nodes']))
                if data['expanded_nodes'] else float('inf')
            ),
            'average_execution_time': avg_time,
            'successful_searches': data['successes'],
            'benchmark_routes': valid_benchmarks,
            'route_mae': regression['mae'],
            'route_mse': regression['mse'],
            'route_rmse': regression['rmse'],
            'route_mape': regression['mape'],
            'route_r2': regression['r2'],
            'regression_samples': regression['samples']
        })

    results_df = pd.DataFrame(results)

    algorithm_order = {
        'A* Search': 0,
        'Uniform Cost Search': 1,
        'BFS': 2,
        'Greedy Best First Search': 3,
        'DFS': 4
    }

    if not results_df.empty:
        results_df['_order'] = results_df['algorithm'].map(algorithm_order)
        results_df = results_df.sort_values(
            by=['final_score', 'optimality', 'efficiency', 'success_rate', '_order'],
            ascending=[False, False, False, False, True]
        ).drop(columns=['_order'])

    best_two = (
        results_df.head(2)['algorithm'].tolist()
        if not results_df.empty else []
    )

    print('\n========== OFFICIAL DATA ALGORITHM EVALUATION ==========')
    for _, row in results_df.iterrows():
        print(
            f"{row['algorithm']}: "
            f"Score={row['final_score']:.3f}, "
            f"Optimality={row['optimality']:.3f}, "
            f"SearchEfficiency={row['efficiency']:.3f}, "
            f"Success={row['success_rate']:.3f}"
        )
    print(f'BEST 2: {best_two}')

    return results_df, best_two


# ============================================================
# 10. ROUTE RECOMMENDATION
# ============================================================

def choose_recommended_result(results, preference='balanced'):
    """Choose the algorithm/result from the user's requested objective.

    IMPORTANT: this function never maps a preference to a fixed algorithm.
    It evaluates the actual route produced by every algorithm for the
    current origin/destination and selects the result that best satisfies
    the selected objective.

    Objectives:
      * distance    -> minimum route distance/cost
      * stops       -> minimum number of stops, then distance
      * search_time -> minimum nodes explored, then execution time
      * balanced    -> best normalized combination of distance, stops,
                       search effort, and route optimality

    """
    if not results:
        return None

    preference = str(preference or 'balanced').strip().lower()

    # Simple objective-based selection. No algorithm names are referenced.
    if preference == 'distance':
        return min(
            results,
            key=lambda r: (
                float(r.get('path_cost', float('inf'))),
                int(r.get('stops', 10**9)),
                int(r.get('nodes', 10**9)),
                float(r.get('time', float('inf')))
            )
        )

    if preference == 'stops':
        return min(
            results,
            key=lambda r: (
                int(r.get('stops', 10**9)),
                float(r.get('path_cost', float('inf'))),
                int(r.get('nodes', 10**9)),
                float(r.get('time', float('inf')))
            )
        )

    if preference == 'search_time':
        return min(
            results,
            key=lambda r: (
                int(r.get('nodes', 10**9)),
                float(r.get('time', float('inf'))),
                float(r.get('path_cost', float('inf'))),
                int(r.get('stops', 10**9))
            )
        )

    # Balanced: score the actual routes relative to each other.
    # Lower distance/stops/search effort is better; higher route
    # optimality is better. The weights describe the Balanced objective,
    # not a fixed algorithm choice.
    def values(field, default):
        return [
            float(r.get(field, default))
            for r in results
        ]

    distances = values('path_cost', float('inf'))
    stops = values('stops', float('inf'))
    expanded = values('nodes', float('inf'))
    optimality = values('route_optimality', 0.0)

    def minmax_inverse(value, series):
        finite = [x for x in series if x != float('inf')]
        if not finite:
            return 0.0
        lo = min(finite)
        hi = max(finite)
        if hi == lo:
            return 1.0
        return (hi - value) / (hi - lo)

    scored = []
    for result, distance, stop_count, expanded_count, opt in zip(
        results,
        distances,
        stops,
        expanded,
        optimality
    ):
        balanced_score = (
            0.40 * minmax_inverse(distance, distances)
            + 0.30 * minmax_inverse(stop_count, stops)
            + 0.20 * minmax_inverse(expanded_count, expanded)
            + 0.10 * max(0.0, min(1.0, opt))
        )
        scored.append((balanced_score, result))

    return max(
        scored,
        key=lambda item: (
            item[0],
            -float(item[1].get('path_cost', float('inf'))),
            -int(item[1].get('stops', 10**9)),
            -int(item[1].get('nodes', 10**9))
        )
    )[1]


def build_journey_legs(path, metadata, node_to_station, bus_edge_routes):
    """Convert a graph path into user-facing bus/transfer instructions."""
    legs = []
    if len(path) < 2:
        return legs

    def label(node):
        return node_label(node, node_to_station, metadata)

    current = None
    for a, b in zip(path, path[1:]):
        routes = list(bus_edge_routes.get((a, b), ()))
        a_mode = metadata.get(a, {}).get('mode', 'Transit')
        b_mode = metadata.get(b, {}).get('mode', 'Transit')

        if routes:
            kind = 'bus'
            route_set = set(routes)
            if current and current['kind'] == 'bus':
                common = set(current['routes']) & route_set
                if common:
                    current['routes'] = sorted(common)
                    current['to'] = label(b)
                    current['stops'] += 1
                    continue
            current = {
                'kind': 'bus',
                'routes': sorted(routes),
                'from': label(a),
                'to': label(b),
                'stops': 1
            }
            legs.append(current)
            continue

        kind = 'transfer' if a_mode != b_mode or a_mode == 'Ward' or b_mode == 'Ward' else 'ride'
        current = {
            'kind': kind,
            'from': label(a),
            'to': label(b),
            'mode': b_mode
        }
        legs.append(current)

    return legs


# ============================================================
# 11. USER ROUTE CALCULATION
# ============================================================

def calculate_user_route(
    graph,
    nodes,
    station_lookup,
    metadata,
    start_station,
    destination_station,
    preference='balanced',
    normalized_station_index=None,
    node_to_station=None,
    bus_edge_routes=None
):

    resolved_start = resolve_station_name(
        start_station,
        station_lookup,
        normalized_station_index
    )

    resolved_destination = resolve_station_name(
        destination_station,
        station_lookup,
        normalized_station_index
    )

    if resolved_start is None:

        return (
            None,
            f"Starting location '{start_station}' "
            f"was not found. Please choose a "
            f"location from the search list."
        )

    if resolved_destination is None:

        return (
            None,
            f"Destination '{destination_station}' "
            f"was not found. Please choose a "
            f"location from the search list."
        )

    start_station = resolved_start
    destination_station = resolved_destination

    start_node = (
        station_lookup[
            start_station
        ]['node']
    )

    destination_node = (
        station_lookup[
            destination_station
        ]['node']
    )

    if start_node == destination_node:

        return (
            None,
            'Starting location and destination '
            'must be different.'
        )

    if station_lookup is cached_station_lookup:
        node_to_station = cached_node_to_station
    else:
        node_to_station = {
            info['node']: name
            for name, info
            in station_lookup.items()
        }

    if bus_edge_routes is None:
        bus_edge_routes = {}

    # --------------------------------------------------------
    # HOT-PATH CACHE
    # --------------------------------------------------------
    # Cache by resolved graph nodes, not raw user text. This means
    # "Kurla", " kurla ", etc. share the same cached route.
    cache_key = (
        start_node,
        destination_node,
        preference
    )

    cached_route = _route_cache.get(cache_key)

    if cached_route is not None:
        _route_cache.move_to_end(cache_key)
        return cached_route, None

    results = []

    # --------------------------------------------------------
    # Run all algorithms
    # --------------------------------------------------------

    for algorithm_name, algorithm in ALGORITHMS.items():

        started = time.perf_counter()

        path, cost, expanded = algorithm(
            graph,
            start_node,
            {destination_node},
            nodes
        )

        execution_time = (
            time.perf_counter()
            - started
        )

        if path is None:
            continue

        path_points = []

        for node in path:

            point = {

                'label': node_label(
                    node,
                    node_to_station,
                    metadata
                ),

                'lat': (
                    float(nodes[node][0])
                    if nodes[node][0] is not None else None
                ),

                'lon': (
                    float(nodes[node][1])
                    if nodes[node][1] is not None else None
                ),

                'mode': metadata.get(
                    node,
                    {}
                ).get(
                    'mode',
                    'Transit'
                ),

                'line': metadata.get(
                    node,
                    {}
                ).get(
                    'line',
                    ''
                ),

                'type':
                    'station'
                    if node in node_to_station
                    else 'ward'
            }

            path_points.append(
                point
            )

        # ----------------------------------------------------
        # Remove consecutive duplicate labels
        # ----------------------------------------------------

        simplified_points = []

        for p in path_points:

            if (
                not simplified_points
                or
                norm_text(simplified_points[-1].get('label', ''))
                != norm_text(p.get('label', ''))
            ):

                simplified_points.append(
                    p
                )

        results.append(
            {
                'algorithm':
                    algorithm_name,

                'route_names':
                    [
                        p['label']
                        for p
                        in simplified_points
                    ],

                'path_points':
                    simplified_points,

                'path_cost':
                    round(
                        float(cost),
                        3
                    ),

                'stops':
                    max(
                        len(
                            simplified_points
                        ) - 1,
                        0
                    ),

                'nodes':
                    int(expanded),

                'time':
                    round(
                        execution_time,
                        6
                    ),

                'mode_summary':
                    build_mode_summary(
                        simplified_points
                    ),

                'route_type':
                    route_type_info(
                        simplified_points
                    ),

                'journey_legs':
                    build_journey_legs(
                        path,
                        metadata,
                        node_to_station,
                        bus_edge_routes
                    ),

                'score_basis':
                    'Official timetable benchmark using '
                    'route optimality, search efficiency '
                    'and measured runtime.'
            }
        )

    if not results:

        return (
            None,
            'No route could be found between '
            'these two locations.'
        )

    # --------------------------------------------------------
    # Route-level metrics use the best successful observed result.
    # No algorithm is assumed to be "correct" merely by its name.
    # --------------------------------------------------------
    best_route_cost = min(
        float(r['path_cost'])
        for r in results
        if np.isfinite(r['path_cost']) and r['path_cost'] > 0
    )
    best_route_nodes = min(
        int(r['nodes'])
        for r in results
        if int(r['nodes']) > 0
    )

    for result in results:
        optimality = (
            best_route_cost / result['path_cost']
            if result['path_cost'] > 0 else 0.0
        )
        efficiency = (
            best_route_nodes / result['nodes']
            if result['nodes'] > 0 else 0.0
        )

        result['route_optimality'] = round(
            max(0.0, min(1.0, optimality)), 3
        )
        result['route_efficiency'] = round(
            max(0.0, min(1.0, efficiency)), 3
        )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT rank the Best 2 using the current route.
    #
    # The global benchmark decides the Best 2.
    # This makes the result stable.
    # --------------------------------------------------------

    global_algorithm_scores = {
        item['algorithm']:
        item['score']
        for item
        in cached_algorithm_results
    }

    for result in results:

        result[
            'network_score'
        ] = round(
            global_algorithm_scores.get(
                result['algorithm'],
                0.0
            ),
            3
        )

        result[
            'is_best_two'
        ] = (
            result['algorithm']
            in cached_best_two
        )

    # --------------------------------------------------------
    # Sort according to the actual benchmark
    # --------------------------------------------------------

    algorithm_order = {
        'A* Search': 0,
        'Uniform Cost Search': 1,
        'BFS': 2,
        'Greedy Best First Search': 3,
        'DFS': 4
    }

    # Objective-relative preference score for display/diagnostics.
    recommended_result = choose_recommended_result(
        results, preference
    )

    if preference == 'distance':
        best_value = min(float(r.get('path_cost', float('inf'))) for r in results)
        for result in results:
            result['preference_score'] = round(
                best_value / max(float(result.get('path_cost', float('inf'))), 0.001),
                6
            )
    elif preference == 'stops':
        best_value = min(int(r.get('stops', 10**9)) for r in results)
        for result in results:
            result['preference_score'] = round(
                best_value / max(int(result.get('stops', 10**9)), 1),
                6
            )
    elif preference == 'search_time':
        best_value = min(int(r.get('nodes', 10**9)) for r in results)
        for result in results:
            result['preference_score'] = round(
                best_value / max(int(result.get('nodes', 10**9)), 1),
                6
            )
    else:
        # Store the chosen result at 1.0 and a simple normalized diagnostic
        # score for the others.
        for result in results:
            result['preference_score'] = (
                1.0
                if result is recommended_result
                else 0.0
            )

    results.sort(
        key=lambda x: (
            -x['network_score'],
            algorithm_order.get(
                x['algorithm'],
                99
            )
        )
    )

    route_response = {
        'start': start_station,
        'destination': destination_station,
        'preference': preference,
        'best_two': cached_best_two,
        'recommended_result': recommended_result,
        'results': results
    }

    _route_cache[cache_key] = route_response
    _route_cache.move_to_end(cache_key)

    while len(_route_cache) > ROUTE_CACHE_MAX_SIZE:
        _route_cache.popitem(last=False)

    return route_response, None


# ============================================================
# 11. NETWORK-WIDE ACCESSIBILITY
# ============================================================
# The official BMC dataset contains ward + population only. The official
# railway datasets do not contain ward coordinates, so geographic transit
# desert calculations are intentionally not performed.
#
# The official-data population planning calculation is implemented immediately
# below in process_transit_data() via calculate_accessibility().
#
# ============================================================
# 12. PROCESS DATA
# ============================================================

WARD_LOCALITY_NAMES = {
    'A': 'Colaba, Navy Nagar, Cuffe Parade, Churchgate, Fort',
    'B': 'Masjid Bunder, Mohd. Ali Road, Dongri, Bhendi Bazar',
    'C': 'Marine Lines, Bhuleshwar, Pydhonie, Chira Bazar',
    'D': 'Malabar Hill, Girgaon, Grant Road, Walkeshwar, Tardeo',
    'E': 'Byculla, Mazgaon, Reay Road, Madanpura',
    'F/S': 'Parel, Sewri, Naigaon, Lalbaug, Kalachowki',
    'F/N': 'Sion, Matunga, Wadala, Antop Hill',
    'G/S': 'Worli, Prabhadevi, Lower Parel, Mahalaxmi',
    'G/N': 'Dadar (West), Mahim, Dharavi',
    'H/E': 'Bandra East, Santacruz East, Khar East, Kalina, Vakola',
    'H/W': 'Bandra West, Santacruz West, Khar West',
    'K/E': 'Andheri East, Jogeshwari East, Vile Parle East',
    'K/W': 'Andheri West, Vile Parle West, Juhu, Versova, Lokhandwala',
    'P/S': 'Goregaon East, Goregaon West, Aarey Colony',
    'P/N': 'Malad East, Malad West, Marve, Aksa, Pathanwadi',
    'R/S': 'Kandivali East, Kandivali West, Charkop, Poisar',
    'R/C': 'Borivali East, Borivali West, Gorai, Magathane',
    'R/N': 'Dahisar East, Dahisar West, IC Colony, Rawalpada',
    'L': 'Kurla East, Kurla West, Sakinaka, Chandivali',
    'M/E': 'Govandi, Mankhurd, Deonar, Shivaji Nagar',
    'M/W': 'Chembur, Tilak Nagar, Shell Colony',
    'N': 'Ghatkopar East, Ghatkopar West, Pant Nagar, Vikhroli West',
    'S': 'Bhandup, Kanjurmarg, Vikhroli East, Powai',
    'T': 'Mulund East, Mulund West, Nahur'
}

def ward_locality_name(value):
    raw = str(value).strip()
    aliases = {
        'F/NORTH': 'F/N',
        'F/SOUTH': 'F/S',
        'G/NORTH': 'G/N',
        'G/SOUTH': 'G/S',
        'H/EAST': 'H/E',
        'H/WEST': 'H/W',
        'K/EAST': 'K/E',
        'K/WEST': 'K/W',
        'M/EAST': 'M/E',
        'M/WEST': 'M/W',
        'P/NORTH': 'P/N',
        'P/SOUTH': 'P/S',
        'R/CENTRAL': 'R/C',
        'R/NORTH': 'R/N',
        'R/SOUTH': 'R/S'
    }
    key = aliases.get(raw.upper(), raw.upper())
    return WARD_LOCALITY_NAMES.get(key, raw)


def calculate_accessibility(df_wards, *args, **kwargs):
    """Official-data-only ward planning view.

    The supplied BMC file contains ward and population only. It does not
    contain ward coordinates or a transit-distance field, so a geographic
    transit-desert score is not calculated. Instead, the dashboard exposes
    a transparent population-based planning metric.
    """
    work = df_wards.copy()
    work['population'] = pd.to_numeric(work['population'], errors='coerce')
    work = work.dropna(subset=['ward', 'population']).copy()

    max_population = float(work['population'].max()) if not work.empty else 0.0
    total_population = float(work['population'].sum()) if not work.empty else 0.0

    if max_population > 0:
        work['population_priority_score'] = (
            work['population'] / max_population * 100.0
        )
    else:
        work['population_priority_score'] = 0.0

    work['population_share'] = (
        work['population'] / total_population * 100.0
        if total_population > 0 else 0.0
    )

    poor_access = []
    for _, row in work.sort_values('population', ascending=False).iterrows():
        poor_access.append({
            'ward': str(row['ward']),
            'name': ward_locality_name(row['ward']),
            'population': int(row['population']),
            'score': round(float(row['population_priority_score']), 2),
            'population_share': round(float(row['population_share']), 2),
            'analysis_status': 'Population-based planning indicator; transit distance is not available in the supplied ward dataset'
        })

    prioritized = [
        {
            'ward': str(row['ward']),
            'name': ward_locality_name(row['ward']),
            'population': int(row['population']),
            'score': round(float(row['population_priority_score']), 2),
            'population_share': round(float(row['population_share']), 2),
            'analysis_status': 'Population-based planning priority; not a transit-access score'
        }
        for _, row in work.sort_values(
            ['population_priority_score', 'population'],
            ascending=False
        ).head(5).iterrows()
    ]

    return poor_access, prioritized


def process_transit_data():
    paths = {
        key: os.path.join(DATA_DIR, filename)
        for key, filename in OFFICIAL_FILES.items()
    }

    for key, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'Missing official dataset: {path}'
            )

    cached = _load_processed_cache(paths)
    if cached is not None:
        return cached

    print('Loading official Mumbai transit datasets...')

    df_wards = pd.read_csv(paths['wards'])
    stations = pd.read_csv(paths['stations'], encoding='utf-8-sig')
    services = pd.read_csv(paths['services'], encoding='utf-8-sig')
    stop_times = pd.read_csv(paths['stop_times'], encoding='utf-8-sig')
    best_routes = pd.read_csv(paths['best_routes'], encoding='utf-8-sig')
    best_summary = pd.read_csv(paths['best_summary'], encoding='utf-8-sig')

    required_ward = {'ward', 'population'}
    required_stations = {'line', 'station'}
    required_services = {'line', 'train_id', 'recorded_stops'}
    required_stop_times = {
        'line', 'train_id', 'station',
        'scheduled_time', 'stop_sequence'
    }

    for required, frame, label in [
        (required_ward, df_wards, 'BMC ward population'),
        (required_stations, stations, 'local train stations'),
        (required_services, services, 'local train services'),
        (required_stop_times, stop_times, 'local train stop times')
    ]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f'{label} is missing required columns: {sorted(missing)}'
            )

    graph, nodes, metadata, edge_records = _build_official_railway_graph(
        stations,
        stop_times
    )

    station_lookup = build_station_lookup(
        stations,
        stop_times,
        nodes,
        metadata
    )

    results_df, best_two = evaluate_algorithms(
        graph,
        nodes,
        station_lookup,
        number_of_pairs=BENCHMARK_ROUTE_COUNT
    )

    poor_access, prioritized = calculate_accessibility(
        df_wards
    )

    algorithm_results = []
    for _, row in results_df.iterrows():
        algorithm_results.append({
            'algorithm': row['algorithm'],
            'score': round(float(row['final_score']), 3),
            'success_rate': round(float(row['success_rate']) * 100, 1),
            'optimality': round(float(row['optimality']) * 100, 1),
            'efficiency': round(float(row['efficiency']) * 100, 1),
            'runtime_efficiency': round(
                float(row['runtime_efficiency']) * 100,
                1
            ),
            'route_mae': (
                round(float(row['route_mae']), 3)
                if np.isfinite(row['route_mae']) else None
            ),
            'route_mse': (
                round(float(row['route_mse']), 3)
                if np.isfinite(row['route_mse']) else None
            ),
            'route_rmse': (
                round(float(row['route_rmse']), 3)
                if np.isfinite(row['route_rmse']) else None
            ),
            'route_mape': (
                round(float(row['route_mape']), 1)
                if np.isfinite(row['route_mape']) else None
            ),
            'route_r2': (
                round(float(row['route_r2']), 3)
                if np.isfinite(row['route_r2']) else None
            ),
            'regression_samples': int(row['regression_samples']),
            'path_cost': (
                round(float(row['average_path_cost']), 3)
                if np.isfinite(row['average_path_cost']) else None
            ),
            'nodes': (
                round(float(row['average_nodes_expanded']), 2)
                if np.isfinite(row['average_nodes_expanded']) else None
            ),
            'time': round(float(row['average_execution_time']), 6),
            'successful_searches': int(row['successful_searches']),
            'benchmark_routes': int(row['benchmark_routes'])
        })

    station_groups = build_station_groups(station_lookup)

    network_summary = []
    for _, row in best_summary.iterrows():
        network_summary.append({
            'metric': str(row.get('metric', '')),
            'value': row.get('value'),
            'unit': str(row.get('unit', '')),
            'source_date': str(row.get('source_date', '')),
            'agency': str(row.get('agency', 'BEST')),
            'notes': str(row.get('notes', ''))
        })

    verified_best_routes = []
    for _, row in best_routes.iterrows():
        verified_best_routes.append({
            'route_no': str(row.get('route_no', '')),
            'from': str(row.get('from', '')),
            'to': str(row.get('to', '')),
            'note': str(row.get('official_itinerary_or_note', '')),
            'verification_basis': str(row.get('verification_basis', ''))
        })

    processed_data = (
        poor_access,
        prioritized,
        algorithm_results,
        best_two,
        station_lookup,
        station_groups,
        graph,
        nodes,
        metadata,
        {},
        network_summary,
        verified_best_routes
    )

    # Cache format now has 12 items.
    payload = {
        'metadata': _cache_metadata(paths),
        'data': processed_data
    }
    temp_path = PROCESSED_CACHE_PATH + '.tmp'
    try:
        with open(temp_path, 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_path, PROCESSED_CACHE_PATH)
    except Exception as exc:
        print(f'Could not save processed cache: {exc}')

    return processed_data


# ============================================================
# 13. STARTUP
# ============================================================

print(
    'Processing transit data on startup...'
)

(
    cached_poor_access,
    cached_prioritized,
    cached_algorithm_results,
    cached_best_two,
    cached_station_lookup,
    cached_station_groups,
    cached_graph,
    cached_nodes,
    cached_metadata,
    cached_bus_edge_routes,
    cached_network_summary,
    cached_verified_best_routes
) = process_transit_data()

cached_node_to_station = {
    info['node']: name
    for name, info in cached_station_lookup.items()
}


cached_normalized_station_index = {}

for name in cached_station_lookup:
    normalized_name = norm_text(name)

    if normalized_name and normalized_name not in cached_normalized_station_index:
        cached_normalized_station_index[normalized_name] = name

print(
    'Data processing complete!'
)

print(
    f'{len(cached_station_lookup)} '
    f'searchable locations.'
)

print(
    f'BEST 2 ALGORITHMS: '
    f'{cached_best_two}'
)


# ============================================================
# 14. HOME PAGE
# ============================================================

@app.route('/')
def home():

    station_names = sorted(
        cached_station_lookup.keys(),
        key=str.lower
    )

    return render_template(

        'index.html',

        poor_access=
            cached_poor_access,

        prioritized=
            cached_prioritized,

        algorithm_results=
            cached_algorithm_results,

        best_two=
            cached_best_two,

        stations=
            station_names,

        station_groups=
            cached_station_groups,

        network_summary=
            cached_network_summary,

        verified_best_routes=
            cached_verified_best_routes,

        benchmark_score_description=
            BENCHMARK_SCORE_DESCRIPTION,

        route_regression_description=
            ROUTE_REGRESSION_DESCRIPTION
    )


# ============================================================
# 15. FIND ROUTE
# ============================================================

@app.route(
    '/find-route',
    methods=['POST']
)
def find_route():

    data = (
        request.get_json(
            silent=True
        )
        or {}
    )

    start_station = str(
        data.get(
            'start',
            ''
        )
    ).strip()

    destination_station = str(
        data.get(
            'destination',
            ''
        )
    ).strip()

    preference = str(
        data.get(
            'preference',
            'balanced'
        )
    ).strip()

    allowed_preferences = {
        'balanced',
        'distance',
        'stops',
        'search_time'
    }

    if preference not in allowed_preferences:
        preference = 'balanced'

    if (
        not start_station
        or not destination_station
    ):

        return jsonify(
            {
                'error':
                    'Please select both a '
                    'starting location and '
                    'destination.'
            }
        ), 400

    route_data, error = (
        calculate_user_route(
            cached_graph,
            cached_nodes,
            cached_station_lookup,
            cached_metadata,
            start_station,
            destination_station,
            preference,
            cached_normalized_station_index,
            cached_node_to_station,
            cached_bus_edge_routes
        )
    )

    if error:

        return jsonify(
            {
                'error': error
            }
        ), 400

    return jsonify(
        route_data
    )


# ============================================================
# 16. RUN
# ============================================================
if __name__ == '__main__':
    # Reloader stays disabled so expensive startup work is never run twice.
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'

    app.run(
        debug=debug_mode,
        use_reloader=False
    )
