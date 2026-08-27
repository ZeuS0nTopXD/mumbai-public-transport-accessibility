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
# FAST STARTUP CACHE
# ============================================================
# Cache the fully processed application state so CSV parsing, graph
# construction, benchmarking and accessibility analysis happen only
# when the code or source datasets actually change.

CACHE_VERSION = 11
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_CACHE_PATH = os.path.join(
    DATA_DIR,
    'processed_app_cache.pkl'
)

# Small in-memory route cache.
# Repeated searches become essentially instant while memory stays bounded.
ROUTE_CACHE_MAX_SIZE = 1024
_route_cache = OrderedDict()


def _file_signature(path):
    stat = os.stat(path)
    return {
        'size': stat.st_size,
        'mtime_ns': stat.st_mtime_ns
    }


def _source_signature():
    try:
        with open(__file__, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def _cache_metadata(wards_path, transit_path, gtfs_stops_path, gtfs_sequences_path):
    return {
        'cache_version': CACHE_VERSION,
        'source_hash': _source_signature(),
        'wards': _file_signature(wards_path),
        'transit': _file_signature(transit_path),
        'gtfs_stops': _file_signature(gtfs_stops_path),
        'gtfs_sequences': _file_signature(gtfs_sequences_path)
    }


def _load_processed_cache(wards_path, transit_path, gtfs_stops_path, gtfs_sequences_path):
    if not os.path.exists(PROCESSED_CACHE_PATH):
        return None

    try:
        with open(PROCESSED_CACHE_PATH, 'rb') as f:
            payload = pickle.load(f)

        if payload.get('metadata') != _cache_metadata(
            wards_path,
            transit_path,
            gtfs_stops_path,
            gtfs_sequences_path
        ):
            print('Processed cache is outdated; rebuilding data...')
            return None

        data = payload.get('data')
        if not isinstance(data, tuple) or len(data) != 10:
            print('Processed cache format is invalid; rebuilding data...')
            return None

        print('Loaded processed data from cache.')
        return data

    except Exception as exc:
        print(
            f'Could not load processed cache ({exc}); rebuilding data...'
        )
        return None


def _save_processed_cache(data, wards_path, transit_path, gtfs_stops_path, gtfs_sequences_path):
    os.makedirs(DATA_DIR, exist_ok=True)

    payload = {
        'metadata': _cache_metadata(
            wards_path,
            transit_path,
            gtfs_stops_path,
            gtfs_sequences_path
        ),
        'data': data
    }

    temp_path = PROCESSED_CACHE_PATH + '.tmp'

    try:
        with open(temp_path, 'wb') as f:
            pickle.dump(
                payload,
                f,
                protocol=pickle.HIGHEST_PROTOCOL
            )

        os.replace(
            temp_path,
            PROCESSED_CACHE_PATH
        )

        print('Saved processed data cache.')

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



def greedy_best_first_search(graph, start, goals, nodes):
    """Greedy best-first search with cached geometry and parent pointers."""
    goals = tuple(goals)

    if len(goals) == 1:
        goal = goals[0]

        def heuristic(node):
            return euclidean_distance(node, goal, nodes)
    else:
        heuristic_cache = {}

        def heuristic(node):
            cached = heuristic_cache.get(node)
            if cached is not None:
                return cached

            value = min(
                euclidean_distance(node, goal, nodes)
                for goal in goals
            )
            heuristic_cache[node] = value
            return value

    counter = 0
    priority_queue = [
        (heuristic(start), counter, start, 0.0)
    ]
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
                (
                    heuristic(neighbour),
                    counter,
                    neighbour,
                    cost + edge_cost
                )
            )

    return None, float('inf'), expanded



def a_star_search(graph, start, goals, nodes):
    """A* with O(V) parent storage and a single-goal fast heuristic path."""
    goals = tuple(goals)

    if len(goals) == 1:
        goal = goals[0]

        def heuristic(node):
            return euclidean_distance(node, goal, nodes)
    else:
        heuristic_cache = {}

        def heuristic(node):
            cached = heuristic_cache.get(node)
            if cached is not None:
                return cached

            value = min(
                euclidean_distance(node, goal, nodes)
                for goal in goals
            )
            heuristic_cache[node] = value
            return value

    counter = 0
    priority_queue = [
        (heuristic(start), 0.0, counter, start)
    ]
    best_cost = {start: 0.0}
    parent = {start: None}
    parent_cost = {}
    expanded = 0

    while priority_queue:
        f_cost, current_cost, _, current = heapq.heappop(
            priority_queue
        )

        if current_cost != best_cost.get(current):
            continue

        expanded += 1

        if current in goals:
            path = _reconstruct_path(parent, current)
            return path, current_cost, expanded

        for neighbour, edge_cost in graph.get(current, ()):
            new_cost = current_cost + edge_cost

            if new_cost < best_cost.get(
                neighbour,
                float('inf')
            ):
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
# 3. LOCAL TRAIN COORDINATES
# ============================================================

LOCAL_COORDS = {

    'CSMT': (18.9402, 72.8356),
    'Masjid': (18.9518, 72.8380),
    'Sandhurst Road': (18.9609, 72.8388),
    'Dockyarad Road': (18.9680, 72.8415),
    'Reay Road': (18.9763, 72.8442),
    'Cotton Green': (18.9860, 72.8447),
    'Sewri': (19.0005, 72.8551),
    'Vadala Road': (19.0173, 72.8586),
    'GTB Nagar': (19.0298, 72.8652),
    'Chunabhatti': (19.0525, 72.8780),
    'Kurla': (19.0650, 72.8790),
    'Tilaknagar': (19.0675, 72.8890),
    'Chembur': (19.0626, 72.8980),
    'Govandi': (19.0552, 72.9146),
    'Mankhurd': (19.0480, 72.9325),
    'Vashi': (19.0660, 72.9980),
    'Sanpada': (19.0678, 73.0097),
    'Juinagar': (19.0557, 73.0155),
    'Nerul': (19.0343, 73.0172),
    'Seawood Darave': (19.0213, 73.0173),
    'Belapur CBD': (19.0167, 73.0397),
    'Kharghar': (19.0485, 73.0690),
    'Mansarovar': (19.0182, 73.0955),
    'Khandeshwar': (19.0065, 73.0985),
    'Panvel': (18.9894, 73.1175),
}

# Real railway topology.  The old code used one global station order for
# every Local Train line, which could accidentally make a Central-line node
# behave like a Harbour-line neighbour.  Keep each physical line separate.
#
# A station such as Kurla is an interchange.  If the CSV contains Kurla only
# once (for example tagged as Central), the topology builder below can still
# use that physical Kurla node as the Harbour-line interchange anchor.
LOCAL_TRAIN_TOPOLOGY = {
    'harbour': [
        'CSMT', 'Masjid', 'Sandhurst Road', 'Dockyarad Road', 'Reay Road',
        'Cotton Green', 'Sewri', 'Vadala Road', 'GTB Nagar', 'Chunabhatti',
        'Kurla', 'Tilaknagar', 'Chembur', 'Govandi', 'Mankhurd', 'Vashi',
        'Sanpada', 'Juinagar', 'Nerul', 'Seawood Darave', 'Belapur CBD',
        'Kharghar', 'Mansarovar', 'Khandeshwar', 'Panvel'
    ],
}

# Routing costs are distance-like, so mode changes need an explicit penalty.
# Otherwise a bus stop a few metres from a railway station looks artificially
# cheaper than staying on a direct train line.
BUS_TRANSFER_RADIUS_KM = 0.35
BUS_TRANSFER_PENALTY_KM = 3.0
OTHER_TRANSFER_RADIUS_KM = 0.40
OTHER_TRANSFER_PENALTY_KM = 0.75


def norm_text(value):
    return ''.join(
        ch.lower()
        for ch in str(value)
        if ch.isalnum()
    )


def estimate_local_coordinate(
    station_name,
    valid_rows
):
    if station_name in LOCAL_COORDS:
        return LOCAL_COORDS[station_name]

    target = norm_text(station_name)

    if not target:
        return None

    candidates = []

    for _, row in valid_rows.iterrows():

        candidate_name = str(
            row.get(
                'Station_Name',
                ''
            )
        ).strip()

        candidate = norm_text(
            candidate_name
        )

        if not candidate:
            continue

        score = 0.0

        if target in candidate or candidate in target:
            score += 1.0

        score += (
            SequenceMatcher(
                None,
                target,
                candidate
            ).ratio()
            * 0.7
        )

        if 'station' in candidate:
            score += 0.15

        if 'railway' in candidate:
            score += 0.15

        if score >= 0.75:

            candidates.append(
                (
                    score,
                    float(row['latitude']),
                    float(row['longitude'])
                )
            )

    if candidates:

        candidates.sort(
            reverse=True
        )

        top = candidates[:8]

        return (
            float(
                np.mean(
                    [x[1] for x in top]
                )
            ),
            float(
                np.mean(
                    [x[2] for x in top]
                )
            )
        )

    return None


def fill_local_coordinates(
    df_transit
):

    df = df_transit.copy()

    valid_rows = df.dropna(
        subset=[
            'latitude',
            'longitude'
        ]
    ).copy()

    local_mask = (
        df['Transport_Mode']
        .astype(str)
        .str.strip()
        .str.lower()
        .eq('local train')
    )

    for idx in df[local_mask].index:

        name = str(
            df.at[
                idx,
                'Station_Name'
            ]
        ).strip()

        coords = estimate_local_coordinate(
            name,
            valid_rows
        )

        if coords is not None:

            df.at[
                idx,
                'latitude'
            ] = coords[0]

            df.at[
                idx,
                'longitude'
            ] = coords[1]

    return df


# ============================================================
# GTFS BUS HELPERS
# ============================================================

def load_gtfs_bus_data():
    """Load compact GTFS bus stops and one ordered trip sequence per route."""
    stops_path = os.path.join(DATA_DIR, 'stops.txt')
    sequences_path = os.path.join(DATA_DIR, 'bus_route_sequences.csv')

    if not os.path.exists(stops_path):
        raise FileNotFoundError(f'Missing GTFS stops file: {stops_path}')
    if not os.path.exists(sequences_path):
        raise FileNotFoundError(f'Missing GTFS sequence file: {sequences_path}')

    stops = pd.read_csv(stops_path, encoding='utf-8-sig')
    sequences = pd.read_csv(sequences_path, encoding='utf-8-sig')

    required_stops = {'stop_id', 'stop_name', 'stop_lat', 'stop_lon'}
    required_sequences = {'route_id', 'direction_id', 'trip_id', 'stop_sequence', 'stop_id'}
    if not required_stops.issubset(stops.columns):
        raise ValueError(f'stops.txt is missing columns: {required_stops - set(stops.columns)}')
    if not required_sequences.issubset(sequences.columns):
        raise ValueError('bus_route_sequences.csv has an unexpected format')

    stops = stops.rename(columns={'stop_lat': 'latitude', 'stop_lon': 'longitude'})
    stops = stops[['stop_id', 'stop_name', 'latitude', 'longitude']].dropna(
        subset=['stop_id', 'latitude', 'longitude']
    ).copy()
    stops['stop_id'] = stops['stop_id'].astype(str).str.strip()
    stops['stop_name'] = stops['stop_name'].fillna('').astype(str).str.strip()
    stops['latitude'] = pd.to_numeric(stops['latitude'], errors='coerce')
    stops['longitude'] = pd.to_numeric(stops['longitude'], errors='coerce')
    stops = stops.dropna(subset=['latitude', 'longitude'])

    sequences = sequences.dropna(
        subset=['route_id', 'direction_id', 'trip_id', 'stop_sequence', 'stop_id']
    ).copy()
    sequences['route_id'] = sequences['route_id'].astype(str).str.strip()
    sequences['direction_id'] = sequences['direction_id'].astype(str).str.strip()
    sequences['trip_id'] = sequences['trip_id'].astype(str).str.strip()
    sequences['stop_id'] = sequences['stop_id'].astype(str).str.strip()
    sequences['stop_sequence'] = pd.to_numeric(
        sequences['stop_sequence'], errors='coerce'
    )
    sequences = sequences.dropna(subset=['stop_sequence'])
    sequences['stop_sequence'] = sequences['stop_sequence'].astype(int)

    return stops, sequences


def add_directed_edge(graph, a, b, cost):
    if a == b:
        return
    graph.setdefault(a, [])
    graph.setdefault(b, [])
    edge = (b, float(cost))
    if edge not in graph[a]:
        graph[a].append(edge)


# ============================================================
# 4. GRAPH HELPERS
# ============================================================

def add_edge(
    graph,
    a,
    b,
    cost
):

    if a == b:
        return

    graph[a].append(
        (
            b,
            float(cost)
        )
    )

    graph[b].append(
        (
            a,
            float(cost)
        )
    )


def connect_line_in_source_order(
    df,
    graph,
    nodes,
    node_rows,
    mode_filter=None
):

    work = df.copy()

    if mode_filter is not None:

        work = work[
            work['Transport_Mode']
            .astype(str)
            .str.strip()
            .eq(mode_filter)
        ]

    work = work.dropna(
        subset=[
            'latitude',
            'longitude'
        ]
    )

    if 'Line' not in work.columns:
        return

    for line_name, group in work.groupby(
        work['Line']
        .astype(str)
        .str.strip(),
        sort=False
    ):

        previous = None

        for idx in group.index:

            node = node_rows.get(idx)

            if node is None:
                continue

            if previous is not None:

                distance_km = euclidean_distance(
                    previous,
                    node,
                    nodes
                )

                distance_km = max(
                    distance_km,
                    0.05
                )

                add_edge(
                    graph,
                    previous,
                    node,
                    distance_km
                )

            previous = node


def _normalized_line(value):
    text = norm_text(value)
    for suffix in ('line', 'railway'):
        if text.endswith(suffix):
            text = text[:-len(suffix)]
    return text


def _canonical_station_key(value):
    key = norm_text(value)
    aliases = {
        'seawoods': 'seawooddarave',
        'seawood': 'seawooddarave',
        'seawooddarave': 'seawooddarave',
        'seawoodsdarave': 'seawooddarave',
        'tilaknagar': 'tilaknagar',
        'tilaknagarl': 'tilaknagar',
        'wadalaroad': 'wadalaroad',
        'vadalaroad': 'wadalaroad',
        'dockyardroad': 'dockyaradroad',
        'dockyaradroad': 'dockyaradroad',
        'cbd belapur': 'belapurcbd',
        'cbdbelapur': 'belapurcbd',
        'gtbnagar': 'gtbnagar'
    }
    return aliases.get(key, key)


def connect_local_train_lines(
    df,
    graph,
    nodes,
    node_rows
):
    """Connect Local Train stations using explicit physical line topology.

    The CSV's row order is not railway order.  More importantly, one global
    station list must never be reused as the order for every railway line.
    That can create bogus shortcuts and can make the router leave a train for
    a bus simply because the real railway edges are missing.
    """
    all_by_station = {}
    by_line = {}

    for idx, row in df.iterrows():
        node = node_rows.get(idx)
        if node is None:
            continue
        if str(row.get('Transport_Mode', '')).strip() != 'Local Train':
            continue

        line = _normalized_line(row.get('Line', ''))
        label = str(row.get('Station_Name', '')).strip()
        station_key = _canonical_station_key(label)
        if not station_key:
            continue

        record = (node, line, label)
        all_by_station.setdefault(station_key, []).append(record)
        if line:
            by_line.setdefault(line, []).append(record)

    topology_station_keys = set()

    # Connect known lines only according to their own physical station order.
    for topology_line, station_names in LOCAL_TRAIN_TOPOLOGY.items():
        previous = None

        for station_name in station_names:
            station_key = _canonical_station_key(station_name)
            topology_station_keys.add(station_key)
            candidates = by_line.get(topology_line, [])
            candidates = [r for r in candidates if _canonical_station_key(r[2]) == station_key]

            # Interchange fallback: datasets sometimes list Kurla only once
            # and tag it as Central even though the physical station is also
            # where the Harbour route continues.  Reuse the same physical node
            # instead of forcing a fake bus transfer.
            if not candidates:
                candidates = all_by_station.get(station_key, [])

            if not candidates:
                continue

            target_coord = LOCAL_COORDS.get(station_name)
            if target_coord is not None:
                current = min(
                    candidates,
                    key=lambda r: point_distance(nodes[r[0]], target_coord)
                )[0]
            else:
                current = candidates[0][0]

            if previous is not None and previous != current:
                add_edge(
                    graph,
                    previous,
                    current,
                    max(euclidean_distance(previous, current, nodes), 0.05)
                )
            previous = current

    # Preserve source order only for lines/stations that are not covered by a
    # curated topology.  This keeps other local-train data usable without
    # allowing known Harbour stations to be connected in an arbitrary order.
    for line, records in by_line.items():
        unknown = [
            record[0]
            for record in records
            if _canonical_station_key(record[2]) not in topology_station_keys
        ]
        for a, b in zip(unknown, unknown[1:]):
            add_edge(
                graph,
                a,
                b,
                max(euclidean_distance(a, b, nodes), 0.05)
            )

def connect_same_station_transfers(
    df,
    graph,
    nodes,
    node_rows
):

    groups = {}

    for idx, row in df.iterrows():

        node = node_rows.get(idx)

        if node is None:
            continue

        name = norm_text(
            row.get(
                'Station_Name',
                ''
            )
        )

        if not name:
            continue

        groups.setdefault(
            name,
            []
        ).append(node)

    for _, group in groups.items():

        unique = list(
            dict.fromkeys(group)
        )

        if len(unique) < 2:
            continue

        for i in range(
            len(unique)
        ):

            for j in range(
                i + 1,
                len(unique)
            ):

                if (
                    unique[i] in nodes
                    and unique[j] in nodes
                ):

                    # Only connect duplicate station records when they are
                    # physically close. Some bus datasets reuse the same
                    # stop name in different parts of Mumbai; treating all
                    # same-name records as a zero-cost transfer creates
                    # impossible teleports through the graph.
                    distance_km = point_distance(
                        nodes[unique[i]],
                        nodes[unique[j]]
                    )

                    if distance_km <= 0.35:
                        add_edge(
                            graph,
                            unique[i],
                            unique[j],
                            0.15
                        )


def build_graph(
    df_wards,
    df_transit,
    gtfs_stops,
    gtfs_sequences
):

    nodes = {}
    graph = {}
    metadata = {}

    # --------------------------------------------------------
    # WARD NODES
    # --------------------------------------------------------

    for i, row in df_wards.iterrows():

        ward = str(
            row['Ward_Alphabet']
        )

        node = f'W_{ward}'

        nodes[node] = (
            float(row['latitude']),
            float(row['longitude'])
        )

        graph[node] = []

        metadata[node] = {
            'label': f'Ward {ward}',
            'mode': 'Ward',
            'line': ''
        }

    # --------------------------------------------------------
    # TRANSPORT NODES
    # --------------------------------------------------------

    node_rows = {}

    for idx, row in df_transit.iterrows():

        # Bus stops are loaded from GTFS below. Keeping CSV bus nodes here
        # would reintroduce the old approximate bus network.
        if str(row.get('Transport_Mode', '')).strip() == 'Bus':
            continue

        if (
            pd.isna(row['latitude'])
            or pd.isna(row['longitude'])
        ):
            continue

        node = f'T_{idx}'

        node_rows[idx] = node

        nodes[node] = (
            float(row['latitude']),
            float(row['longitude'])
        )

        graph[node] = []

        metadata[node] = {
            'label': str(
                row['Station_Name']
            ).strip(),

            'mode': str(
                row.get(
                    'Transport_Mode',
                    'Transit'
                )
            ).strip(),

            'line': str(
                row.get(
                    'Line',
                    ''
                )
            ).strip()
        }

    # --------------------------------------------------------
    # REAL GTFS BUS STOP NODES
    # --------------------------------------------------------
    stop_id_to_node = {}

    for _, row in gtfs_stops.iterrows():
        stop_id = str(row['stop_id']).strip()
        if not stop_id or stop_id in stop_id_to_node:
            continue

        node = f'B_{stop_id}'
        nodes[node] = (float(row['latitude']), float(row['longitude']))
        graph[node] = []
        metadata[node] = {
            'label': str(row['stop_name']).strip() or stop_id,
            'mode': 'Bus',
            'line': '',
            'stop_id': stop_id
        }
        stop_id_to_node[stop_id] = node

    ward_nodes = [
        n for n in nodes
        if n.startswith('W_')
    ]

    transport_nodes = [
        n for n in nodes
        if n.startswith('T_') or n.startswith('B_')
    ]

    # --------------------------------------------------------
    # REAL LINE CONNECTIONS
    # --------------------------------------------------------

    # Local Train edges are handled separately because the CSV row
    # order is not a reliable representation of railway order.
    # Metro/Monorail can continue using their source ordering.
    connect_local_train_lines(
        df_transit,
        graph,
        nodes,
        node_rows
    )

    for mode in [
        'Metro',
        'Monorail'
    ]:
        connect_line_in_source_order(
            df_transit,
            graph,
            nodes,
            node_rows,
            mode_filter=mode
        )

    # --------------------------------------------------------
    # SAME STATION TRANSFERS
    # --------------------------------------------------------

    connect_same_station_transfers(
        df_transit,
        graph,
        nodes,
        node_rows
    )

    # --------------------------------------------------------
    # WARD -> TRANSPORT CONNECTIONS
    # --------------------------------------------------------

    if transport_nodes:

        transport_coordinates = np.array(
            [
                nodes[n]
                for n in transport_nodes
            ]
        )

        transport_tree = cKDTree(
            transport_coordinates
        )

        for ward_node in ward_nodes:

            ward_coord = np.array(
                nodes[ward_node]
            )

            k = min(
                5,
                len(transport_nodes)
            )

            distances, indices = (
                transport_tree.query(
                    ward_coord,
                    k=k
                )
            )

            for distance, transport_index in zip(
                np.atleast_1d(distances),
                np.atleast_1d(indices)
            ):

                transport_node = (
                    transport_nodes[
                        int(transport_index)
                    ]
                )

                distance_km = (
                    float(distance)
                    * 111.0
                )

                add_edge(
                    graph,
                    ward_node,
                    transport_node,
                    distance_km
                )

    # --------------------------------------------------------
    # REAL GTFS BUS ROUTE CONNECTIONS
    # --------------------------------------------------------
    # Only consecutive stops in the GTFS trip sequence are connected.
    # This replaces the old nearest-neighbour geographic bus mesh.
    bus_edge_routes = {}
    bus_edges_added = 0

    for route_key, route_stops in gtfs_sequences.groupby(
        ['route_id', 'direction_id', 'trip_id'], sort=False
    ):
        route_id, direction_id, trip_id = route_key
        route_stops = route_stops.sort_values('stop_sequence')

        ordered_nodes = [
            stop_id_to_node.get(str(stop_id).strip())
            for stop_id in route_stops['stop_id']
        ]
        ordered_nodes = [node for node in ordered_nodes if node is not None]

        for previous, current in zip(ordered_nodes, ordered_nodes[1:]):
            distance_km = max(
                euclidean_distance(previous, current, nodes),
                0.05
            )
            add_directed_edge(graph, previous, current, distance_km)
            bus_edge_routes.setdefault((previous, current), set()).add(str(route_id))
            bus_edges_added += 1

    # Make route lists deterministic and JSON/cache friendly.
    bus_edge_routes = {
        key: tuple(sorted(value))
        for key, value in bus_edge_routes.items()
    }

    print(f'Real GTFS bus connections added: {bus_edges_added}')

    # --------------------------------------------------------
    # SHORT WALKING / TRANSFER CONNECTIONS
    # --------------------------------------------------------

    if transport_nodes:

        coords = np.array(
            [
                nodes[n]
                for n in transport_nodes
            ]
        )

        tree = cKDTree(coords)

        k = min(
            5,
            len(transport_nodes) - 1
        ) if len(transport_nodes) > 1 else 0

        if k:

            distances, indices = (
                tree.query(
                    coords,
                    k=k + 1
                )
            )

            for i, node in enumerate(
                transport_nodes
            ):

                for distance, j in zip(
                    np.atleast_1d(
                        distances[i]
                    )[1:],

                    np.atleast_1d(
                        indices[i]
                    )[1:]
                ):

                    neighbour = (
                        transport_nodes[
                            int(j)
                        ]
                    )

                    mode_a = metadata[
                        node
                    ]['mode']

                    mode_b = metadata[
                        neighbour
                    ]['mode']

                    # Bus-to-bus travel must come only from actual GTFS
                    # consecutive-stop sequences, never geographic proximity.
                    if mode_a == 'Bus' and mode_b == 'Bus':
                        continue

                    # Prevent fake railway shortcuts.
                    if (
                        mode_a == 'Local Train'
                        and mode_b == 'Local Train'
                    ):
                        continue

                    if (
                        mode_a == 'Metro'
                        and mode_b == 'Metro'
                    ):
                        continue

                    if (
                        mode_a == 'Monorail'
                        and mode_b == 'Monorail'
                    ):
                        continue

                    distance_km = float(distance) * 111.0

                    # Bus interchanges are allowed only when the stops are
                    # genuinely close, and they carry a boarding/transfer
                    # penalty.  This stops Kurla -> Seawoods from abandoning
                    # a continuous Harbour local-train path just to chase a
                    # nearby bus stop.
                    if 'Bus' in (mode_a, mode_b):
                        if distance_km <= BUS_TRANSFER_RADIUS_KM:
                            add_edge(
                                graph,
                                node,
                                neighbour,
                                distance_km + BUS_TRANSFER_PENALTY_KM
                            )
                    elif distance_km <= OTHER_TRANSFER_RADIUS_KM:
                        add_edge(
                            graph,
                            node,
                            neighbour,
                            distance_km + OTHER_TRANSFER_PENALTY_KM
                        )

    return (
        graph,
        nodes,
        transport_nodes,
        metadata,
        node_rows,
        bus_edge_routes
    )


# ============================================================
# 5. STATION LOOKUP
# ============================================================

def build_station_lookup(
    df_transit,
    node_rows,
    nodes=None,
    metadata=None
):
    """Build searchable locations from CSV transit plus real GTFS bus stops."""
    station_lookup = {}
    priority = {
        'Local Train': 0,
        'Metro': 1,
        'Monorail': 2,
        'Bus': 3
    }

    rows = []
    for idx, row in df_transit.iterrows():
        node = node_rows.get(idx)
        if node is None:
            continue
        name = str(row.get('Station_Name', '')).strip()
        if not name:
            continue
        mode = str(row.get('Transport_Mode', 'Transit')).strip()
        rows.append((priority.get(mode, 9), str(name).lower(), name, node, mode, str(row.get('Line', '')).strip()))

    if nodes is not None and metadata is not None:
        for node, info in metadata.items():
            if info.get('mode') != 'Bus':
                continue
            name = str(info.get('label', '')).strip()
            if not name or node not in nodes:
                continue
            rows.append((priority['Bus'], name.lower(), name, node, 'Bus', ''))

    rows.sort(key=lambda x: (x[0], x[1], x[2]))
    for _, _, name, node, mode, line in rows:
        # Keep the first occurrence for display, preserving the mode priority.
        if name in station_lookup:
            continue
        station_lookup[name] = {
            'node': node,
            'latitude': float(nodes[node][0]) if nodes is not None else None,
            'longitude': float(nodes[node][1]) if nodes is not None else None,
            'mode': mode,
            'line': line
        }

    return station_lookup


def node_label(
    node,
    node_to_station,
    metadata
):

    if node in node_to_station:
        return node_to_station[node]

    if str(node).startswith('W_'):
        return (
            'Ward '
            + str(node)[2:]
        )

    return metadata.get(
        node,
        {}
    ).get(
        'label',
        str(node)
    )


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
   We use a harmonic mean between optimality and efficiency.

   This prevents an algorithm from winning simply because it
   is extremely fast to search but produces poor routes.

   Final Score =
       Success Rate × Harmonic Mean(Optimality, Efficiency)

Execution time is DISPLAYED but not used in the final ranking,
because machine CPU load can make microsecond timings unstable.

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


def create_benchmark_pairs(
    station_lookup,
    number_of_pairs=60
):

    # --------------------------------------------------------
    # Unique station nodes
    # --------------------------------------------------------

    unique_nodes = {}

    for name, info in station_lookup.items():

        node = info['node']

        if node not in unique_nodes:

            unique_nodes[node] = {
                'name': name,
                'node': node,
                'mode': info.get(
                    'mode',
                    ''
                )
            }

    stations = list(
        unique_nodes.values()
    )

    if len(stations) < 2:
        return []

    # --------------------------------------------------------
    # FIXED SEED
    # --------------------------------------------------------

    rng = random.Random(42)

    pairs = []
    used = set()

    max_attempts = (
        number_of_pairs * 20
    )

    attempts = 0

    while (
        len(pairs) < number_of_pairs
        and attempts < max_attempts
    ):

        attempts += 1

        a, b = rng.sample(
            stations,
            2
        )

        key = (
            a['node'],
            b['node']
        )

        reverse_key = (
            b['node'],
            a['node']
        )

        if (
            key in used
            or reverse_key in used
        ):
            continue

        used.add(key)

        pairs.append(
            (
                a['node'],
                b['node'],
                a['name'],
                b['name']
            )
        )

    return pairs


def evaluate_algorithms(
    graph,
    nodes,
    station_lookup,
    number_of_pairs=60
):

    print(
        f'Benchmarking algorithms on '
        f'{number_of_pairs} fixed routes...'
    )

    pairs = create_benchmark_pairs(
        station_lookup,
        number_of_pairs
    )

    if not pairs:
        return pd.DataFrame(), []

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    metrics = {
        name: {
            'successes': 0,
            'optimality_values': [],
            'efficiency_values': [],
            'execution_times': [],
            'route_costs': [],
            'expanded_nodes': []
        }
        for name in ALGORITHMS
    }

    valid_benchmarks = 0

    # --------------------------------------------------------
    # Run every algorithm on every pair
    # --------------------------------------------------------

    for pair_number, (
        start,
        destination,
        start_name,
        destination_name
    ) in enumerate(pairs, 1):

        # ----------------------------------------------------
        # UCS ground truth for route optimality
        # ----------------------------------------------------

        ucs_path, ucs_cost, ucs_expanded = (
            uniform_cost_search(
                graph,
                start,
                {destination},
                nodes
            )
        )

        # If even UCS cannot connect these nodes,
        # don't use this pair as a benchmark.
        if (
            ucs_path is None
            or not np.isfinite(ucs_cost)
            or ucs_cost <= 0
        ):
            continue

        valid_benchmarks += 1

        # ----------------------------------------------------
        # Run every algorithm first. Efficiency must be scored
        # only AFTER all algorithms have been measured, otherwise
        # anything that beats UCS gets incorrectly capped at 1.0.
        # ----------------------------------------------------

        pair_results = {}

        for algorithm_name, algorithm in ALGORITHMS.items():

            started = time.perf_counter()

            path, cost, expanded = algorithm(
                graph,
                start,
                {destination},
                nodes
            )

            elapsed = (
                time.perf_counter()
                - started
            )

            metrics[
                algorithm_name
            ]['execution_times'].append(
                elapsed
            )

            # Treat only valid finite positive-cost routes as
            # successful benchmark results.
            if (
                path is None
                or not np.isfinite(cost)
                or cost <= 0
            ):
                continue

            expanded = int(expanded)

            pair_results[algorithm_name] = {
                'cost': float(cost),
                'expanded': expanded
            }

            metrics[
                algorithm_name
            ]['successes'] += 1

            metrics[
                algorithm_name
            ]['route_costs'].append(
                float(cost)
            )

            metrics[
                algorithm_name
            ]['expanded_nodes'].append(
                expanded
            )

            # ------------------------------------------------
            # Optimality: UCS remains the ground truth.
            # ------------------------------------------------

            optimality = ucs_cost / cost

            optimality = max(
                0.0,
                min(
                    1.0,
                    optimality
                )
            )

            metrics[
                algorithm_name
            ]['optimality_values'].append(
                optimality
            )

        # ----------------------------------------------------
        # Search efficiency
        # ----------------------------------------------------
        # Compare every successful algorithm against the least
        # number of nodes expanded on THIS SAME route.
        #
        # best_efficiency = min_expanded / algorithm_expanded
        #
        # This guarantees that only the most search-efficient
        # algorithm(s) on a route receive 1.0. UCS no longer gets
        # an automatic perfect score just for being the optimality
        # reference.
        # ----------------------------------------------------

        positive_expanded = [
            result['expanded']
            for result in pair_results.values()
            if result['expanded'] > 0
        ]

        if positive_expanded:

            best_expanded = min(
                positive_expanded
            )

            for algorithm_name, result in pair_results.items():

                expanded = result['expanded']

                efficiency = (
                    best_expanded / expanded
                    if expanded > 0
                    else 0.0
                )

                efficiency = max(
                    0.0,
                    min(
                        1.0,
                        efficiency
                    )
                )

                metrics[
                    algorithm_name
                ]['efficiency_values'].append(
                    efficiency
                )

        else:

            # Keep the arrays aligned with successful searches if
            # a search reports zero expanded nodes.
            for algorithm_name in pair_results:
                metrics[
                    algorithm_name
                ]['efficiency_values'].append(
                    0.0
                )

    # --------------------------------------------------------
    # Convert benchmark data to final scores
    # --------------------------------------------------------

    results = []

    for algorithm_name in ALGORITHMS:

        data = metrics[
            algorithm_name
        ]

        if valid_benchmarks == 0:

            success_rate = 0.0
            optimality = 0.0
            efficiency = 0.0
            final_score = 0.0

        else:

            success_rate = (
                data['successes']
                / valid_benchmarks
            )

            optimality = (
                float(
                    np.mean(
                        data[
                            'optimality_values'
                        ]
                    )
                )
                if data[
                    'optimality_values'
                ]
                else 0.0
            )

            efficiency = (
                float(
                    np.mean(
                        data[
                            'efficiency_values'
                        ]
                    )
                )
                if data[
                    'efficiency_values'
                ]
                else 0.0
            )

            quality_efficiency = (
                harmonic_mean(
                    optimality,
                    efficiency
                )
            )

            final_score = (
                success_rate
                * quality_efficiency
            )

        avg_time = (
            float(
                np.mean(
                    data[
                        'execution_times'
                    ]
                )
            )
            if data[
                'execution_times'
            ]
            else 0.0
        )

        avg_cost = (
            float(
                np.mean(
                    data[
                        'route_costs'
                    ]
                )
            )
            if data[
                'route_costs'
            ]
            else float('inf')
        )

        avg_expanded = (
            float(
                np.mean(
                    data[
                        'expanded_nodes'
                    ]
                )
            )
            if data[
                'expanded_nodes'
            ]
            else float('inf')
        )

        results.append(
            {
                'algorithm': algorithm_name,

                'success_rate':
                    success_rate,

                'optimality':
                    optimality,

                'efficiency':
                    efficiency,

                'final_score':
                    final_score,

                'average_path_cost':
                    avg_cost,

                'average_nodes_expanded':
                    avg_expanded,

                'average_execution_time':
                    avg_time,

                'successful_searches':
                    data['successes'],

                'benchmark_routes':
                    valid_benchmarks
            }
        )

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Deterministic ordering
    # --------------------------------------------------------

    algorithm_order = {
        'A* Search': 0,
        'Uniform Cost Search': 1,
        'BFS': 2,
        'Greedy Best First Search': 3,
        'DFS': 4
    }

    results_df['_order'] = (
        results_df['algorithm']
        .map(algorithm_order)
    )

    results_df = results_df.sort_values(
        by=[
            'final_score',
            'optimality',
            'efficiency',
            'success_rate',
            '_order'
        ],
        ascending=[
            False,
            False,
            False,
            False,
            True
        ]
    )

    results_df = results_df.drop(
        columns=['_order']
    )

    best_two = (
        results_df
        .head(2)
        ['algorithm']
        .tolist()
    )

    print('\n========== ALGORITHM EVALUATION ==========')

    for _, row in results_df.iterrows():

        print(
            f"{row['algorithm']}: "
            f"Score={row['final_score']:.3f}, "
            f"Optimality={row['optimality']:.3f}, "
            f"Efficiency={row['efficiency']:.3f}, "
            f"Success={row['success_rate']:.3f}"
        )

    print(
        f'\nBEST 2: {best_two}'
    )

    print(
        '===========================================\n'
    )

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

                'lat': float(
                    nodes[node][0]
                ),

                'lon': float(
                    nodes[node][1]
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
                    'Network benchmark using '
                    'success rate, route optimality '
                    'and search efficiency.'
            }
        )

    if not results:

        return (
            None,
            'No route could be found between '
            'these two locations.'
        )

    # --------------------------------------------------------
    # Find UCS result as route-level optimal reference
    # --------------------------------------------------------

    ucs_result = next(
        (
            r for r in results
            if r['algorithm']
            == 'Uniform Cost Search'
        ),
        None
    )

    if ucs_result:

        ucs_cost = ucs_result[
            'path_cost'
        ]

        ucs_nodes = ucs_result[
            'nodes'
        ]

        for result in results:

            if result['path_cost'] > 0:

                optimality = (
                    ucs_cost
                    / result['path_cost']
                )

            else:
                optimality = 1.0

            optimality = max(
                0.0,
                min(
                    1.0,
                    optimality
                )
            )

            if result['nodes'] > 0:

                efficiency = (
                    ucs_nodes
                    / result['nodes']
                )

            else:

                efficiency = 0.0

            efficiency = max(
                0.0,
                min(
                    1.0,
                    efficiency
                )
            )

            result[
                'route_optimality'
            ] = round(
                optimality,
                3
            )

            result[
                'route_efficiency'
            ] = round(
                efficiency,
                3
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

def calculate_accessibility(
    df_wards,
    df_transit,
    graph,
    nodes,
    transport_nodes,
    best_algorithms
):

    valid_transit = df_transit.dropna(
        subset=[
            'latitude',
            'longitude'
        ]
    )

    transit_coords = (
        valid_transit[
            [
                'latitude',
                'longitude'
            ]
        ].values
    )

    ward_coords = (
        df_wards[
            [
                'latitude',
                'longitude'
            ]
        ].values
    )

    if len(transit_coords):

        tree = cKDTree(
            transit_coords
        )

        distances, _ = tree.query(
            ward_coords
        )

        df_wards[
            'distance_km'
        ] = distances * 111.0

    else:

        df_wards[
            'distance_km'
        ] = 0.0

    df_wards[
        'Accessibility_Gap_Score'
    ] = (
        df_wards['distance_km']
        * df_wards['TOT_P_DEN']
    )

    df_poor = (
        df_wards[
            df_wards['distance_km']
            > 0.30
        ]
        .sort_values(
            'distance_km',
            ascending=False
        )
    )

    poor_access = [

        {
            'ward':
                row['Ward_Alphabet'],

            'name':
                row['Ward_Names'],

            'distance':
                round(
                    row['distance_km'],
                    2
                )
        }

        for _, row
        in df_poor.iterrows()
    ]

    algorithm_scores = {}

    for algorithm_name in best_algorithms:

        algorithm = (
            ALGORITHMS[
                algorithm_name
            ]
        )

        ward_results = []

        for _, row in df_wards.iterrows():

            ward_node = (
                f"W_"
                f"{row['Ward_Alphabet']}"
            )

            path, cost, expanded = (
                algorithm(
                    graph,
                    ward_node,
                    set(transport_nodes),
                    nodes
                )
            )

            if path is not None:

                ward_results.append(
                    {
                        'ward':
                            row[
                                'Ward_Alphabet'
                            ],

                        'path_cost':
                            cost,

                        'nodes_expanded':
                            expanded
                    }
                )

        algorithm_scores[
            algorithm_name
        ] = pd.DataFrame(
            ward_results
        )

    df_final = df_wards.copy()

    for algorithm_name in best_algorithms:

        result = algorithm_scores[
            algorithm_name
        ]

        if result.empty:
            continue

        df_final = df_final.merge(

            result.rename(
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

    cost_columns = [

        f'{algorithm}_cost'

        for algorithm
        in best_algorithms

        if f'{algorithm}_cost'
        in df_final.columns
    ]

    if cost_columns:

        df_final[
            'Search_Average_Cost'
        ] = (
            df_final[
                cost_columns
            ]
            .mean(axis=1)
            .fillna(0)
        )

    else:

        df_final[
            'Search_Average_Cost'
        ] = 0

    df_final[
        'Final_Priority_Score'
    ] = (
        df_final[
            'Accessibility_Gap_Score'
        ]
        *
        (
            1
            +
            df_final[
                'Search_Average_Cost'
            ]
        )
    )

    df_priority = (
        df_final
        .sort_values(
            'Final_Priority_Score',
            ascending=False
        )
        .head(5)
    )

    prioritized = [

        {
            'ward':
                row[
                    'Ward_Alphabet'
                ],

            'name':
                row['Ward_Names'],

            'distance':
                round(
                    row['distance_km'],
                    2
                ),

            'score':
                round(
                    row[
                        'Final_Priority_Score'
                    ],
                    2
                )
        }

        for _, row
        in df_priority.iterrows()
    ]

    return (
        poor_access,
        prioritized
    )


# ============================================================
# 12. PROCESS DATA
# ============================================================

def process_transit_data():

    wards_path = os.path.join(
        DATA_DIR,
        'ward_level_collated.csv'
    )

    transit_path = os.path.join(
        DATA_DIR,
        'unified_mumbai_transport.csv'
    )

    gtfs_stops_path = os.path.join(DATA_DIR, 'stops.txt')
    gtfs_sequences_path = os.path.join(DATA_DIR, 'bus_route_sequences.csv')

    # Fast path: load all processed application state from disk.
    cached = _load_processed_cache(
        wards_path,
        transit_path,
        gtfs_stops_path,
        gtfs_sequences_path
    )

    if cached is not None:
        return cached

    # Slow path: process raw datasets once.
    df_wards = pd.read_csv(
        wards_path
    )

    df_transit = pd.read_csv(
        transit_path
    )

    df_transit = df_transit.rename(
        columns={
            'Latitude':
                'latitude',

            'Longitude':
                'longitude'
        }
    )

    # IMPORTANT:
    # Recover Local Train coordinates BEFORE dropping NaN.
    df_transit = (
        fill_local_coordinates(
            df_transit
        )
    )

    df_transit = (
        df_transit
        .dropna(
            subset=[
                'latitude',
                'longitude'
            ]
        )
        .copy()
    )

    df_wards = (
        df_wards
        .dropna(
            subset=[
                'Ward_Alphabet',
                'TOT_P_DEN'
            ]
        )
        .copy()
    )

    # --------------------------------------------------------
    # Mumbai ward coordinates
    # --------------------------------------------------------

    mumbai_ward_coords = {

        'A':
            (18.9220, 72.8347),

        'B':
            (18.9548, 72.8377),

        'C':
            (18.9449, 72.8259),

        'D':
            (18.9647, 72.8130),

        'E':
            (18.9696, 72.8423),

        'F/N':
            (19.0238, 72.8550),

        'F/S':
            (19.0014, 72.8452),

        'G/N':
            (19.0330, 72.8475),

        'G/S':
            (19.0103, 72.8262),

        'H/E':
            (19.0700, 72.8468),

        'H/W':
            (19.0657, 72.8310),

        'K/E':
            (19.1136, 72.8697),

        'K/W':
            (19.1197, 72.8464),

        'L':
            (19.0759, 72.8877),

        'M/E':
            (19.0473, 72.9158),

        'M/W':
            (19.0596, 72.8958),

        'N':
            (19.1417, 72.9331),

        'P/N':
            (19.1874, 72.8484),

        'P/S':
            (19.1551, 72.8464),

        'R/C':
            (19.2215, 72.8556),

        'R/N':
            (19.2804, 72.8597),

        'R/S':
            (19.2094, 72.8126),

        'S':
            (19.1306, 72.9375),

        'T':
            (19.1735, 72.9495)
    }

    df_wards['latitude'] = (
        df_wards[
            'Ward_Alphabet'
        ].map(
            lambda x:
            mumbai_ward_coords.get(
                x,
                (
                    19.0760,
                    72.8777
                )
            )[0]
        )
    )

    df_wards['longitude'] = (
        df_wards[
            'Ward_Alphabet'
        ].map(
            lambda x:
            mumbai_ward_coords.get(
                x,
                (
                    19.0760,
                    72.8777
                )
            )[1]
        )
    )

    # --------------------------------------------------------
    # LOAD REAL GTFS BUS DATA
    # --------------------------------------------------------
    gtfs_stops, gtfs_sequences = load_gtfs_bus_data()

    # --------------------------------------------------------
    # BUILD GRAPH
    # --------------------------------------------------------

    (
        graph,
        nodes,
        transport_nodes,
        metadata,
        node_rows,
        bus_edge_routes
    ) = build_graph(
        df_wards,
        df_transit,
        gtfs_stops,
        gtfs_sequences
    )

    # --------------------------------------------------------
    # SEARCHABLE LOCATIONS
    # --------------------------------------------------------

    station_lookup = (
        build_station_lookup(
            df_transit,
            node_rows,
            nodes,
            metadata
        )
    )

    # --------------------------------------------------------
    # ALGORITHM BENCHMARK
    # --------------------------------------------------------
    # This is reached only when the processed cache is missing or stale.

    results_df, best_two = (
        evaluate_algorithms(
            graph,
            nodes,
            station_lookup,
            number_of_pairs=60
        )
    )

    # --------------------------------------------------------
    # ACCESSIBILITY
    # --------------------------------------------------------

    poor_access, prioritized = (
        calculate_accessibility(
            df_wards,
            df_transit,
            graph,
            nodes,
            transport_nodes,
            best_two
        )
    )

    # --------------------------------------------------------
    # FORMAT ALGORITHM RESULTS FOR HTML
    # --------------------------------------------------------

    algorithm_results = []

    for _, row in results_df.iterrows():

        algorithm_results.append(
            {

                'algorithm':
                    row['algorithm'],

                # Overall actual evaluation score
                'score':
                    round(
                        row['final_score'],
                        3
                    ),

                # Human-readable percentages
                'success_rate':
                    round(
                        row[
                            'success_rate'
                        ] * 100,
                        1
                    ),

                'optimality':
                    round(
                        row[
                            'optimality'
                        ] * 100,
                        1
                    ),

                'efficiency':
                    round(
                        row[
                            'efficiency'
                        ] * 100,
                        1
                    ),

                'path_cost':
                    round(
                        row[
                            'average_path_cost'
                        ],
                        3
                    ),

                'nodes':
                    round(
                        row[
                            'average_nodes_expanded'
                        ],
                        2
                    ),

                'time':
                    round(
                        row[
                            'average_execution_time'
                        ],
                        6
                    ),

                'successful_searches':
                    int(
                        row[
                            'successful_searches'
                        ]
                    ),

                'benchmark_routes':
                    int(
                        row[
                            'benchmark_routes'
                        ]
                    )
            }
        )

    station_groups = (
        build_station_groups(
            station_lookup
        )
    )

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
        bus_edge_routes
    )

    _save_processed_cache(
        processed_data,
        wards_path,
        transit_path,
        gtfs_stops_path,
        gtfs_sequences_path
    )

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
    cached_bus_edge_routes
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
            cached_station_groups
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
