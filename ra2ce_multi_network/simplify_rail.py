import geopandas as gpd
import numpy as np
import snkit.network

from snkit.network import *

from trails import *


def detect_possible_terminals(network_gdf: gpd.GeoDataFrame) -> snkit.network.Network:
    ## Inspired by drop_hanging_nodes in trails/simplify.py
    network = make_network_from_gdf(network_gdf=network_gdf)
    # hanging_nodes : An array of the indices of nodes with degree 1
    hanging_nodes = find_hanging_nodes(network=network)
    possible_terminals = []
    # Check for terminal criteria
    possible_terminals = network.edges[['from_id', 'to_id', 'service']].apply(
        lambda x: check_terminal_criteria(x['from_id'], x['to_id'], x['service'], hanging_nodes), axis=1
    ).dropna().tolist()
    network.nodes['possible_terminals'] = network.nodes['id'].apply(lambda x: 1 if x in possible_terminals else 0)
    return network


def make_network_from_gdf(network_gdf) -> snkit.network.Network:
    net = Network(edges=network_gdf)
    net = add_endpoints(network=net)
    net = split_edges_at_nodes(network=net)
    net = add_endpoints(network=net)
    net = add_ids(network=net)
    net = add_topology(network=net)
    net.set_crs(crs="EPSG:4326")
    net.nodes.id = net.nodes.id.str.extract(r"(\d+)", expand=False).astype(int)
    net.edges.from_id = net.edges.from_id.str.extract(r"(\d+)", expand=False).astype(int)
    net.edges.to_id = net.edges.to_id.str.extract(r"(\d+)", expand=False).astype(int)
    return net


def find_hanging_nodes(network) -> np.ndarray:
    if 'degree' not in network.nodes.columns:
        deg = calculate_degree(network)
    else:
        deg = network.nodes['degree'].to_numpy()
    return np.where(deg == 1)[0]


def calculate_degree(network):
    """Calculates the degree of the nodes from the from and to ids. It
    is not wise to call this method after removing nodes or edges
    without first resetting the ids

    from ra2ce_multi_network.trails.simplify import calculate_degree

    Args:
        network (class): A network composed of nodes (points in space) and edges (lines)

    Returns:
        Connectivity degree (numpy.array): [description]
    """
    # the number of nodes(from index) to use as the number of bins
    ndC = len(network.nodes.index)
    if ndC - 1 > max(network.edges.from_id) and ndC - 1 > max(network.edges.to_id): print(
        "Calculate_degree possibly unhappy")
    return np.bincount(network.edges['from_id'], None, ndC) + np.bincount(network.edges['to_id'], None, ndC)


def check_terminal_criteria(from_node_id, to_node_id, edge_property, hanging_nodes):
    if from_node_id in hanging_nodes and edge_property == 'spur':
        return from_node_id
    elif to_node_id in hanging_nodes and edge_property == 'spur':
        return to_node_id
