import geopandas as gpd
import numpy as np
import pandas as pd
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
    # Merge possible_terminals based on a range
    network.nodes = merge_terminal_nodes(network, range=0.0001)
    return network


def merge_terminal_nodes(network: snkit.network.Network, range: float) -> snkit.network.Network:
    # The following 2 functions are used in the body of merge_terminal_nodes function.
    # merge_terminal_nodes starts after the following 2 functions.
    def update_node_gdf(node_gdf: gpd.GeoDataFrame, terminal_collection: gpd.GeoDataFrame,
                        considered_node_ids: list):
        terminal_collection_ids = terminal_collection['id'].to_numpy().tolist()
        if len(terminal_collection_ids) == 1:
            return node_gdf
        else:

            if not any(node_id in considered_node_ids for node_id in terminal_collection_ids):
                aggregated_terminal_gdf = get_centroid_terminal_gdf(terminal_collection)
            else:
                terminal_to_aggr_again_gdf = node_gdf[node_gdf['terminal_collection'].apply(
                    lambda x: x is not None and any(node_id in x for node_id in terminal_collection_ids))]
                aggregated_terminal_gdf = get_centroid_terminal_gdf(terminal_to_aggr_again_gdf)
                aggregated_again = terminal_to_aggr_again_gdf[
                    terminal_to_aggr_again_gdf['terminal_collection'].apply(lambda x: len(x) > 1)]['id'].tolist()
                node_gdf = node_gdf[~node_gdf['id'].isin(aggregated_again)]
            considered_node_ids += terminal_collection_ids
            considered_node_ids = list(set(considered_node_ids))
            return pd.concat([node_gdf, aggregated_terminal_gdf], ignore_index=True)

    def get_centroid_terminal_gdf(_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        _centroid_terminal_collection = _gdf['geometry'].unary_union.centroid
        _terminal_collection_ids = _gdf['id'].tolist()
        _aggregated_terminal = Point(_centroid_terminal_collection.x, _centroid_terminal_collection.y)

        return gpd.GeoDataFrame({'id': new_id, 'geometry': _aggregated_terminal,
                                 'possible_terminals': 1,
                                 'terminal_collection': [
                                     {term_col for _id in _terminal_collection_ids
                                      for term_col in node_gdf.loc[
                                          node_gdf['id'] == _id, 'terminal_collection'
                                      ].values[0]}
                                 ],
                                 'buffer': _aggregated_terminal.buffer(range)},
                                crs=node_gdf.crs)

    # merge_terminal_nodes function starts here
    considered_node_ids = []
    node_gdf = network.nodes
    node_gdf['terminal_collection'] = node_gdf.apply(
        lambda row: {row['id']} if row['possible_terminals'] == 1 else None, axis=1)
    node_gdf['buffer'] = node_gdf.apply(
        lambda row: row.geometry.buffer(range) if row['possible_terminals'] == 1 else None, axis=1)
    new_id = node_gdf['id'].max() + 1

    for _, node in node_gdf[node_gdf["possible_terminals"] == 1].iterrows():
        # Get possible_terminal nodes fall within the range of each possible_terminal
        if node['id'] in considered_node_ids:
            continue
        terminal_collection = node_gdf[node_gdf["possible_terminals"] == 1][
            node_gdf[node_gdf["possible_terminals"] == 1].geometry.within(node['buffer'])]
        # create the aggregated terminal based on the possible_terminals that it contains
        node_gdf = update_node_gdf(node_gdf, terminal_collection, considered_node_ids)
        new_id += 1

    node_gdf.drop(columns='buffer', inplace=True)

    return node_gdf


def make_network_from_gdf(network_gdf) -> snkit.network.Network:
    net = Network(edges=network_gdf)
    net = add_endpoints(network=net)
    # In _split_edges_at_nodes if turn the Index=True then we get attribute names instead of _12, for instance
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
    """ based on trails.simplify
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
