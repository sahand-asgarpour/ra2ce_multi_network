from typing import Union

import geopandas as gpd
import numpy as np
import pandas as pd
import snkit.network

from snkit.network import *

from trails import *


def detect_possible_terminals(network_gdf: gpd.GeoDataFrame, aggregation_range: float) -> snkit.network.Network:
    ## Inspired by drop_hanging_nodes in trails/simplify.py
    network = make_network_from_gdf(network_gdf=network_gdf)
    # hanging_nodes : An array of the indices of nodes with degree 1
    hanging_nodes = find_hanging_nodes(network=network)
    possible_terminal = []
    # Check for terminal criteria
    possible_terminal = network.edges[['from_id', 'to_id', 'service']].apply(
        lambda x: check_terminal_criteria(x['from_id'], x['to_id'], x['service'], hanging_nodes), axis=1
    ).dropna().tolist()
    network.nodes['possible_terminal'] = network.nodes['id'].apply(lambda x: 1 if x in possible_terminal else 0)
    # Merge possible_terminal based on an aggregation_range

    network.nodes = aggregate_terminal_nodes(network, aggregation_range=aggregation_range)
    return network


def aggregate_terminal_nodes(network: snkit.network.Network, aggregation_range: float) -> snkit.network.Network:
    # The following 2 functions are used in the body of merge_terminal_nodes function.
    # merge_terminal_nodes starts after the following 2 functions.
    def update_node_gdf(node_gdf: gpd.GeoDataFrame, terminal_collection: gpd.GeoDataFrame, considered_node_ids: list) \
            -> gpd.GeoDataFrame:
        if len(terminal_collection['id'].to_numpy().tolist()) == 1:
            return node_gdf
        if not any(node_id in considered_node_ids for node_id in terminal_collection['id'].to_numpy().tolist()):
            aggregated_terminal_gdf = get_centroid_terminal_gdf(terminal_collection)
            aggregated_ids = terminal_collection['id'].to_numpy().tolist()
        else:
            # find terminal_to_aggr_again_gdf
            terminal_to_aggr_again_gdf = node_gdf[
                node_gdf['terminal_collection'].apply(
                    lambda x: x is not None and any(
                        node_id in x for node_id in terminal_collection['id'].to_numpy().tolist()) if isinstance(x, set)
                    else False)
            ]
            aggregated_ids = terminal_to_aggr_again_gdf['id'].to_numpy().tolist()

            aggregated_terminal_gdf = get_centroid_terminal_gdf(terminal_to_aggr_again_gdf)
            aggregated_again = terminal_to_aggr_again_gdf[
                terminal_to_aggr_again_gdf['terminal_collection'].apply(lambda x: len(x) > 1)]['id'].tolist()
            node_gdf = node_gdf[~node_gdf['id'].isin(aggregated_again)]

        node_gdf.loc[node_gdf['id'].isin(aggregated_ids), 'aggregated'] = 1
        considered_node_ids += aggregated_ids
        return pd.concat([node_gdf, aggregated_terminal_gdf], ignore_index=True)

    def get_centroid_terminal_gdf(_gdf: gpd.GeoDataFrame) -> Union[gpd.GeoDataFrame, None]:
        if _gdf.empty:
            return None
        _centroid_terminal_collection = _gdf['geometry'].unary_union.centroid
        _terminal_collection_ids = _gdf['id'].tolist()
        _aggregated_terminal = Point(_centroid_terminal_collection.x, _centroid_terminal_collection.y)

        return gpd.GeoDataFrame({'id': new_id, 'geometry': _aggregated_terminal,
                                 'possible_terminal': 1,
                                 'terminal_collection': [
                                     {term_col for _id in _terminal_collection_ids
                                      for term_col in node_gdf.loc[
                                          node_gdf['id'] == _id, 'terminal_collection'
                                      ].values[0]}
                                 ],
                                 'aggregated': 0,
                                 'buffer': _aggregated_terminal.buffer(aggregation_range),
                                 },
                                crs=node_gdf.crs)

    # merge_terminal_nodes function starts here
    considered_node_ids = []
    node_gdf = network.nodes
    node_gdf['terminal_collection'] = node_gdf.apply(
        lambda row: {row['id']} if row['possible_terminal'] == 1 else 'n.a.', axis=1)
    node_gdf['aggregated'] = node_gdf.apply(
        lambda row: 0 if row['possible_terminal'] == 1 else 'n.a.', axis=1)
    if aggregation_range == 0:
        return node_gdf

    node_gdf['buffer'] = node_gdf.apply(
        lambda row: row.geometry.buffer(aggregation_range) if row['possible_terminal'] == 1 else None, axis=1)
    new_id = node_gdf['id'].max() + 1

    for _, node in node_gdf[node_gdf["possible_terminal"] == 1].iterrows():
        # Get possible_terminal nodes fall within the aggregation_range of each possible_terminal
        if node['id'] in considered_node_ids:
            continue
        terminal_collection = node_gdf[node_gdf["possible_terminal"] == 1][
            node_gdf[node_gdf["possible_terminal"] == 1].geometry.within(node['buffer'])]
        # create the aggregated terminal based on the possible_terminal that it contains
        node_gdf = update_node_gdf(node_gdf, terminal_collection, considered_node_ids)
        new_id += 1

    node_gdf.drop(columns='buffer', inplace=True)
    node_gdf['terminal_collection'] = node_gdf['terminal_collection'].apply(
        lambda val: sorted(list(val)) if isinstance(val, set) else val)
    node_gdf['terminal_collection'] = node_gdf.apply(
        lambda row: row['terminal_collection'] if isinstance(row['aggregated'], int) and row['aggregated'] == 0
        else 'n.a.', axis=1)
    return node_gdf


def make_network_from_gdf(network_gdf: gpd.GeoDataFrame) -> snkit.network.Network:
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


def find_hanging_nodes(network: snkit.network.Network) -> np.ndarray:
    if 'degree' not in network.nodes.columns:
        deg = calculate_degree(network)
    else:
        deg = network.nodes['degree'].to_numpy()
    return np.where(deg == 1)[0]


def calculate_degree(network: snkit.network.Network) -> np.ndarray:
    """ based on trails.simplify
    """
    # the number of nodes(from index) to use as the number of bins
    ndC = len(network.nodes.index)
    if ndC - 1 > max(network.edges.from_id) and ndC - 1 > max(network.edges.to_id): print(
        "Calculate_degree possibly unhappy")
    return np.bincount(network.edges['from_id'], None, ndC) + np.bincount(network.edges['to_id'], None, ndC)


def check_terminal_criteria(from_node_id: int, to_node_id: int, edge_property: str, hanging_nodes: np.ndarray) -> int:
    if from_node_id in hanging_nodes and edge_property == 'spur':
        return from_node_id
    elif to_node_id in hanging_nodes and edge_property == 'spur':
        return to_node_id
