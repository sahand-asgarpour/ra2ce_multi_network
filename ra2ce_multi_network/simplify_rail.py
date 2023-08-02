from networkx import MultiDiGraph, Graph
from typing import Union

import geopandas as gpd
import numpy as np
import pandas as pd
import snkit.network

from snkit.network import *

from trails import *


def detect_possible_terminals(network_gdf: gpd.GeoDataFrame, aggregation_range: float) -> snkit.network.Network:
    ## Inspired by drop_hanging_nodes in trails/simplify.py
    network = _make_network_from_gdf(network_gdf=network_gdf)
    # hanging_nodes : An array of the indices of nodes with degree 1
    hanging_nodes = _find_hanging_nodes(network=network)
    possible_terminal = []
    # Check for terminal criteria
    possible_terminal = network.edges[['from_id', 'to_id', 'service']].apply(
        lambda x: _check_terminal_criteria(x['from_id'], x['to_id'], x['service'], hanging_nodes), axis=1
    ).dropna().tolist()
    network.nodes['possible_terminal'] = network.nodes['id'].apply(lambda x: 1 if x in possible_terminal else 0)

    # Merge possible_terminal based on an aggregation_range
    network.nodes = _aggregate_terminal_nodes(network, aggregation_range=aggregation_range)
    # Add demand links between aggregate_terminal nodes and network nodes
    network = _add_demand_link(network)
    return network


def _aggregate_terminal_nodes(network: snkit.network.Network, aggregation_range: float) -> snkit.network.Network:
    # The following 2 functions are used in the body of merge_terminal_nodes function.
    # merge_terminal_nodes starts after the following 2 functions.
    def update_node_gdf(node_gdf: gpd.GeoDataFrame, terminal_collection: gpd.GeoDataFrame, considered_node_ids: list) \
            -> gpd.GeoDataFrame:
        if len(terminal_collection['id'].to_numpy().tolist()) == 1:
            return node_gdf
        if not any(node_id in considered_node_ids for node_id in terminal_collection['id'].to_numpy().tolist()):
            aggregate_terminal_gdf = get_centroid_terminal_gdf(terminal_collection)
            to_aggregate_ids = terminal_collection['id'].to_numpy().tolist()
        else:
            # find terminal_to_aggr_again_gdf
            terminal_to_aggr_again_gdf = node_gdf[
                node_gdf['terminal_collection'].apply(
                    lambda x: x is not None and any(
                        node_id in x for node_id in terminal_collection['id'].to_numpy().tolist()) if isinstance(x, set)
                    else False)
            ]
            to_aggregate_ids = terminal_to_aggr_again_gdf['id'].to_numpy().tolist()

            aggregate_terminal_gdf = get_centroid_terminal_gdf(terminal_to_aggr_again_gdf)
            to_aggregate_again = terminal_to_aggr_again_gdf[
                terminal_to_aggr_again_gdf['terminal_collection'].apply(lambda x: len(x) > 1)]['id'].tolist()
            node_gdf = node_gdf[~node_gdf['id'].isin(to_aggregate_again)]

        node_gdf.loc[node_gdf['id'].isin(to_aggregate_ids), 'aggregate'] = 0
        considered_node_ids += to_aggregate_ids
        return pd.concat([node_gdf, aggregate_terminal_gdf], ignore_index=True)

    def get_centroid_terminal_gdf(_gdf: gpd.GeoDataFrame) -> Union[gpd.GeoDataFrame, None]:
        if _gdf.empty:
            return None
        _centroid_terminal_collection = _gdf['geometry'].unary_union.centroid
        _terminal_collection_ids = _gdf['id'].tolist()

        return gpd.GeoDataFrame({'id': new_id, 'geometry': Point(_centroid_terminal_collection.x,
                                                                 _centroid_terminal_collection.y),
                                 'possible_terminal': 1,
                                 'terminal_collection': [
                                     {term_col for _id in _terminal_collection_ids
                                      for term_col in node_gdf.loc[
                                          node_gdf['id'] == _id, 'terminal_collection'
                                      ].values[0]}
                                 ],
                                 'aggregate': 1,
                                 'buffer': Point(
                                     _centroid_terminal_collection.x, _centroid_terminal_collection.y
                                 ).buffer(aggregation_range),
                                 },
                                crs=node_gdf.crs)

    # merge_terminal_nodes function starts here
    considered_node_ids = []
    node_gdf = network.nodes
    node_gdf['terminal_collection'] = node_gdf.apply(
        lambda row: {row['id']} if row['possible_terminal'] == 1 else 'n.a.', axis=1)
    node_gdf['aggregate'] = node_gdf.apply(
        lambda row: 1 if row['possible_terminal'] == 1 else 'n.a.', axis=1)
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
        # create the aggregate terminal based on the possible_terminal that it contains
        node_gdf = update_node_gdf(node_gdf, terminal_collection, considered_node_ids)
        new_id += 1

    node_gdf.drop(columns='buffer', inplace=True)
    node_gdf['terminal_collection'] = node_gdf['terminal_collection'].apply(
        lambda val: sorted(list(val)) if isinstance(val, set) else val)
    node_gdf['terminal_collection'] = node_gdf.apply(
        lambda row: row['terminal_collection'] if isinstance(row['aggregate'], int) and row['aggregate'] == 1
        else 'n.a.', axis=1)
    return node_gdf


def _add_demand_link(network: snkit.network.Network) -> snkit.network.Network:
    node_gdf = network.nodes
    edge_columns = network.edges.columns
    aggregate_demand_nodes = node_gdf[(node_gdf['aggregate'] == 1) & (node_gdf['terminal_collection'].str.len() > 1)]

    new_edges_data = [
        {
            'geometry': LineString([
                (row['geometry'].x, row['geometry'].y),
                (node_gdf.loc[child_ter_id, 'geometry'].x, node_gdf.loc[child_ter_id, 'geometry'].y)
            ]),
            'from_id': row['id'],
            'to_id': child_ter_id,
            **{column: None for column in edge_columns if column not in ['geometry', 'id', 'from_id', 'to_id']}
        }
        for _, row in aggregate_demand_nodes.iterrows()
        for child_ter_id in row['terminal_collection']
        if len(row['terminal_collection']) > 1
    ]

    if new_edges_data:
        new_edges_df = gpd.GeoDataFrame(new_edges_data, crs=node_gdf.crs)
        network.edges = pd.concat([network.edges, new_edges_df], ignore_index=True)
        network.edges['id'] = range(len(network.edges))

    network_nx = _get_demand_link_attributes(network)

    return network


def _get_demand_link_attributes(network: snkit.network.Network) -> Union[Graph, MultiDiGraph]:
    edge_cols = network.edges.columns.tolist()
    node_cols = network.nodes.columns.tolist()
    network_nx = _to_networkx(network=network, directed=False, node_attributes=node_cols, edge_attributes=edge_cols)
    aggregate_demand_nodes = network.nodes[(network.nodes['aggregate'] == 1) &
                                           (network.nodes['terminal_collection'].str.len() > 1)]
    for _, row in aggregate_demand_nodes.iterrows():
        for child_ter_id in row['terminal_collection']:
            neighbor_edges = list(network_nx.edges(child_ter_id, data=True))
            edges_with_none_osm_id = [edge for edge in neighbor_edges if edge[2]['osm_id'] is None]

            if len(edges_with_none_osm_id) == 1 and len(neighbor_edges) > 1:
                edge_to_update = edges_with_none_osm_id[0]
                reference_edge = [edge for edge in neighbor_edges if edge != edge_to_update][0]

                attributes_to_update = [attr for attr, value in edge_to_update[2].items() if value is None]
                edge_to_update[2].update(
                    {attribute: reference_edge[2][attribute] for attribute in attributes_to_update})

                network_nx[edge_to_update[0]][edge_to_update[1]].update(edge_to_update[2])

    return network_nx


def _to_networkx(network: snkit.network.Network, directed: bool, node_attributes: list, edge_attributes: list) \
        -> Union[Graph, MultiDiGraph]:
    if not directed:
        g = nx.Graph()
    else:
        g = nx.MultiDiGraph()
    _add_nodes_to_nx(g, network, node_attributes)
    _add_links_to_nx(g, network, edge_attributes)
    return g


def _add_nodes_to_nx(g: Union[Graph, MultiDiGraph], net: snkit.network.Network, node_atts: list):
    if 'id' not in net.nodes.columns.tolist():
        raise ValueError("'id' must exist in the GeoDataFrame.")

    for _, row in net.nodes.iterrows():
        node_id = row['id']
        attributes = row[node_atts].to_dict() if node_atts else {'geometry': row['geometry']}
        g.add_node(node_id, **attributes)


def _add_links_to_nx(g: Union[Graph, MultiDiGraph], net: snkit.network.Network, edge_atts: list):
    if 'from_id' not in net.edges.columns.tolist() or 'to_id' not in net.edges.columns.tolist():
        raise ValueError("'from_id' and 'to_id' must exist in the GeoDataFrame.")

    for _, row in net.edges.iterrows():
        u, v = row['from_id'], row['to_id']
        attributes = row[edge_atts].to_dict() if edge_atts else {'geometry': row['geometry']}
        g.add_edge(u, v, **attributes)


def _make_network_from_gdf(network_gdf: gpd.GeoDataFrame) -> snkit.network.Network:
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
    net.edges.id = net.edges.id.str.extract(r"(\d+)", expand=False).astype(int)
    return net


def _find_hanging_nodes(network: snkit.network.Network) -> np.ndarray:
    if 'degree' not in network.nodes.columns:
        deg = _calculate_degree(network)
    else:
        deg = network.nodes['degree'].to_numpy()
    return np.where(deg == 1)[0]


def _calculate_degree(network: snkit.network.Network) -> np.ndarray:
    """ based on trails.simplify
    """
    # the number of nodes(from index) to use as the number of bins
    ndC = len(network.nodes.index)
    if ndC - 1 > max(network.edges.from_id) and ndC - 1 > max(network.edges.to_id): print(
        "Calculate_degree possibly unhappy")
    return np.bincount(network.edges['from_id'], None, ndC) + np.bincount(network.edges['to_id'], None, ndC)


def _check_terminal_criteria(from_node_id: int, to_node_id: int, edge_property: str, hanging_nodes: np.ndarray) -> int:
    if from_node_id in hanging_nodes and edge_property == 'spur':
        return from_node_id
    elif to_node_id in hanging_nodes and edge_property == 'spur':
        return to_node_id
