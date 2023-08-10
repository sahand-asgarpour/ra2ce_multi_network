from networkx import MultiDiGraph, Graph
from typing import Union, List

import geopandas as gpd
import numpy as np
import pandas as pd
import snkit.network

from snkit.network import *

from trails import *


def get_rail_network_with_terminals(network_gdf: gpd.GeoDataFrame, aggregation_range: float) -> snkit.network.Network:
    network = _make_network_from_gdf(network_gdf=network_gdf)
    # detect possible_terminal
    network = _detect_possible_terminals(network)
    # Merge possible_terminal based on an aggregation_range
    network.nodes = _aggregate_terminal_nodes(network, aggregation_range=aggregation_range)
    # Add demand links between aggregate_terminal nodes and network nodes
    network = _add_demand_link(network)
    network = _reset_indices(network)
    return network


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


def _detect_possible_terminals(network: snkit.network.Network) -> snkit.network.Network:
    # hanging_nodes : An array of the indices of nodes with degree 1
    hanging_nodes = _find_hanging_nodes(network=network)
    possible_terminal = []
    # Check for terminal criteria
    possible_terminal = network.edges[['from_id', 'to_id', 'service']].apply(
        lambda x: _check_terminal_criteria(x['from_id'], x['to_id'], x['service'], hanging_nodes), axis=1
    ).dropna().tolist()
    network.nodes['possible_terminal'] = network.nodes['id'].apply(lambda x: 1 if x in possible_terminal else 0)
    return network


def _aggregate_terminal_nodes(network: snkit.network.Network, aggregation_range: float) -> snkit.network.Network:
    # The following 2 functions are used in the body of _aggregate_terminal_nodes function.
    # _aggregate_terminal_nodes starts after the following 2 functions.
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

    # _aggregate_terminal_nodes function starts here
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
            'demand_link': int(1),
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
        network.edges['demand_link'].fillna(0, inplace=True)
        network.edges['id'] = range(len(network.edges))

    network = _get_demand_link_attributes(network)

    return network


def _reset_indices(network: snkit.network.Network) -> snkit.network.Network:
    updated_nodes = network.nodes.reset_index(drop=True)
    updated_edges = network.edges.reset_index(drop=True)
    return Network(nodes=updated_nodes, edges=updated_edges)


def _get_demand_link_attributes(network: snkit.network.Network) -> snkit.network.Network:
    edge_cols = network.edges.columns.tolist()
    node_cols = network.nodes.columns.tolist()
    network_x = _to_networkx(network=network, directed=False, node_attributes=node_cols, edge_attributes=edge_cols)
    aggregate_demand_nodes = network.nodes[(network.nodes['aggregate'] == 1) &
                                           (network.nodes['terminal_collection'].str.len() > 1)]
    for _, row in aggregate_demand_nodes.iterrows():
        for child_ter_id in row['terminal_collection']:
            neighbor_edges = list(network_x.edges(child_ter_id, data=True))
            edge_to_update_info = [edge for edge in neighbor_edges if edge[2]['demand_link'] == 1]

            if len(edge_to_update_info) == 1 and len(neighbor_edges) > 1:
                edge_to_update = edge_to_update_info[0]
                reference_edge = [edge for edge in neighbor_edges if edge != edge_to_update][0]

                attributes_to_update = [attr for attr, value in edge_to_update[2].items() if value is None]
                edge_to_update[2].update(
                    {attribute: reference_edge[2][attribute] for attribute in attributes_to_update})

                network_x[edge_to_update[0]][edge_to_update[1]].update(edge_to_update[2])
            else:
                raise ValueError("""Invalid edge graph. There is more or less than one demand node connected or 
                                there are no neighbouring edges except the demand link""")

    network = _to_snkit_network(network_x)

    return network


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


def _to_snkit_network(netx: Union[Graph, MultiDiGraph]) -> snkit.network.Network:
    # Convert nodes to GeoDataFrame
    node_dict = {node: {**data, 'id': node} for node, data in netx.nodes(data=True)}
    node_gdf = gpd.GeoDataFrame.from_dict(node_dict, orient='index')
    node_gdf.set_geometry('geometry', inplace=True)

    # Convert edges to GeoDataFrame
    edge_dict = {ind: {**edges[2], 'from_id': edges[0], 'to_id': edges[1]}
                 for ind, edges in enumerate(netx.edges(data=True))}
    edge_gdf = gpd.GeoDataFrame.from_dict(edge_dict, orient='index')
    edge_gdf.set_geometry('geometry', inplace=True)

    net = Network(nodes=node_gdf, edges=edge_gdf)

    return net


def _find_hanging_nodes(network: snkit.network.Network) -> np.ndarray:
    if 'degree' not in network.nodes.columns:
        deg = _calculate_degree(network)
    else:
        deg = network.nodes['degree'].to_numpy()
    return np.where(deg == 1)[0]


def _calculate_degree(network: snkit.network.Network) -> np.ndarray:
    # Get the maximum node ID from both 'from_id' and 'to_id' arrays
    max_node_id = int(max(max(network.edges['from_id']), max(network.edges['to_id'])))
    # Initialize a weights array to count the degrees for each node
    degrees = np.zeros(max_node_id + 1)
    # Calculate the degree for the 'from_id' array and add it to degrees
    np.add.at(degrees, network.edges['from_id'], 1)
    # Calculate the degree for the 'to_id' array and add it to degrees
    np.add.at(degrees, network.edges['to_id'], 1)

    return degrees


def _check_terminal_criteria(from_node_id: int, to_node_id: int, edge_property: str, hanging_nodes: np.ndarray) -> int:
    if from_node_id in hanging_nodes and edge_property == 'spur':
        return from_node_id
    elif to_node_id in hanging_nodes and edge_property == 'spur':
        return to_node_id


def simplify_rail(network: snkit.network.Network) -> snkit.network.Network:
    network = _merge_edges(network, excluded_edge_types=['bridge', 'tunnel'])
    return network


def _merge_edges(network: snkit.network.Network, excluded_edge_types: List[str]) -> snkit.network.Network:
    # _merge_edges starts here. add the degree column to nodes and put it high for the excluded_edge_types objects
    network = _get_nodes_degree(network)
    # _exclude_edge_types()
    # merge_edges
    cols = [col for col in network.edges.columns if col != 'geometry']
    network = merge_edges_modified(network, by=excluded_edge_types, aggfunc={
        col: (
            lambda col_data: '; '.join(filter(None, col_data.unique())) if col_data.dtype == 'O' else col_data.iloc[0])
        for col in cols
    })
    # update teh degree column with normal values
    network = _get_nodes_degree(network)
    return network


def _get_nodes_degree(network: snkit.network.Network) -> snkit.network.Network:
    degrees = _calculate_degree(network)
    node_degrees_dict = {node_id: degree for node_id, degree in enumerate(degrees) if degree > 0}
    network.nodes['degree'] = network.nodes['id'].map(node_degrees_dict)
    return network


def merge_edges_modified(network: snkit.network.Network, aggfunc: Union[str, dict], by: Union[str, list],
                         id_col="id") -> snkit.network.Network:
    if "degree" not in network.nodes.columns:
        network.nodes["degree"] = network.nodes[id_col].apply(
            lambda x: node_connectivity_degree(x, network)
        )

    degree2 = list(network.nodes[id_col].loc[network.nodes.degree == 2])
    d2_set = set(degree2)
    edge_paths = []

    while d2_set:
        if len(d2_set) % 1000 == 0:
            print(len(d2_set))
        popped_node = d2_set.pop()
        node_path = set([popped_node])
        candidates = set([popped_node])
        while candidates:
            popped_cand = candidates.pop()
            matches = set(
                np.unique(
                    network.edges[["from_id", "to_id"]]
                    .loc[
                        (network.edges.from_id == popped_cand)
                        | (network.edges.to_id == popped_cand)
                        ]
                    .values
                )
            )
            matches.remove(popped_cand)
            matches = matches - node_path
            for match in matches:
                if match in degree2:
                    candidates.add(match)
                    node_path.add(match)
                    d2_set.remove(match)
                else:
                    node_path.add(match)
        if len(node_path) > 2:
            edge_paths.append(
                network.edges.loc[
                    (network.edges.from_id.isin(node_path))
                    & (network.edges.to_id.isin(node_path))
                    ]
            )

    concat_edge_paths = []
    unique_edge_ids = set()
    new_node_ids = set(network.nodes[id_col]) - set(degree2)

    for edge_path in tqdm(edge_paths, desc="merge_edge_paths"):
        unique_edge_ids.update(list(edge_path[id_col]))
        unique_values_dict = {}

        for col in by:
            unique_values_dict[col] = edge_path[col].unique()

        # Convert None values to a placeholder value
        placeholder = "None"
        for col in by:
            edge_path[col] = edge_path[col].fillna(placeholder)

        # Perform dissolve operation
        edge_path = edge_path.dissolve(by=by, aggfunc=aggfunc)

        # Replace the placeholder value back to None
        edge_path.replace(placeholder, None, inplace=True)

        # edge_path = edge_path.dissolve(by=by, aggfunc=aggfunc)
        edge_path_dicts = []

        for edge in edge_path.itertuples(index=False):
            if edge.geometry.geom_type == "MultiLineString":
                edge_geom = linemerge(edge.geometry)
                if edge_geom.geom_type == "MultiLineString":
                    edge_geoms = list(edge_geom.geoms)
                else:
                    edge_geoms = [edge_geom]
            else:
                edge_geoms = [edge.geometry]

            for geom in edge_geoms:
                start, end = line_endpoints(geom)
                start = nearest_node(start, network.nodes)
                end = nearest_node(end, network.nodes)
                edge_path_dict = {
                    "from_id": start[id_col],
                    "to_id": end[id_col],
                    "geometry": geom,
                }
                for col in by:
                    edge_path_dict[col] = '; '.join(str(item) if item is not None else placeholder
                                                    for item in unique_values_dict[col])

                for i, col in enumerate(edge_path.columns):
                    if col not in ("from_id", "to_id", "geometry") and col not in by:
                        edge_path_dict[col] = edge[i]

                edge_path_dicts.append(edge_path_dict)

        concat_edge_paths.append(geopandas.GeoDataFrame(edge_path_dicts))
        new_node_ids.update(list(edge_path.from_id) + list(edge_path.to_id))

    edges_new = network.edges.copy()
    edges_new = edges_new.loc[~(edges_new.id.isin(list(unique_edge_ids)))]
    edges_new.geometry = edges_new.geometry.apply(merge_multilinestring)
    edges = pd.concat(
        [edges_new, pd.concat(concat_edge_paths).reset_index(drop=True)], sort=False
    ).applymap(lambda x: None if pd.isna(x) or x == '' else x)

    nodes = network.nodes.set_index(id_col).loc[list(new_node_ids)].copy().reset_index()

    return Network(nodes=nodes, edges=edges)


# def merge_edges_modified(network: snkit.network.Network, aggfunc: Union[str, dict], by: Union[str, list], id_col="id")\
#         -> snkit.network.Network:
#     """
#     Based on snkit.network.
#     aggfunc is added. Modified to resolve error for edge_geoms = list(edge_geom)
#
#     Merge edges that share a node with a connectivity degree of 2
#
#     Parameters
#     ----------
#     network : snkit.network.Network
#     id_col : string
#     by : List[string], optional
#       list of columns to use when merging an edge path - will not merge if
#       edges have different values.
#     aggfunc : Aggregation function for manipulation of data associated with each group
#     """
#     if "degree" not in network.nodes.columns:
#         network.nodes["degree"] = network.nodes[id_col].apply(
#             lambda x: node_connectivity_degree(x, network)
#         )
#
#     degree2 = list(network.nodes[id_col].loc[network.nodes.degree == 2])
#     d2_set = set(degree2)
#     edge_paths = []
#
#     while d2_set:
#         if len(d2_set) % 1000 == 0:
#             print(len(d2_set))
#         popped_node = d2_set.pop()
#         node_path = set([popped_node])
#         candidates = set([popped_node])
#         while candidates:
#             popped_cand = candidates.pop()
#             matches = set(
#                 np.unique(
#                     network.edges[["from_id", "to_id"]]
#                     .loc[
#                         (network.edges.from_id == popped_cand)
#                         | (network.edges.to_id == popped_cand)
#                         ]
#                     .values
#                 )
#             )
#             matches.remove(popped_cand)
#             matches = matches - node_path
#             for match in matches:
#                 if match in degree2:
#                     candidates.add(match)
#                     node_path.add(match)
#                     d2_set.remove(match)
#                 else:
#                     node_path.add(match)
#         if len(node_path) > 2:
#             edge_paths.append(
#                 network.edges.loc[
#                     (network.edges.from_id.isin(node_path))
#                     & (network.edges.to_id.isin(node_path))
#                     ]
#             )
#
#     concat_edge_paths = []
#     unique_edge_ids = set()
#     new_node_ids = set(network.nodes[id_col]) - set(degree2)
#
#     for edge_path in tqdm(edge_paths, desc="merge_edge_paths"):
#         unique_edge_ids.update(list(edge_path[id_col]))
#         edge_path = edge_path.dissolve(by=by, aggfunc=aggfunc)
#         edge_path_dicts = []
#         for edge in edge_path.itertuples(index=False):
#             if edge.geometry.geom_type == "MultiLineString":
#                 edge_geom = linemerge(edge.geometry)
#                 if edge_geom.geom_type == "MultiLineString":
#                     edge_geoms = list(edge_geom.geoms)
#                 else:
#                     edge_geoms = [edge_geom]
#             else:
#                 edge_geoms = [edge.geometry]
#             for geom in edge_geoms:
#                 start, end = line_endpoints(geom)
#                 start = nearest_node(start, network.nodes)
#                 end = nearest_node(end, network.nodes)
#                 edge_path_dict = {
#                     "from_id": start[id_col],
#                     "to_id": end[id_col],
#                     "geometry": geom,
#                 }
#                 for i, col in enumerate(edge_path.columns):
#                     if col not in ("from_id", "to_id", "geometry"):
#                         edge_path_dict[col] = edge[i]
#                 edge_path_dicts.append(edge_path_dict)
#
#         concat_edge_paths.append(geopandas.GeoDataFrame(edge_path_dicts))
#         new_node_ids.update(list(edge_path.from_id) + list(edge_path.to_id))
#
#     edges_new = network.edges.copy()
#     edges_new = edges_new.loc[~(edges_new.id.isin(list(unique_edge_ids)))]
#     edges_new.geometry = edges_new.geometry.apply(merge_multilinestring)
#     edges = pd.concat(
#         [edges_new, pd.concat(concat_edge_paths).reset_index()], sort=False
#     ).applymap(lambda x: None if pd.isna(x) else x)
#
#     nodes = network.nodes.set_index(id_col).loc[list(new_node_ids)].copy().reset_index()
#
#     return Network(nodes=nodes, edges=edges)

def _exclude_edge_types(net: snkit.network.Network, ex_edge_types: list):
    degree_2_nodes = net.nodes[net.nodes['degree'] == 2]
    for ex_edge_type in ex_edge_types:
        if ex_edge_type not in net.edges.columns.tolist():
            raise ValueError(f"{ex_edge_type} does not exist.")
        # Check if the nodes with degree 2 are connected to excluded_edge
        connected_to_ex_ed = degree_2_nodes.apply(lambda row:
                                                  any(net.edges[(net.edges['from_id'] == row['id']) |
                                                                (net.edges['to_id'] == row['id'])]
                                                      [ex_edge_type].notna()), axis=1)

        net.nodes.loc[net.nodes['id'].isin(degree_2_nodes.loc[connected_to_ex_ed, 'id']), 'degree'] = 1e10
