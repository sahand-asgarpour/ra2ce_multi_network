from pathlib import Path

import pyproj
import triangle as tr
from geopandas import GeoSeries
from networkx import MultiDiGraph, Graph, MultiGraph
from pandas import Index
from shapely import Polygon, MultiPolygon, MultiLineString
from typing import Union, List, Optional
from pyproj import CRS
import geopandas as gpd
import pandas as pd
import snkit.network

from snkit.network import *

from ra2ce.graph.origins_destinations import add_od_nodes
from ra2ce_multi_network.trails import *


def get_rail_network_with_terminals(network_gdf: gpd.GeoDataFrame, aggregation_range: float) -> snkit.network.Network:
    network = _make_network_from_gdf(network_gdf=network_gdf)
    # detect possible_terminal
    network = _detect_possible_terminals(network)
    # Merge possible_terminal based on an aggregation_range
    network.nodes = _aggregate_terminal_nodes(network, aggregation_range=aggregation_range)
    # Add demand links between aggregate_terminal nodes and network nodes
    network = _add_demand_edge(network)
    network = _reset_indices(network)
    network.set_crs(crs="EPSG:4326")
    return network


def get_rail_network_with_given_terminals(network_gdf: gpd.GeoDataFrame, od_file: Path, network=None) -> (
        snkit.network.Network):
    if network is None:
        network = _make_network_from_gdf(network_gdf=network_gdf)
    network.set_crs(crs="EPSG:4326")
    network = _drop_hanging_nodes(network)
    graph = _network_to_nx(network)
    od_gdf, graph = add_od_nodes(od=gpd.read_file(od_file), graph=graph, crs=pyproj.CRS("EPSG:4326"))
    network = _nx_to_network(graph)
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


def _network_to_nx(net: snkit.network.Network, node_id_column_name='id',
                   edge_from_id_column='from_id', edge_to_id_column='to_id',
                   default_crs: Optional[CRS] = CRS.from_epsg(4326)) -> MultiGraph:
    g = nx.MultiGraph()

    # Add nodes to the graph
    for index, row in net.nodes.iterrows():
        node_id = row[node_id_column_name]
        attributes = {k: v for k, v in row.items()}
        g.add_node(node_id, **attributes)

    # Add edges to the graph
    for index, row in net.edges.iterrows():
        u = row[edge_from_id_column]
        v = row[edge_to_id_column]
        attributes = {k: v for k, v in row.items()}
        g.add_edge(u, v, **attributes)

    # Add CRS information to the graph
    if 'crs' not in g.graph:
        g.graph['crs'] = default_crs

    return g


def _nx_to_network(g: Union[Graph, MultiGraph, MultiDiGraph], node_id_column_name='id',
                   edge_from_id_column='from_id', edge_to_id_column='to_id',
                   default_crs: CRS = CRS.from_epsg(4326)) -> snkit.network.Network:
    network = snkit.network.Network()

    node_attributes = [{node_id_column_name: node, **data} for node, data in g.nodes(data=True)]
    network.nodes = gpd.GeoDataFrame(node_attributes)
    network.nodes.set_geometry('geometry', inplace=True)

    edge_attributes = [{edge_from_id_column: u, edge_to_id_column: v, **data} for u, v, data in g.edges(data=True)]
    network.edges = gpd.GeoDataFrame(edge_attributes)
    network.edges.set_geometry('geometry', inplace=True)

    # Set network CRS to default_crs
    network.set_crs(default_crs)

    return network


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
        aggregation_range_degree = _km_distance_to_degrees(aggregation_range)
        if _gdf.empty:
            return None
        _centroid_terminal_collection = _gdf['geometry'].unary_union.centroid  # the point(its gdf) does not have crs
        transformer = pyproj.Transformer.from_crs("epsg:4326", pyproj.CRS("epsg:4326"), always_xy=True)
        projected_centroid_terminal_collection = Point(transformer.transform(_centroid_terminal_collection.x,
                                                                             _centroid_terminal_collection.y))
        _terminal_collection_ids = _gdf['id'].tolist()

        return gpd.GeoDataFrame({'id': new_id, 'geometry': Point(projected_centroid_terminal_collection.x,
                                                                 projected_centroid_terminal_collection.y),
                                 'possible_terminal': 1,
                                 'terminal_collection': [
                                     {term_col for _id in _terminal_collection_ids
                                      for term_col in node_gdf.loc[
                                          node_gdf['id'] == _id, 'terminal_collection'
                                      ].values[0]}
                                 ],
                                 'aggregate': 1,
                                 'buffer': Point(projected_centroid_terminal_collection.x,
                                                 projected_centroid_terminal_collection.y
                                                 ).buffer(aggregation_range_degree),
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
        node_gdf['terminal_collection'] = node_gdf.apply(
            lambda row: list(row['terminal_collection']) if isinstance(row['terminal_collection'], set) else
            row['possible_terminal'], axis=1)
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


def _km_distance_to_degrees(distance_km: float):
    return distance_km / 111.32


def _add_demand_edge(network: snkit.network.Network) -> snkit.network.Network:
    node_gdf = network.nodes
    edge_columns = network.edges.columns
    aggregate_demand_nodes = node_gdf[(node_gdf['aggregate'] == 1) & (node_gdf['terminal_collection'].str.len() > 1)]

    new_edges_data = [
        {
            'geometry': LineString([
                (row['geometry'].x, row['geometry'].y),
                (node_gdf.loc[child_ter_id, 'geometry'].x, node_gdf.loc[child_ter_id, 'geometry'].y)
            ]),
            'demand_edge': int(1),
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
        network.edges['demand_edge'].fillna(0, inplace=True)
        network.edges['id'] = range(len(network.edges))

    network = _get_demand_edge_attributes(network)

    return network


def _reset_indices(network: snkit.network.Network) -> snkit.network.Network:
    updated_nodes = network.nodes.reset_index(drop=True)
    updated_edges = network.edges.reset_index(drop=True)
    return Network(nodes=updated_nodes, edges=updated_edges)


def _get_demand_edge_attributes(network: snkit.network.Network) -> snkit.network.Network:
    edge_cols = network.edges.columns.tolist()
    node_cols = network.nodes.columns.tolist()
    network_x = _to_networkx(network=network, directed=False, node_attributes=node_cols, edge_attributes=edge_cols)
    aggregate_demand_nodes = network.nodes[(network.nodes['aggregate'] == 1) &
                                           (network.nodes['terminal_collection'].str.len() > 1)]
    for _, row in aggregate_demand_nodes.iterrows():
        for child_ter_id in row['terminal_collection']:
            neighbor_edges = list(network_x.edges(child_ter_id, data=True))
            edge_to_update_info = [edge for edge in neighbor_edges if edge[2]['demand_edge'] == 1]

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


def _calculate_degree(net: snkit.network.Network) -> np.ndarray:
    # Get the maximum node ID from both 'from_id' and 'to_id' arrays
    max_node_id = int(max(max(net.edges['from_id']), max(net.edges['to_id'])))
    # Initialize a weights array to count the degrees for each node
    degrees = np.zeros(max_node_id + 1)
    # Calculate the degree for the 'from_id' array and add it to degrees
    from_ids = net.edges['from_id'].to_numpy(dtype=np.int64)
    np.add.at(degrees, from_ids, 1)
    # Calculate the degree for the 'to_id' array and add it to degrees
    to_id = net.edges['to_id'].to_numpy(dtype=np.int64)
    np.add.at(degrees, to_id, 1)
    return degrees


def _check_terminal_criteria(from_node_id: int, to_node_id: int, edge_property: str, hanging_nodes: np.ndarray) -> int:
    if from_node_id in hanging_nodes and edge_property == 'spur':
        return from_node_id
    elif to_node_id in hanging_nodes and edge_property == 'spur':
        return to_node_id


def simplify_rail(network: snkit.network.Network) -> snkit.network.Network:
    network = _merge_edges(excluded_edge_types=['bridge', 'tunnel'], network=network)
    network = _simplify_tracks(network, 0.012, 0.01)
    return network


def _merge_edges(excluded_edge_types: List[str], network: snkit.network.Network = None, graph: Graph = None) \
        -> snkit.network.Network:
    # _merge_edges starts here. add the degree column to nodes and put it high for the excluded_edge_types objects
    if (not network) and (not graph):
        raise ValueError("a network or graph should be introduced")
    if graph and (not network):
        network = _nx_to_network(graph)

    network = _get_nodes_degree(network)
    # merge_edges
    cols = [col for col in network.edges.columns if col != 'geometry']

    if 'demand_edge' not in excluded_edge_types:
        aggfunc = {
            col: (
                lambda col_data: '; '.join(str(item) for item in col_data if isinstance(item, str))
                if col_data.dtype == 'O'
                else col_data.iloc[0]
                if col != "demand_edge"
                else max(col_data)
            )
            for col in cols
        }
    else:
        aggfunc = {
            col: (
                lambda col_data: '; '.join(str(item) for item in col_data if isinstance(item, str))
                if col_data.dtype == 'O'
                else col_data.iloc[0]
            )
            for col in cols
        }

    network = merge_edges(network, aggfunc=aggfunc, by=excluded_edge_types)
    network.edges['length'] = network.edges['geometry'].length * 111.32  # length in km
    network.edges = network.edges[network.edges['length'] != 0]  # sometimes such links emerge during merging.

    convert_to_line_string = lambda geom: linemerge(
        [line for line in geom.geoms]) if isinstance(geom, MultiLineString) else geom
    network.edges['geometry'] = network.edges['geometry'].apply(convert_to_line_string)
    return network


def _get_nodes_degree(network: snkit.network.Network) -> snkit.network.Network:
    degrees = _calculate_degree(network)
    node_degrees_dict = {node_id: degree for node_id, degree in enumerate(degrees) if degree > 0}
    network.nodes['degree'] = network.nodes.apply(lambda node: node_degrees_dict[node.id], axis=1)
    return network


def _get_merged_edges(paths_to_group: list, by: list,
                      aggfunc: Union[str, dict], net: snkit.network.Network) -> GeoDataFrame:
    updated_edges = gpd.GeoDataFrame(columns=net.edges.columns, crs=net.edges.crs)  # merged edges

    for edge_path in tqdm(paths_to_group, desc="merge_edge_paths"):
        # Convert None values to a placeholder value
        placeholder = "None"
        for col in by:
            edge_path[col] = edge_path[col].fillna(placeholder)
        merged_edges = _get_merge_edge_paths(edge_path, by, aggfunc, net)
        updated_edges = pd.concat([updated_edges, merged_edges], ignore_index=True)

    return updated_edges


def _get_merge_edge_paths(edges: GeoDataFrame, excluded_edge_types: list, aggfunc: Union[str, dict],
                          net: snkit.network.Network) -> GeoDataFrame:
    def _get_sub_path_parts(ids: Index) -> list:
        sub_path_parts = []
        edge_group = edges.loc[ids.tolist()]  # loc finds the elements based on the index numbers
        edge_group['intersections'] = edge_group.apply(lambda x: _get_intersections(x, edges), axis=1)
        for edge in edge_group.itertuples(index=False):
            sub_path_part = [edge.id]  # list of edge.id  #
            for other_edge in edge_group.itertuples(index=False):
                if edge.id != other_edge.id:
                    if len(set(edge.intersections) & set(other_edge.intersections)) > 0:
                        sub_path_part.append(other_edge.id)
            sub_path_parts.append(sorted(sub_path_part))
        return sub_path_parts

    def _get_unified_unified_sub_paths(_sub_path_parts: list) -> list:
        _unified_sub_paths = []  # list of a group's edge.id that should be merged considering the to-exclude columns
        for i, sub_path_part in enumerate(_sub_path_parts):
            for j, other_sub_path_part in enumerate(_sub_path_parts):
                if i <= j and len(set(sub_path_part) & set(other_sub_path_part)) > 0:  # find the connected edges
                    union = sorted(set(sub_path_part + other_sub_path_part))
                    _unified_sub_paths = _merge_element_in_a_list(set(union), _unified_sub_paths)
                elif i <= j and len(set(sub_path_part) & set(other_sub_path_part)) == 0:
                    if set(sub_path_part) not in _unified_sub_paths:
                        _unified_sub_paths = _merge_element_in_a_list(set(sub_path_part), _unified_sub_paths)
        return _unified_sub_paths

    def _merge_element_in_a_list(element_of_concern: set, a_list: list[set]) -> list:
        considered = []
        for element in a_list:
            if element_of_concern != element and len(element_of_concern & element) > 0:
                union = set(sorted(list(element_of_concern) + list(element)))
                if union not in a_list:
                    a_list.append(set(sorted(list(element_of_concern) + list(element))))
                if element != union:
                    del a_list[a_list.index(element)]
                considered.append(element_of_concern)
        if element_of_concern not in considered and element_of_concern not in a_list:
            a_list.append(element_of_concern)
        return a_list

    def _get_paths_to_merge(groups: dict) -> list:
        _paths_to_merge = []  # list of gpds to merge
        for _, edge_group_ids in groups.items():
            sub_path_parts = _get_sub_path_parts(edge_group_ids)
            unified_sub_paths = _get_unified_unified_sub_paths(sub_path_parts)
            _paths_to_merge.extend(unified_sub_paths)
            _paths_to_merge = sorted([list(sorted(i)) for i in _paths_to_merge])
        return _paths_to_merge

    def _merge(gdf: GeoDataFrame, by: list, _aggfunc: dict, ntw: snkit.network.Network) -> GeoDataFrame:
        def _get_merged_in_a_loop(_merged: GeoDataFrame) -> GeoDataFrame:
            # 2.1.2. pick one with one intersection point
            start_path_extrms = [gdf_node_slice[gdf_node_slice['degree'] > 2].iloc[0].id]
            end_path_extrms = [gdf_node_slice[gdf_node_slice['degree'] > 2].iloc[1].id]
            _merged.from_id = [start_path_extremity for start_path_extremity in start_path_extrms]
            _merged.to_id = [end_path_extremity for end_path_extremity in end_path_extrms]
            return _merged

        def _get_merged_multiple_demand_edges(_merged: GeoDataFrame, path_extrms_nod_ids: set) -> GeoDataFrame:
            _mrgd = _get_split_edges_info(_merged)
            _mrgd.from_id = _mrgd.apply(lambda row: _get_node_id(row, 'from_id', path_extrms_nod_ids), axis=1)
            _mrgd.to_id = _mrgd.apply(lambda row: _get_node_id(row, 'to_id', path_extrms_nod_ids), axis=1)
            return _mrgd

        def _get_split_edges_info(_merged: GeoDataFrame) -> tuple:
            # used for the cases where demand nodes exist in the to-be-merged paths
            # make the demand node from_id of the merged edge
            dem_nod_ids = [
                i for i in set(gdf.from_id.tolist() + gdf.to_id.tolist())
                if (
                           gdf[gdf.demand_edge == 1].from_id.tolist() + gdf[gdf.demand_edge == 1].to_id.tolist()
                   ).count(i) == 2
            ]
            split_parts = [_merged['geometry'].iloc[0]]
            split_edges_gdf = gpd.GeoDataFrame(columns=_merged.columns)
            for dem_nod_id in dem_nod_ids:
                for part in split_parts:
                    part_splits, split_edges_gdf = _split(part, dem_nod_id, split_edges_gdf)
                    if part_splits is not None:
                        split_parts.extend(part_splits)
                        split_parts.remove(part)
            return split_edges_gdf

        def _split(_line_geom: Union[MultiLineString, LineString], dem_nod_id: int, splits_gdf: GeoDataFrame) -> tuple:
            # used for the cases where demand nodes exist in the to-be-merged paths
            dem_nod_geom = ntw.nodes[ntw.nodes.id == dem_nod_id].geometry.iloc[0]
            if _line_geom.contains(dem_nod_geom):
                if isinstance(_line_geom, MultiLineString):
                    coords = [linemerge(_line_geom).coords[0],
                              linemerge(_line_geom).coords[-1]]
                else:
                    coords = [_line_geom.coords[0],
                              _line_geom.coords[-1]]
                # Add the coords from the points
                coords += dem_nod_geom.coords
                # Calculate the distance along the line for each point
                dists = [linemerge(_line_geom).project(Point(p)) for p in coords]
                # sort the coordinates
                coords = [p for (d, p) in sorted(zip(dists, coords))]
                splits = [LineString([coords[i], coords[i + 1]]) for i in range(len(coords) - 1)]
                splits_gdf = _update_slite_edges_gdf(splits, dem_nod_id, splits_gdf, _line_geom)
                return splits, splits_gdf
            else:
                return None, splits_gdf

        def _update_slite_edges_gdf(parts: list, dem_nod_id: int, splt_edgs: GeoDataFrame,
                                    _split_line_geom: LineString) -> GeoDataFrame:
            # used for the cases where demand nodes exist in the to-be-merged paths
            for part in parts:
                # _split_line_geom is the line divided and produced parts
                if _split_line_geom not in splt_edgs.geometry.tolist():
                    part_gdf = gpd.GeoDataFrame({'geometry': part,
                                                 'id': len(splt_edgs),
                                                 'from_id': dem_nod_id,
                                                 'to_id': -1,
                                                 **_merged.drop(columns=['geometry', 'id', 'from_id', 'to_id'])
                                                 })
                else:
                    # if _split_line_geom is divided and n stored in splt_edgs, we need to retrieve from/to_id info
                    # and update splt_edgs
                    part_gdf = gpd.GeoDataFrame({'geometry': part,
                                                 'id': -1,
                                                 'from_id': splt_edgs[
                                                     splt_edgs.geometry == _split_line_geom].apply(
                                                     lambda row: row.from_id
                                                     if row.from_id != -1 else dem_nod_id, axis=1),
                                                 'to_id': splt_edgs[
                                                     splt_edgs.geometry == _split_line_geom].apply(
                                                     lambda row: row.to_id
                                                     if row.to_id != -1 else dem_nod_id, axis=1),
                                                 **_merged.drop(columns=['geometry', 'id', 'from_id', 'to_id'])
                                                 }
                                                )
                    _split_line_index = splt_edgs.loc[splt_edgs.geometry == _split_line_geom].index[0]
                    splt_edgs = splt_edgs.drop(_split_line_index)
                splt_edgs = pd.concat([splt_edgs, part_gdf], ignore_index=True)
            return splt_edgs

        def _get_node_id(r: GeoSeries, attr: str, path_extrms_nod_ids: set) -> int:
            # to fill from_id and to_id of the to-be-merged paths
            if r[attr] == -1:
                for path_extremities_node_id in path_extrms_nod_ids:
                    path_extremities_node_geom = ntw.nodes[ntw.nodes.id == path_extremities_node_id].geometry.iloc[0]
                    if r.geometry.intersects(path_extremities_node_geom):
                        return path_extremities_node_id
            else:
                return r[attr]

        def _get_merged_one_or_none_demand_edges(_merged, path_extrms_nod_ids: set) -> GeoDataFrame:
            _start_edges = gdf[gdf['intersections'].apply(lambda x: len(x) == 1)]
            if ('demand_edge' in gdf.columns) and (len(gdf[gdf['demand_edge'] == 1])) == 1:
                _start_edge = _start_edges[_start_edges.demand_edge == 1].iloc[0]
            elif ('demand_edge' not in gdf.columns) or (len(gdf[gdf['demand_edge'] == 1])) != 1:
                _start_edge = _start_edges.iloc[0]
            start_path_extrms = [_start_edge['from_id']
                                 if _start_edge['from_id'] in list(path_extrms_nod_ids)
                                 else _start_edge['to_id']]
            end_path_extrms = [(path_extrms_nod_ids - set(start_path_extrms)).pop()]
            _merged.from_id = [start_path_extremity for start_path_extremity in start_path_extrms]
            _merged.to_id = [end_path_extremity for end_path_extremity in end_path_extrms]
            return _merged

        # _merge function starts from here:
        gdf['intersections'] = gdf.apply(lambda x: _get_intersections(x, gdf), axis=1)
        _merged = gdf.dissolve(by=by, aggfunc=_aggfunc, sort=False)
        merged_id = _merged['id']  # the edge id of the merged edge
        if len(gdf) == 1:
            # 1. no merging is occurring
            start_path_extremities = [gdf.iloc[0]['from_id']]
            end_path_extremities = [gdf.iloc[0]['to_id']]
            _merged.from_id = start_path_extremities[0]
            _merged.to_id = end_path_extremities[0]
        else:
            # 2. merging is occurring
            if len(gdf[gdf['intersections'].apply(lambda x: len(x) == 1)]) == 0:
                # 2.1. a loop with two nodes degree > 2
                gdf_node_ids = list(set(gdf.from_id.tolist() + gdf.to_id.tolist()))
                gdf_node_slice = ntw.nodes[ntw.nodes['id'].isin(gdf_node_ids)]
                if len(gdf_node_slice[gdf_node_slice['degree'] > 2]) == 0:
                    # 2.1.1. a loop with only degree 2 edges => isolated from the rest of the graph
                    warnings.warn(f'''
                    A sub-graph loop isolated from the main graph is detected and removed.
                    This isolated part had {len(gdf_node_slice)} nodes with node_fids {gdf_node_slice.id.tolist()} in
                    the input node graph.
                    ''')
                    if 'demand_edge' in gdf.columns:
                        warnings.warn(f''''This sub-graph had these demand nodes {(
                                gdf[gdf.demand_edge == 1].from_id.tolist() +
                                gdf[gdf.demand_edge == 1].to_id.tolist()
                        )}''')
                    return gpd.GeoDataFrame(data=None, columns=ntw.edges.columns, crs=ntw.edges.crs)

                elif len(gdf_node_slice[gdf_node_slice['degree'] > 2]) == 1:
                    # 2.1.2. If there is only one node with the degree bigger than 2
                    if 'demand_edge' not in gdf.columns or len(gdf[gdf['demand_edge'] == 1]) == 0:
                        # No demand node is in this loop. Then omit this loop and return empty gdf
                        return gpd.GeoDataFrame(data=None, columns=ntw.edges.columns, crs=ntw.edges.crs)
                    elif 'demand_edge' in gdf.columns and len(gdf[gdf['demand_edge'] == 1]) > 0:
                        demand_node_ids = [
                            i for i in set(gdf.from_id.tolist() + gdf.to_id.tolist())
                            if (
                                       gdf[gdf.demand_edge == 1].from_id.tolist() +
                                       gdf[gdf.demand_edge == 1].to_id.tolist()
                               ).count(i) == 2
                        ]
                        if len(demand_node_ids) > 1:
                            return gdf  # merging this situation is skipped: not probable + complicated
                        else:
                            # Only one demand node exists in the loop
                            if isinstance(linemerge(_merged.geometry.iloc[0]), MultiLineString):
                                # to exclude the merged geoms for which linemerge does not work
                                return gdf
                            path_extremities_node_ids = {x for x in
                                                         gdf_node_slice[gdf_node_slice['degree'] > 2].id.tolist()
                                                         + demand_node_ids
                                                         }
                            _merged = _get_merged_multiple_demand_edges(_merged, path_extremities_node_ids)
                else:
                    # 2.1.3. the only remaining option is two nodes with degrees bigger than 2
                    if 'demand_edge' not in gdf.columns or len(gdf[gdf['demand_edge'] == 1]) == 0:
                        # No demand node is in this loop. Then merge
                        _merged = _get_merged_in_a_loop(_merged)
                    else:
                        return gdf
            else:
                # 2.2. merging non-loop paths
                path_extremities_node_ids = {i for i in set(gdf.from_id.tolist() + gdf.to_id.tolist())
                                             if (gdf.from_id.tolist() + gdf.to_id.tolist()).count(i) == 1}
                # if len(path_extremities_node_ids) > 0:
                if ('demand_edge' in gdf.columns) and (len(gdf[gdf['demand_edge'] == 1]) > 1):
                    _merged = _get_merged_multiple_demand_edges(_merged, path_extremities_node_ids)
                elif (('demand_edge' in gdf.columns and len(gdf[gdf['demand_edge'] == 1]) <= 1) or
                      ('demand_edge' not in gdf.columns)):
                    # 2.2.2.no dem node is in the to_be_merged path or only one dem node. In the later case dem node
                    # will not be dissolved because it is in the path_extremities_node_ids
                    _merged = _get_merged_one_or_none_demand_edges(_merged, path_extremities_node_ids)
                # else:
                #     raise Warning(f"""Check the lines with the following ids {gdf.id.tolist()} """)

            merged_id = 'to_be_updated'  # the edge id of the merged edge will be updated later
        _merged.id = merged_id
        _merged.crs = gdf.crs
        return _merged

    # _get_merged_paths starts here
    grouped_edges = edges.groupby(excluded_edge_types)
    if len(grouped_edges.groups) == 1:
        merged_edges = _merge(gdf=edges, by=excluded_edge_types, _aggfunc=aggfunc, ntw=net)
    else:
        merged_edges = gpd.GeoDataFrame(columns=edges.columns, crs=edges.crs)  # merged edges
        edge_groups = edges.groupby(excluded_edge_types).groups
        paths_to_merge = _get_paths_to_merge(edge_groups)

        for path_indices in paths_to_merge:
            path_to_merge = edges[edges['id'].isin(path_indices)].copy()  # indices of the edges in edges gdf
            merged = _merge(gdf=path_to_merge, by=excluded_edge_types, _aggfunc=aggfunc, ntw=net)
            merged_edges = pd.concat([merged_edges, merged], ignore_index=True)

        merged_edges.crs = edges.crs

    return merged_edges


def _get_intersections(_edge, _edges):
    intersections = []
    edge_geometry = _edge.geometry.simplify(tolerance=1e-8)

    for other_edge_index, other_edge in _edges.iterrows():
        other_edge_geometry = other_edge.geometry.simplify(tolerance=1e-8)

        if not edge_geometry.equals(other_edge_geometry):  # avoid self-intersection
            intersection = edge_geometry.intersection(other_edge_geometry)

            if not intersection.is_empty and any(
                    intersection.intersects(boundary) for boundary in edge_geometry.boundary.geoms):
                if isinstance(intersection, MultiPoint):
                    intersections.extend(
                        [point.coords[0] for point in intersection.geoms if
                         point in other_edge_geometry.boundary.geoms])
                else:
                    intersections.append(intersection.coords[0])

    return sorted(intersections, key=lambda x: x[0])


def merge_edges(net: snkit.network.Network, aggfunc: Union[str, dict], by: Union[str, list],
                id_col="id") -> snkit.network.Network:
    def _get_edge_ids_to_update(edges_list: list) -> list:
        ids_to_update = []
        for edges in edges_list:
            ids_to_update.extend(edges.id.tolist())
        return ids_to_update

    if "degree" not in net.nodes.columns:
        net.nodes["degree"] = net.nodes[id_col].apply(
            lambda x: node_connectivity_degree(x, net)
        )

    degree_2 = list(net.nodes[id_col].loc[net.nodes.degree == 2])
    degree_2_set = set(degree_2)
    edge_paths = _get_edge_paths(degree_2_set, net)

    edge_ids_to_update = _get_edge_ids_to_update(edge_paths)
    edges_to_keep = net.edges[~net.edges['id'].isin(edge_ids_to_update)]

    updated_edges = _get_merged_edges(paths_to_group=edge_paths, by=by, aggfunc=aggfunc, net=net)
    updated_edges = updated_edges.reset_index(drop=True)

    new_edges = pd.concat([edges_to_keep, updated_edges], ignore_index=True)
    new_edges.id = range(len(new_edges))
    new_edges = new_edges.reset_index(drop=True)

    nodes_to_keep = list(set(new_edges.from_id.tolist() + new_edges.to_id.tolist()))
    new_nodes = net.nodes[net.nodes[id_col].isin(nodes_to_keep)]
    new_nodes = new_nodes.reset_index(drop=True)

    return Network(nodes=new_nodes, edges=new_edges)


def _get_edge_paths(node_set: set, net: snkit.network.Network) -> list:
    edge_paths = []

    while node_set:
        popped_node = node_set.pop()
        node_path = {popped_node}
        candidates = {popped_node}
        while candidates:
            popped_cand = candidates.pop()
            matches = set(
                np.unique(
                    net.edges[["from_id", "to_id"]]
                    .loc[
                        (net.edges.from_id == popped_cand)
                        | (net.edges.to_id == popped_cand)
                        ]
                    .values
                )
            )
            matches.remove(popped_cand)
            matches = matches - node_path
            for match in matches:
                if match in node_set:
                    candidates.add(match)
                    node_path.add(match)
                    node_set.remove(match)
                else:
                    node_path.add(match)
        if len(node_path) >= 2:
            edge_paths.append(
                net.edges.loc[
                    (net.edges.from_id.isin(node_path))
                    & (net.edges.to_id.isin(node_path))
                    ]
            )
    return edge_paths


def _simplify_tracks(net: snkit.network.Network, buffer_distance: float, hole_area_threshold: float):
    net = _drop_hanging_nodes(net)
    unified_buffer_gdf = _unified_buffer_tracks(net, buffer_distance)
    unified_buffer_gdf = get_largest_polygon(unified_buffer_gdf)
    unified_buffer_gdf = _remove_small_holes(unified_buffer_gdf, hole_area_threshold)
    triangulation_data = _triangulate_polygon(unified_buffer_gdf)
    triangles_gdf = _create_triangle_gdf(triangulation_data, unified_buffer_gdf)
    raise NotImplementedError()


def _drop_hanging_nodes(net, tolerance=1):
    """
    inspired from trail

    Args:
        net (class): A network composed of nodes (points in space) and edges (lines)
        tolerance (float, optional): The maximum allowed distance from hanging nodes to the network. Defaults to 0.005.

    Returns:
        network (class): A network composed of nodes (points in space) and edges (lines)

    """

    def _get_edges_with_hanging_nodes(egs: GeoDataFrame, hng_nds: np.ndarray) -> GeoDataFrame:
        to_ids = egs['to_id'].to_numpy()
        from_ids = egs['from_id'].to_numpy()
        to_ids_ind = [(net.nodes['id'] == to_id).idxmax() for to_id in to_ids]
        from_ids_ind = [(net.nodes['id'] == from_id).idxmax() for from_id in from_ids]
        hng_to = np.isin(to_ids_ind, hng_nds)
        hng_from = np.isin(from_ids_ind, hng_nds)
        # eInd : An array containing the indices of edges that connect the degree 1 node
        edge_ind = np.hstack((np.nonzero(hng_to), np.nonzero(hng_from)))
        return egs.iloc[np.sort(edge_ind[0])]

    net = _get_nodes_degree(net)
    ed = net.edges.copy()
    deg = net.nodes['degree'].to_numpy()
    # hang_nodes : An array of the indices of nodes with degree 1
    hang_nodes = np.where(deg == 1)
    deg_ed = _get_edges_with_hanging_nodes(ed, hang_nodes)
    edge_id_drop = []
    if 'demand_edge' in deg_ed.columns:
        while len(deg_ed[deg_ed['demand_edge'] != 1]) > 0:  # while there are edges with deg 1 node, excluding dem_edges
            for d in deg_ed[~(deg_ed['id'].isin(edge_id_drop))].itertuples():
                dist = shapely.measurement.length(d.geometry)
                # If the edge is shorter than the tolerance
                # add the ID to the drop list and update involved node degrees
                if dist < tolerance and d.demand_edge != 1:
                    edge_id_drop.append(d.id)
                    deg[(net.nodes['id'] == d.from_id).idxmax()] -= 1
                    deg[(net.nodes['id'] == d.to_id).idxmax()] -= 1
            edges_copy = net.edges.copy()
            edg = edges_copy.loc[~(edges_copy.id.isin(edge_id_drop))]
            hang_nodes = np.where(deg == 1)
            deg_ed = _get_edges_with_hanging_nodes(edg, hang_nodes)
    else:
        while len(deg_ed) > 0:  # while there are edges with deg 1 node, excluding dem_edges
            for d in deg_ed[~(deg_ed['id'].isin(edge_id_drop))].itertuples():
                dist = shapely.measurement.length(d.geometry)
                # If the edge is shorter than the tolerance
                # add the ID to the drop list and update involved node degrees
                if dist < tolerance:
                    edge_id_drop.append(d.id)
                    deg[(net.nodes['id'] == d.from_id).idxmax()] -= 1
                    deg[(net.nodes['id'] == d.to_id).idxmax()] -= 1
            edges_copy = net.edges.copy()
            edg = edges_copy.loc[~(edges_copy.id.isin(edge_id_drop))]
            hang_nodes = np.where(deg == 1)
            deg_ed = _get_edges_with_hanging_nodes(edg, hang_nodes)

    edg = ed.loc[~(ed.id.isin(edge_id_drop))].reset_index(drop=True)
    edg.drop(labels=['id'], axis=1, inplace=True)
    edg['id'] = range(len(edg))
    n = net.nodes.copy()
    # Degree 0 Nodes are cleaned in the merge_2 method
    nod = n.iloc[deg > 0].reset_index(drop=True)
    nod['degree'] = deg[deg > 0]
    return Network(nodes=nod, edges=edg)


def _unified_buffer_tracks(net: snkit.network.Network, buffer_distance_km: float) -> GeoDataFrame:
    buffer_distance_deg = _km_distance_to_degrees(buffer_distance_km)
    buffered_data = net.edges.set_crs(net.nodes.crs).geometry.buffer(distance=buffer_distance_deg, cap_style='square')
    data = [{'id': 1, 'geometry': unary_union(buffered_data)}]
    buffered_gdf = gpd.GeoDataFrame(data, geometry='geometry', crs=net.nodes.crs)
    return buffered_gdf


def get_largest_polygon(input_poly_gdf: GeoDataFrame) -> GeoDataFrame:
    largest_polygon = None
    max_area = 0.0

    for polygon in input_poly_gdf.loc[0, 'geometry'].geoms:
        area = polygon.area
        if area > max_area:
            max_area = area
            largest_polygon = polygon

    return gpd.GeoDataFrame({'id': [1], 'geometry': MultiPolygon([largest_polygon])},
                            geometry='geometry', crs='EPSG:4326')


def _remove_small_holes(input_poly_gdf: GeoDataFrame, hole_area_threshold: float) \
        -> GeoDataFrame:
    """
    Remove small holes from a buffered polygon that are smaller than the specified threshold.
    """

    def _km_area_to_degrees(area_km: float):
        return area_km * 1 / 111.32 * 1 / 111.32

    if isinstance(input_poly_gdf.loc[0, 'geometry'], Polygon):
        raise NotImplementedError("Not implemented yet")

    list_interiors = []
    new_polygons = []
    hole_threshold_degree = _km_area_to_degrees(hole_area_threshold)

    multi_polygon = input_poly_gdf.loc[0, 'geometry']
    for polygon in multi_polygon.geoms:
        for interior in polygon.interiors:
            p = Polygon(interior)
            if p.area > hole_threshold_degree:
                list_interiors.append(interior)

        new_polygons.append(Polygon(polygon.exterior.coords, holes=list_interiors))
    return gpd.GeoDataFrame({'geometry': [MultiPolygon([new_polygon for new_polygon in new_polygons])]},
                            geometry='geometry', crs=input_poly_gdf.crs)


def _triangulate_polygon(poly_gdf: gpd.GeoDataFrame) -> dict:
    """
    Triangulate a polygon using constrained Delaunay triangulation.
    """
    poly_geom = poly_gdf.loc[0, 'geometry']
    _triangles = []
    _triangulation_data = {}

    if isinstance(poly_geom, MultiPolygon):
        for p in poly_geom.geoms:
            triangulation = _triangulation(p)
            new_triangulation_data = _filter_triangles(p, triangulation)
            _triangulation_data = _update_dict(_triangulation_data, new_triangulation_data)
    else:
        triangulation = _triangulation(poly_geom)
        new_triangulation_data = _filter_triangles(poly_geom, triangulation)
        _triangulation_data = _update_dict(_triangulation_data, new_triangulation_data)

    return _triangulation_data


def _triangulation(polygon: Polygon) -> dict:
    """
    Triangulate a single polygon using constrained Delaunay triangulation.
    """
    boundary = polygon.boundary
    coords = []
    _triangle_geoms = []
    if isinstance(boundary, LineString):
        coords = list(boundary.coords)
    [coords.extend(list(b.coords)) for b in boundary.geoms]
    segments = [(i, i + 1) for i in range(len(coords) - 1)]
    segments.append((len(coords) - 1, 0))  # Close the polygon

    return tr.triangulate({'vertices': coords, 'segments': segments})


def _update_dict(existing_dict: dict, new_dict: dict) -> dict:
    updated_dict = existing_dict
    for key, new_value in new_dict.items():
        if key in updated_dict:
            updated_dict[key].extend(new_value)
        else:
            updated_dict[key] = new_value
    return updated_dict


def _filter_triangles(polygon: Union[Polygon, MultiPolygon], triangulation_result: dict) -> dict:
    # Filter on the triangles to exclude the external and holes' triangles
    _triangles = []
    _sides = []
    _triangle_types = []

    for _, tri in enumerate(triangulation_result['triangles']):
        _tri_coords = [triangulation_result['vertices'][vertex_id] for vertex_id in tri]
        _tri_sides = [LineString([_tri_coords[i], _tri_coords[(i + 1) % 3]]) for i, vertex_id in enumerate(tri)]

        edge_polygon_intersections = {edge.intersection(polygon) for edge in _tri_sides}
        sum_edges_touching_triangulated_boundary = sum(isinstance(edge_polygon_intersection, (Point, MultiPoint)) for
                                                       edge_polygon_intersection in edge_polygon_intersections)
        sum_edges_within_polygon = sum(polygon.contains(edge_polygon_intersection)
                                       for edge_polygon_intersection in edge_polygon_intersections)
        if sum_edges_touching_triangulated_boundary >= 2 and sum_edges_within_polygon == 0:
            continue
        elif sum_edges_touching_triangulated_boundary != 3 and (3 > sum_edges_within_polygon > 0):
            _triangle_types.append("normal")
        elif sum_edges_touching_triangulated_boundary == 1 and sum_edges_within_polygon == 0:
            _triangle_types.append("normal")
        elif sum_edges_touching_triangulated_boundary == 0 and \
                (sum_edges_within_polygon == 3 or sum_edges_within_polygon == 0):
            _triangle_types.append("interior")

        _triangles.append(Polygon(_tri_coords))
        _sides.append(_tri_sides)

    return {
        "triangle_geometries": _triangles,
        "sides": _sides,
        "triangle_types": _triangle_types,
    }


def _create_triangle_gdf(_triangulation_data: dict, polygon: GeoDataFrame) -> GeoDataFrame:
    def create_triangle_row(idx: int, triangle: Polygon, _triangulation_data: dict) -> dict:
        return {
            'id': idx,
            'geometry': triangle,
            'track_edges_geom': [],
            'type': _triangulation_data["triangle_types"][idx],
            'constellation_type': ''
        }

    rows = (create_triangle_row(idx, triangle, _triangulation_data) for idx, triangle in
            enumerate(_triangulation_data['triangle_geometries']))
    triangle_gdf = gpd.GeoDataFrame.from_records(rows)
    triangle_gdf = triangle_gdf.set_geometry('geometry')
    triangle_gdf = triangle_gdf.set_crs(polygon.crs)
    return triangle_gdf


# def trace_track_edges(triangles: list, net: snkit.network.Network) -> snkit.network.Network:
#     for triangle in triangles:
#         affected_track_edges = _identify_intersected_track_edges(triangle, net.edges)
#         for track_edge in affected_track_edges:
#             cut_track_edges_at_triangle_sides(track_edge, triangle)
#             # if triangle_type(triangle) == 1:
#             #     create_skeleton_edge(triangle)
#             # else:
#             #     insert_center_point(triangle)
#             #     link_to_midpoints(triangle)
#
#     return updated_track_data


def _identify_intersected_track_edges(triangle: list, net: snkit.network.Network) -> list:
    # Use spatial operations to find track_edges within the triangle's area
    affected_track_edges = []
    triangle_polygon = Polygon(triangle)  # Assuming triangle is a list of points
    for _, track_edge in net.edges.iterrows():
        if track_edge.geometry.intersects(triangle_polygon):
            affected_track_edges.append(track_edge)
    return affected_track_edges


def cut_track_edges_at_triangle_sides(track_edges: GeoDataFrame, triangulation_data: dict) -> GeoDataFrame:
    cut_track_edges = []
    for tri in triangulation_data['triangles']:
        cut_result = gpd.overlay(track_edges, gpd.GeoDataFrame(geometry=[tri]), how='intersection')
        cut_track_edges.append(cut_result)

    cut_edges_gdf = gpd.GeoDataFrame(pd.concat(cut_track_edges, ignore_index=True))

    return cut_track_edges
