# ToDo: adjust the route finding algorithm to find routes on given graph types: rail, road, all
import networkx as nx
import pandas as pd
import pyproj
from pathlib import Path
import geopandas as gpd
from networkx.classes.multigraph import Graph, MultiGraph
from geopandas.geodataframe import GeoDataFrame


class MultiModalGraph:
    def __init__(
            self,
            od_file: Path,
            graph_types: dict[str, Graph],
            crs: pyproj.CRS,
            graphs_to_add_attributes: list = []
    ) -> None:
        self.crs = crs
        self.multi_modal_graph: Graph = nx.Graph(crs=self.crs)
        if len(graphs_to_add_attributes) > 0:
            self.graphs_to_add_attributes = graphs_to_add_attributes
        self.graph_types: dict[str, Graph] = self._update_separate_graphs(graph_types)
        self.od_gdf: GeoDataFrame = gpd.read_file(od_file)
        self.od_multi_modal_ods: set = self._filter_ods()

        # Dynamically create attributes for mapped ods based on graph types
        for graph_name in graph_types:
            self._find_graph_mapped_ods(graph_name)

        self._find_corresponding_multi_modal_nodes()

    def _update_separate_graphs(self, g_types: dict):
        for g_type, g in g_types.items():
            g.graph['crs'] = self.crs
            # Add length, max_speed and time to the edges of the graph_type. User should state which graph_types
            # should go through this step
            if hasattr(self, 'graphs_to_add_attributes') and g_type in self.graphs_to_add_attributes:
                g = self._add_time_speed_attributes(g)
            g = self._rename_graphs(g_type, g)
            g_types[g_type] = g
        return g_types

    def _add_time_speed_attributes(self, graph: Graph) -> Graph:
        # Add length, max_speed and time to the edges
        for u, v, data in graph.edges(data=True):
            # Check if 'length' is already defined, if not, calculate and assign
            data.setdefault('length', data['geometry'].length)

            max_speed = pd.to_numeric(data.get('maxspeed', 0), errors='coerce')
            data.setdefault('max_speed', max_speed if not pd.isna(max_speed) else max_speed.mean())

            data.setdefault('time', data['length'] / data['max_speed'])
        return graph

    def _rename_graphs(self, graph_type: str, graph: Graph) -> Graph:
        # Relabel nodes
        node_mapping = {old_node: f'{graph_type}_{old_node}' for old_node in graph.nodes}
        graph = nx.relabel_nodes(graph, node_mapping)

        # Add node_type attributes
        for node, data in graph.nodes(data=True):
            data['node_type'] = graph_type

        # Add edge_type attributes
        for u, v, data in graph.edges(data=True):
            data['edge_type'] = graph_type

        return graph

    def _filter_ods(self, attr: str = 'multi_modal_terminal', o_id_column: str = 'o_id', d_id_column: str = 'd_id') \
            -> set:
        _filtered_ods: set = set()
        for _, row in self.od_gdf[self.od_gdf[attr] == 1].iterrows():
            _filtered_ods.add(row[o_id_column])
            _filtered_ods.add(row[d_id_column])
        return _filtered_ods

    def _find_graph_mapped_ods(self, graph_type: str, od_id_column: str = "od_id") -> None:
        # Ensure the provided graph type is valid
        if graph_type not in self.graph_types:
            raise ValueError(f"Invalid graph type: {graph_type}")

        # Find nodes in the specified graph type with 'O_number' or 'D_number' in 'od_id' attribute
        od_graph_mapped_nodes: list[str] = [
            node for node, data in self.graph_types[graph_type].nodes(data=True)
            if not pd.isna(data.get(od_id_column, '')) and any(
                f'O_{i}' in data.get(od_id_column, '').split(',') or
                f'D_{i}' in data.get(od_id_column, '').split(',')
                for i in range(0, len(self.od_gdf) + 1)
                if f'O_{i}' in data.get(od_id_column, '') or f'D_{i}' in data.get(od_id_column, '')
            )
        ]

        # Filter mapped ods for the specified graph type
        mapped_multi_modal_ods, mapped_single_modal_ods = (
            self._filter_mapped_ods(self.graph_types[graph_type], od_graph_mapped_nodes, True),
            self._filter_mapped_ods(self.graph_types[graph_type], od_graph_mapped_nodes, False)
        )

        # Store the results based on the specified graph type
        setattr(self, f'{graph_type}_mapped_multi_modal_ods', mapped_multi_modal_ods)
        setattr(self, f'{graph_type}_mapped_single_modal_ods', mapped_single_modal_ods)

    def _filter_mapped_ods(
            self,
            graph: Graph,
            mapped_ods: list[str],
            is_multi_modal_ods: bool = True,
            od_id_column: str = "od_id"
    ) -> dict[str, str]:
        return {
            node: data[od_id_column] for node, data in graph.nodes(data=True)
            if node in mapped_ods and any(
                (element in self.od_multi_modal_ods) == is_multi_modal_ods
                for element in data.get('od_id', "").split(',')
            )
        }

    def _find_corresponding_multi_modal_nodes(self):
        for g_type in self.graph_types:
            corresponding_multi_modal_ods = {}
            mapped_multi_modal_ods: dict[str, str] = getattr(self, f'{g_type}_mapped_multi_modal_ods')
            for u, ods in mapped_multi_modal_ods.items():
                closest_graph_nodes = set()
                shared_ods = set(ods.split(','))
                for other_g_type in set(self.graph_types.keys()) - {g_type}:
                    other_mapped_multi_modal_ods: dict[str, str] \
                        = getattr(self, f'{other_g_type}_mapped_multi_modal_ods')
                    for v, other_ods in other_mapped_multi_modal_ods.items():
                        other_ods_set = set(other_ods.split(','))
                        # Check for at least one shared element
                        if shared_ods & other_ods_set:
                            closest_graph_nodes.add(v)
                            shared_ods.update(other_ods_set)
                corresponding_multi_modal_ods[u] = {
                    "corresponding_multi_modal_od": closest_graph_nodes,
                    "shared_ods": shared_ods
                }
            setattr(self, f'{g_type}_corresponding_multi_modal_ods', corresponding_multi_modal_ods)

    def create_multi_modal_graph(self):
        graph_types = self.graph_types.values()
        self.multi_modal_graph = self._join_graphs()
        self._connect_joined_graphs_graphs()
        return self.multi_modal_graph

    def _join_graphs(self) -> Graph:
        joined_g = nx.MultiDiGraph()
        for _, g in self.graph_types.items():
            joined_g.add_nodes_from(g.nodes(data=True))
            joined_g.add_edges_from(g.edges(data=True))
            if isinstance(g, (nx.Graph, nx.MultiGraph)) and not isinstance(g, (nx.DiGraph, nx.MultiDiGraph)):
                # add the reverse edge from the union graph
                for u, v, data in g.edges(data=True):
                    if not joined_g.has_edge(v, u):
                        joined_g.add_edge(v, u, **data)

        return joined_g

    def _connect_joined_graphs_graphs(self):
        #  add bidirectional virtual edges between multi_modal terminals
        for g_type in self.graph_types:
            for node_g, corresponding_multi_modal_ods_data in (
                    getattr(self, f'{g_type}_corresponding_multi_modal_ods').items()):
                other_g_nodes = corresponding_multi_modal_ods_data["corresponding_multi_modal_od"]

                for other_g_node in other_g_nodes:
                    if not self.multi_modal_graph.has_edge(node_g, other_g_node):
                        self.multi_modal_graph.add_edge(node_g, other_g_node, **{"edge_type": "terminal"})

                    if not self.multi_modal_graph.has_edge(other_g_node, node_g):
                        self.multi_modal_graph.add_edge(other_g_node, node_g, **{"edge_type": "terminal"})

    def find_optimal_route_origin_destination(self):
        # create list of origin-destination pairs
        od_nodes = self._get_origin_destination_pairs(graph)
        pref_routes = find_route_ods(graph, od_nodes, analysis["weighing"])
        return pref_routes
