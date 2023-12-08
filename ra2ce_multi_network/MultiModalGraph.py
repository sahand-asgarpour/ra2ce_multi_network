# ToDo: adjust the route finding algorithm to find routes on given graph types: rail, road, all
from typing import Union, Any

import networkx as nx
import numpy as np
import pandas as pd
import pyproj
from pathlib import Path
import geopandas as gpd
from networkx.classes.multigraph import Graph, MultiGraph
from geopandas.geodataframe import GeoDataFrame

from ra2ce.analyses.indirect.analyses_indirect import save_gdf
from ra2ce.analyses.analysis_config_data.enums.weighing_enum import WeighingEnum
from ra2ce.analyses.indirect.analyses_indirect import IndirectAnalyses
from ra2ce.graph.graph_files.graph_files_collection import GraphFilesCollection
from ra2ce.graph.network_config_data.network_config_data import NetworkConfigData
from ra2ce.analyses.analysis_config_data.enums.analysis_direct_enum import AnalysisDirectEnum
from ra2ce.analyses.analysis_config_data.enums.analysis_indirect_enum import AnalysisIndirectEnum
from ra2ce.analyses.analysis_config_data.analysis_config_data import (
    AnalysisConfigData,
    AnalysisSectionBase,
    AnalysisSectionDirect,
    AnalysisSectionIndirect,
    DirectAnalysisNameList,
    IndirectAnalysisNameList,
)


class MultiModalGraph:
    def __init__(
            self,
            od_file: Path,
            graph_types: dict[str, Graph],
            crs: pyproj.CRS,
            graphs_to_add_attributes=None
    ) -> None:
        if graphs_to_add_attributes is None:
            graphs_to_add_attributes = []
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

        # ToDo: length is meter? use ra2ce function created for conversion
        def convert_maxspeed(speed: Union[str, list[Union[str, None]]]) -> list[Union[float, None]]:
            if isinstance(speed, list):
                return [pd.to_numeric(s, errors='coerce') for s in speed]
            else:
                return [pd.to_numeric(speed, errors='coerce')]

        def get_average_max_speed(graph: nx.Graph) -> Union[float, None]:
            all_max_speeds = []

            for _, _, edge_data in graph.edges(data=True):
                maxspeed_values = edge_data.get('maxspeed', [None])

                if edge_data.get('maxspeed') is not None:
                    converted_speeds = convert_maxspeed(maxspeed_values)
                    non_nan_speeds = [val for val in converted_speeds if not pd.isna(val)]

                    if non_nan_speeds:
                        all_max_speeds.extend(non_nan_speeds)

            if all_max_speeds:
                return np.nanmean(all_max_speeds)  # Use np.nanmean to handle NaN values
            else:
                return None

        # Calculate the mean of all_max_speeds
        average_max_speed = get_average_max_speed(graph)

        # Iterate over edges
        for u, v, data in graph.edges(data=True):
            # Check if 'length' is already defined, if not, calculate and assign
            # ToDo: length is meter? use ra2ce function created for conversion
            data.setdefault('length', data['geometry'].length)

            # Get max_speed as numeric
            max_speed = pd.to_numeric(data.get('maxspeed', average_max_speed), errors='coerce')

            # If max_speed is an array, use np.nanmean to calculate the mean excluding NaN
            max_speed = np.nanmean(max_speed) if isinstance(max_speed, np.ndarray) else max_speed

            # Replace with average_max_speed if avg_max_speed is None or NaN
            max_speed = average_max_speed if pd.isna(max_speed) or max_speed is None else max_speed

            # If max_speed is NaN or None, set the default value using the calculated mean
            data.setdefault('max_speed', max_speed)

            # Calculate and set the 'time' attribute
            data.setdefault('time', data['length'] / max_speed)
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
                        self.multi_modal_graph.add_edge(node_g, other_g_node, **{"edge_type": "terminal",
                                                                                 "time": 10e-1000,
                                                                                 "distance": 10e-1000
                                                                                 })

                    if not self.multi_modal_graph.has_edge(other_g_node, node_g):
                        self.multi_modal_graph.add_edge(other_g_node, node_g, **{"edge_type": "terminal",
                                                                                 "time": 10e-1000,
                                                                                 "distance": 10e-1000
                                                                                 })

    def _configure_analysis(self, modes: list, project_input: dict, analysis_path: Path) -> dict:
        accepted_modes = list(self.graph_types.keys()) + ["multi_modal"]
        config = {}
        _analyses_config = AnalysisConfigData(
            root_path=analysis_path,
            input_path=analysis_path.joinpath(Path("input")),
            static_path=analysis_path.joinpath(Path("static")),
            output_path=analysis_path.joinpath(Path("output"))
        )
        _analyses_config.analyses = self.get_analysis_sections(project_input)
        _analyses_config.project.name = project_input["project_name"]
        for mode in modes:
            if not mode in accepted_modes:
                raise ValueError(f"{mode} is not within the accepted modes of {accepted_modes}")
            mode_analyses_config = _analyses_config
            mode_analyses = IndirectAnalyses(config=mode_analyses_config, graph_files=GraphFilesCollection())
            if mode != "multi_modal":
                mode_analyses.graph_files.origins_destinations_graph = self.graph_types[mode]
            else:
                mode_analyses.graph_files.origins_destinations_graph = self.multi_modal_graph

            mode_analyses = IndirectAnalyses(config=mode_analyses_config, graph_files=mode_analyses.graph_files)
            config[mode] = {
                "analyses": mode_analyses,
                "analyses_config": mode_analyses_config

            }
        return config

    @staticmethod
    def get_analysis_sections(project_setting: dict) -> list[AnalysisSectionBase]:
        """
        based on get_project_section in C:\repos\ra2ce\ra2ce\analyses\analysis_config_data
        """
        _analysis_sections = []

        for analysis_setting in project_setting["analysis_settings"]:
            _analysis_type = analysis_setting["analysis"]
            print(_analysis_type)
            if _analysis_type in DirectAnalysisNameList:
                analysis_setting["analysis"] = AnalysisDirectEnum.get_enum(_analysis_type)
                _analysis_section = AnalysisSectionDirect(**analysis_setting)
                raise NotImplementedError("Direct analysis is not implemented yet")
            elif _analysis_type in IndirectAnalysisNameList:
                analysis_setting["analysis"] = AnalysisIndirectEnum.get_enum(_analysis_type)
                if analysis_setting["weighing"] == "distance":
                    analysis_setting["weighing"] = WeighingEnum.LENGTH
                else:
                    analysis_setting["weighing"] = WeighingEnum.get_enum(analysis_setting["weighing"])
                _analysis_section = AnalysisSectionIndirect(**analysis_setting)
            else:
                raise ValueError(f"Analysis {_analysis_type} not supported.")
            _analysis_sections.append(_analysis_section)

        return _analysis_sections

    def run_analysis(self, modes: list, project_input: dict, analysis_path: Path):
        _config = self._configure_analysis(modes=modes,
                                           project_input=project_input,
                                           analysis_path=analysis_path)
        for mode, mode_config in _config.items():
            mode_analyses = mode_config["analyses"]
            mode_analyses_config = mode_config["analyses_config"]
            for mode_analysis_config in mode_analyses_config.analyses:
                func = getattr(mode_analyses, mode_analysis_config.analysis.config_value)
                result = func(mode_analyses.graph_files.origins_destinations_graph, mode_analysis_config)
                result = result[~result['geometry'].is_empty]
                self._save_results(mode_analyses, result, analysis_path)

    @staticmethod
    def _save_results(analyses: IndirectAnalyses, analysis_results: GeoDataFrame, to_save_path: Path):
        output_path = to_save_path.joinpath(Path("output"))
        for analysis in analyses.config.indirect:
            result_path = output_path.joinpath(Path(analysis.analysis.name.lower()))
            # according to origin_closest_destination procedure in
            # C:\repos\ra2ce\ra2ce\analyses\indirect\analysis_indirect.py
            if analysis.save_gpkg:
                save_gdf(analysis_results, Path(str(result_path) + ".gpkg"))
            if analysis.save_csv:
                analysis_results.to_csv(Path(str(result_path) + ".gpkg"), index=False)
