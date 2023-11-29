# ToDo: add an attribute to od stating multi_modal_terminal
# ToDo: rerun the od mapping for both merged rail and simplified road
# ToDo: Define a function to detect the mapped multi_modal_ods in the separate graphs and connect the closest ones

from pathlib import Path
from typing import List, Dict, Tuple
import networkx as nx
import geopandas as gpd
from networkx.classes.multigraph import MultiGraph
from shapely.geometry import Point, MultiLineString
from shapely.ops import linemerge
from geopandas.geodataframe import GeoDataFrame


class MultiModalGraph:
    def __init__(
            self,
            od_file: Path,
            graph_types: Dict[str, MultiGraph],
            od_id_column: str = "id"
    ) -> None:
        self.graph_types: Dict[str, MultiGraph] = graph_types
        self.od_gdf: GeoDataFrame = gpd.read_file(od_file)
        self.od_id_column: str = od_id_column
        self._od_multi_modal_ods: Dict[str, Tuple[str, str]] = self._filter_ods()

        # Dynamically create attributes for mapped ods based on graph types
        for graph_type in graph_types:
            setattr(self, f'{graph_type}_mapped_single_modal_ods', None)
            setattr(self, f'{graph_type}_mapped_multi_modal_ods', None)

    def _filter_ods(self, attr: str = 'multi_modal_terminal') -> Dict[str, Tuple[str, str]]:
        filtered_ods: Dict[str, Tuple[str, str]] = {}
        for index, row in self.od_gdf[self.od_gdf[attr] == 1].iterrows():
            filtered_ods[row[self.od_id_column]] = (row['o_id'], row['d_id'])
        return filtered_ods

    def _filter_mapped_ods(
            self,
            graph: MultiGraph,
            mapped_ods: List[str],
            is_multi_modal_ods: bool = True
    ) -> Dict[str, str]:
        return {
            node: data['od_id'] for node, data in graph.nodes(data=True)
            if node in mapped_ods and
               any(
                   f'O_{i}' in data['od_id'].split(',') or
                   f'D_{i}' in data['od_id'].split(',')
                   for i in range(1, len(self.od_gdf) + 1)
                   if f'O_{i}' in data['od_id'] or f'D_{i}' in data['od_id']
               ) and (node in self._od_multi_modal_ods.values()) == is_multi_modal_ods
        }

    def find_graph_mapped_ods(self, graph_type: str) -> None:
        # Ensure the provided graph type is valid
        if graph_type not in self.graph_types:
            raise ValueError(f"Invalid graph type: {graph_type}")

        # Find nodes in the specified graph type with 'O_number' or 'D_number' in 'od_id' attribute
        od_graph_mapped_nodes: List[str] = [
            node for node, data in self.graph_types[graph_type].nodes(data=True)
            if any(
                f'O_{i}' in data.get('od_id', '').split(',') or
                f'D_{i}' in data.get('od_id', '').split(',')
                for i in range(0, len(self.od_gdf) + 1)
                if f'O_{i}' in data.get('od_id', '') or f'D_{i}' in data.get('od_id', '')
            )
        ]

        # Filter mapped ods for the specified graph type
        mapped_multi_modal_ods, mapped_single_modal_ods = (
            self._filter_mapped_ods(self.graph_types[graph_type], od_graph_mapped_nodes, True)
        )

        # Store the results based on the specified graph type
        setattr(self, f'{graph_type}_mapped_multi_modal_ods', mapped_multi_modal_ods)
        setattr(self, f'{graph_type}_mapped_single_modal_ods', mapped_single_modal_ods)
