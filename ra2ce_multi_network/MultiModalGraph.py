# ToDo: Define a function to detect the mapped multi_modal_ods in the separate graphs and connect the closest ones
# ToDo: adjust the route finding algorithm
import pandas as pd
from pathlib import Path
import geopandas as gpd
from networkx.classes.multigraph import MultiGraph
from geopandas.geodataframe import GeoDataFrame


class MultiModalGraph:
    def __init__(
            self,
            od_file: Path,
            graph_types: dict[str, MultiGraph],
    ) -> None:
        self.graph_types: dict[str, MultiGraph] = graph_types
        self.od_gdf: GeoDataFrame = gpd.read_file(od_file)
        self.od_multi_modal_ods: set = self._filter_ods()

        # Dynamically create attributes for mapped ods based on graph types
        for graph_type in graph_types:
            self._find_graph_mapped_ods(graph_type)

    def _filter_ods(self, attr: str = 'multi_modal_terminal', o_id_column: str = 'o_id', d_id_column: str = 'd_id') -> set:
        _filtered_ods: set = set()
        for _, row in self.od_gdf[self.od_gdf[attr] == 1].iterrows():
            _filtered_ods.add(row[o_id_column])
            _filtered_ods.add(row[d_id_column])
        return _filtered_ods

    def _filter_mapped_ods(
            self,
            graph: MultiGraph,
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
