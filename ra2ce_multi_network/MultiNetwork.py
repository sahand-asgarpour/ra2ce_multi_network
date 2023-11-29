from geopandas import GeoDataFrame
from pathlib import Path
import geopandas as gpd

from networkx import MultiGraph


# ToDo: add an attribute to od stating multi_modal_terminal
# ToDo: rerun the od mapping for both merged rail and simplified road
# ToDo: Define a function to detect the mapped multi_modal_terminals in the separate graphs and connect the closest ones

class MultiNetwork:
    def __init__(self, od_file: Path, rail_graph: MultiGraph, road_graph: MultiGraph, od_id_column: str = "id") -> None:
        self.od_gdf: GeoDataFrame = gpd.read_file(od_file)
        self.od_id_column = od_id_column
        self._od_multi_modal_terminals: dict = self._filter_ods()
        self.rail_graph: MultiGraph = rail_graph
        self.road_graph: MultiGraph = road_graph
        pass

    def _filter_ods(self, attr: str = 'multi_modal_terminal') -> dict:
        filtered_ods = {}
        for index, row in self.od_gdf[self.od_gdf[attr] == 1].iterrows():
            filtered_ods[row['id']] = (row['o_id'], row['d_id'])
        return filtered_ods

    def _find_multi_network_connections(self):
        pass
