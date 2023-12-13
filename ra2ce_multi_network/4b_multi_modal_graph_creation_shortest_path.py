from pathlib import Path
import pickle
from pyproj import CRS

from ra2ce_multi_network.MultiModalGraph import MultiModalGraph
import geopandas as gpd
from ra2ce_multi_network.simplify_rail import _nx_to_network, _network_to_nx

root_folder = Path(
    r'C:\Users\asgarpou\OneDrive - Stichting Deltares\Documents\Projects\Long involvement\SITO_Netwerk to system\project_files\multi_network')
od_file_path = root_folder.joinpath('static/network/od_nodes.geojson')

# od_table is made and saved based on the ra2ce expected attributes (names and structure)
od_gdf = gpd.read_file(od_file_path)
od_gdf['category'] = 'terminal'
od_gdf['flow'] = [i * 10 for i in range(1, len(od_gdf) + 1)]  # optional at this stage

od_gdf.to_feather(root_folder.joinpath(f'static/output_graph/origin_destination_table.feather'))

# load merged and simplified rail and road graphs
with open(root_folder.joinpath(f'static/output_graph/road_simple_graph_study_area_GR.p'), 'rb') as road_f:
    road_graph = pickle.load(road_f)

with open(root_folder.joinpath(f'static/output_graph/merged_rail_network_study_area_GR.p'),
          'rb') as rail_f:
    rail_graph = pickle.load(rail_f)

# Define graph types (origin destination mapped (necessary), and simplified (optional)).
graph_types = {'road': road_graph, 'rail': rail_graph}
# define the setting of the analysis
optimal_route_setting = {
    "name": "optimal_route_od",
    "analysis": "optimal_route_origin_destination",
    "weighing": "time",
    "save_gpkg": True,
    "save_csv": True
}
# create project_input
project_input = {
    "project_name": "multi_network_ra2ce",
    "analysis_settings": [optimal_route_setting]
}
#  ToDo: write the comment about different types and why => also other important stuff
# for approach b: given od
# Each graph can be directed or indirect ...

# graph_types = {
#     'rail': nx.MultiGraph([('A', 'B'), ('B', 'C')]),
#     'road': nx.DiGraph([('C', 'D'), ('D', 'F')]),
#     'ww': nx.MultiDiGraph([('F', 'G'), ('G', 'H')]),
# }

# Create an instance of MultiModalGraph and run the optimal route analysis
multi_modal_object = MultiModalGraph(od_file_path, graph_types, CRS.from_epsg(4326), map_od=True,
                                     graphs_to_add_attributes=["rail", "road"])
multi_modal_graph = multi_modal_object.create_multi_modal_graph()
config = multi_modal_object.run_analysis(
    modes=['rail', 'road', 'multi_modal'], project_input=project_input, analysis_path=root_folder)
