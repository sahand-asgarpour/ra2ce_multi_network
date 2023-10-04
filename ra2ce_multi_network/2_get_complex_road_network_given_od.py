import pickle

import pyproj
from pathlib import Path

from ra2ce.graph.network_config_data.network_config_data import NetworkConfigData, NetworkSection
from ra2ce.graph.network_wrappers.osm_network_wrapper.osm_network_wrapper import OsmNetworkWrapper
from ra2ce_multi_network.simplify_rail import _nx_to_network

# Defining ini variables
root_folder = Path(
    r'C:\Users\asgarpou\OneDrive - Stichting Deltares\Documents\Projects\Long involvement\SITO_Netwerk to system\project_files\get_shortest_routes_road')
study_area_suffix = '_GR'
output_name = f'study_area{study_area_suffix}'
polygon_path = root_folder.joinpath(f'static/network/polygon{study_area_suffix}.geojson')

# Setting up the config data
config_data = NetworkConfigData(
    static_path=root_folder.joinpath('static'),
    crs=pyproj.CRS("EPSG:4326")
)

config_data.network = NetworkSection(
    network_type='drive',
    road_types=['motorway', 'motorway_link', 'primary', 'primary_link', 'secondary', 'secondary_link', 'tertiary',
                'tertiary_link', 'residential'],
    polygon=polygon_path
)

# Download the graphs
osm_network_wrapper = OsmNetworkWrapper(config_data=config_data)
complex_graph = osm_network_wrapper.get_clean_graph_from_osm()

with open(root_folder.joinpath(f'static/output_graph/road_complex_graph_{output_name}.pkl'), 'wb') as f:
    pickle.dump(complex_graph, f)

with open(root_folder.joinpath(f'static/output_graph/road_complex_graph_{output_name}.pkl'), 'rb') as f:
    complex_graph = pickle.load(f)

road_network = _nx_to_network(complex_graph)

road_network.nodes.to_file(root_folder.joinpath(
    f'static/output_graph/road_nodes_complex_{output_name}.geojson'), driver="GeoJSON")
road_network.edges.to_file(root_folder.joinpath(
    f'static/output_graph/road_edges_complex__{output_name}.geojson'), driver="GeoJSON")
a = 1
