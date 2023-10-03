import pickle

from pathlib import Path

root_folder = Path(
    r'C:\Users\asgarpou\OneDrive - Stichting Deltares\Documents\Projects\Long involvement\SITO_Netwerk to system\project_files\get_shortest_routes')
## Defining ini variables
study_area_suffix = '_GR'  # small case study area that works: '_ROTTERDAM_PORT'
clip_output_name = f'study_area{study_area_suffix}'
rail_net_file = root_folder.joinpath(f'static/network/merged_rail_network_{clip_output_name}.geojson')

# Load rail_net
with open(rail_net_file, 'rb') as handle:
    rail_net = pickle.load(handle)

rail_net.edges = rail_net.edges.to_crs(4326)
rail_net.nodes = rail_net.nodes.to_crs(4326)

rail_net.nodes['terminal_collection'] = rail_net.nodes.apply(lambda row: str(row['terminal_collection']), axis=1)

rail_net.edges.to_file(root_folder.joinpath(f'static/network/merged_rail_edges_{clip_output_name}.geojson'),
                       driver="GeoJSON")
rail_net.nodes.to_file(root_folder.joinpath(f'static/network/merged_rail_nodes_{clip_output_name}.geojson'),
                       driver="GeoJSON")
