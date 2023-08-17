import pickle

from osm_flex.config import *
from ra2ce_multi_network.simplify_rail import _merge_edges

## Defining ini variables
root_folder = OSM_DATA_DIR.parent
study_area_suffix = '_test'
clip_output_name = f'study_area{study_area_suffix}'
complex_network_file = root_folder.joinpath(f'networks/complex_rail_network_{clip_output_name}.geojson')

## Load complex and merged networks
with open(complex_network_file, 'rb') as handle:
    complex_rail_network = pickle.load(handle)

## Merge tracks
merged_rail_network = _merge_edges(network=complex_rail_network, excluded_edge_types=['bridge', 'tunnel'])

with open(root_folder.joinpath(f'networks/merged_rail_network_{clip_output_name}.geojson'), 'wb') as handle:
    pickle.dump(complex_rail_network, handle, protocol=pickle.HIGHEST_PROTOCOL)