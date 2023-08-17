import pickle
from osm_flex.config import *
from ra2ce_multi_network.simplify_rail import _simplify_tracks

## Defining ini variables
root_folder = OSM_DATA_DIR.parent
study_area_suffix = '_test'
clip_output_name = f'study_area{study_area_suffix}'
merged_network_file = root_folder.joinpath(f'networks/merged_rail_network_{clip_output_name}.geojson')

## Load complex and merged networks
with open(merged_network_file, 'rb') as handle:
    merged_rail_network = pickle.load(handle)

## Simplify tracks
simplified_rail_network = _simplify_tracks(net=merged_rail_network, buffer_distance=0.012, hole_threshold=0.012)
# ToDo: triangle_gdf are saved. Load them to save time. Visualise them and continue with the line aggregations
