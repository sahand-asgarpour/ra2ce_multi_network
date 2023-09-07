import pickle
from osm_flex.config import *
from ra2ce_multi_network.simplify_rail import _simplify_tracks

## Defining ini variables
root_folder = OSM_DATA_DIR.parent
study_area_suffix = '_ROTTERDAM_PORT'
clip_output_name = f'study_area{study_area_suffix}'
merged_network_file = root_folder.joinpath(f'networks/merged_rail_network_{clip_output_name}.geojson')

## Load complex and merged networks
with open(merged_network_file, 'rb') as handle:
    merged_rail_network = pickle.load(handle)

## Simplify tracks
simplified_rail_network = _simplify_tracks(net=merged_rail_network, buffer_distance=0.012, hole_area_threshold=0.02)
triangles_file = root_folder.joinpath(f'networks/triangle_gdf_{clip_output_name}.geojson')
with open(triangles_file, 'rb') as handle:
    triangles_gdf = pickle.load(handle)
a = 1
# ToDo: Fix the triangulation and continue with the line aggregations
