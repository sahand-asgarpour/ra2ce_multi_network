import pickle

from osm_flex.config import *
from ra2ce_multi_network.simplify_rail import *

### Defining ini variables
root_folder = OSM_DATA_DIR.parent
study_area_suffix = '_ROTTERDAM_PORT'
clip_output_name = f'study_area{study_area_suffix}'


rail_track_file = root_folder.joinpath(f'extracts/rail_track_{clip_output_name}.geojson')
rail_track_gdf = gpd.read_file(rail_track_file)

complex_rail_network = get_rail_network_with_terminals(network_gdf=rail_track_gdf, aggregation_range=0.01)

with open(root_folder.joinpath(f'networks/complex_rail_network_{clip_output_name}.geojson'), 'wb') as handle:
    pickle.dump(complex_rail_network, handle, protocol=pickle.HIGHEST_PROTOCOL)

a = 1