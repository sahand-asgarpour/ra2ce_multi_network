import pickle

from osm_flex.config import *
from ra2ce_multi_network.simplify_rail import *

### Defining ini variables
root_folder = OSM_DATA_DIR.parent
study_area_suffix = '_test'
clip_output_name = f'study_area{study_area_suffix}'


### Simplifying functions
## Load extracted network elements
rail_track_file = root_folder.joinpath(f'extracts/rail_track_{clip_output_name}.geojson')
rail_track_gdf = gpd.read_file(rail_track_file)

## Find possible terminals
complex_rail_network = get_rail_network_with_terminals(network_gdf=rail_track_gdf, aggregation_range=0.1)

with open(root_folder.joinpath(f'networks/complex_rail_network_{clip_output_name}.geojson'), 'wb') as handle:
    pickle.dump(complex_rail_network, handle, protocol=pickle.HIGHEST_PROTOCOL)

a = 1

#  ToDo: simplify railway network, two or more parallel tracks to one (e.g., around the yard/emplacement area)
#  ToDo: Update the Jupyter notebook per new functionality.