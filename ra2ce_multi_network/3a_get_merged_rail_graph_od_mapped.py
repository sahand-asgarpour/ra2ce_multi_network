import pickle
import pyproj

from osm_flex.config import *

from ra2ce.graph.origins_destinations import add_od_nodes
from simplify_rail import _merge_edges, _nx_to_network, _network_to_nx
from snkit import Network
import geopandas as gpd



## Defining ini variables
root_folder = Path(
    r'C:\Users\asgarpou\OneDrive - Stichting Deltares\Documents\Projects\Long involvement\SITO_Netwerk to system\project_files\get_shortest_routes')
study_area_suffix = '_GR'
output_name = f'study_area{study_area_suffix}'
od_file = root_folder.joinpath('static/network/od_nodes.geojson')
od_gdf = gpd.read_file(od_file)

# Load extracted network elements
with open(root_folder.joinpath(f'static/output_graph/rail_complex_graph_origin_destinations_mapped_{output_name}.pkl'), 'rb') as f:
    complex_graph = pickle.load(f)

complex_rail_network = Network(
        nodes=_nx_to_network(complex_graph).nodes,
        edges=_nx_to_network(complex_graph).edges
    )

# # Merge the railway network: merge edges
# merged_rail_network = _merge_edges(complex_rail_network, excluded_edge_types=['bridge', 'tunnel'])

# load an earlier merged network
merged_rail_network_file = root_folder.joinpath(f'static/output_graph/merged_rail_network_{output_name}.geojson')
with open(merged_rail_network_file, 'rb') as handle:
    merged_rail_network = pickle.load(handle)

# map od nodes to the rail network
complex_graph = _network_to_nx(merged_rail_network)
od_gdf, complex_graph = add_od_nodes(od=od_gdf, graph=complex_graph, crs=pyproj.CRS("EPSG:4326"))
merged_rail_network = _nx_to_network(complex_graph)
merged_rail_network.set_crs(crs="EPSG:4326")

# save the merged network
with open(root_folder.joinpath(f'static/output_graph/merged_rail_network_{output_name}.geojson'), 'wb') as handle:
    pickle.dump(merged_rail_network, handle, protocol=pickle.HIGHEST_PROTOCOL)
