import pickle

from osm_flex.config import *
from ra2ce_multi_network.simplify_rail import *
from ra2ce_multi_network.simplify_rail import _make_network_from_gdf, _drop_hanging_nodes, _network_to_nx, \
    _nx_to_network, _reset_indices

### Defining ini variables
root_folder = Path(
    r'C:\Users\asgarpou\OneDrive - Stichting Deltares\Documents\Projects\Long involvement\SITO_Netwerk to system\project_files\get_shortest_routes')
output_folder = root_folder.joinpath('static/output_graph')
study_area_suffix = '_GR'  # small case study area that works: '_ROTTERDAM_PORT'
clip_output_name = f'study_area{study_area_suffix}'

rail_track_file = root_folder.joinpath(f'static/network/rail_track_{clip_output_name}.geojson')
rail_track_gdf = gpd.read_file(rail_track_file)

od_file = root_folder.joinpath('static/network/od_nodes.geojson')
od_gdf = gpd.read_file(od_file)

# # Create a network and save it
# rail_network = _make_network_from_gdf(network_gdf=rail_track_gdf)
# rail_network.edges.to_file(output_folder.joinpath(f'rail_edges_{clip_output_name}.geojson'), driver="GeoJSON")
# rail_network.nodes.to_file(output_folder.joinpath(f'rail_nodes_{clip_output_name}.geojson'), driver="GeoJSON")

# # _drop_hanging_nodes of a rail network
# rail_network = Network(nodes=gpd.read_file(output_folder.joinpath(f'rail_nodes_{clip_output_name}.geojson')),
#                        edges=gpd.read_file(output_folder.joinpath(f'rail_edges_{clip_output_name}.geojson')))

# rail_network.set_crs(crs="EPSG:4326")
# rail_network = _drop_hanging_nodes(rail_network)
# rail_network.nodes.to_file(output_folder.joinpath(f'rail_nodes_dropped_hanging_{clip_output_name}.geojson'),
#                            driver="GeoJSON")
# rail_network.edges.to_file(output_folder.joinpath(f'rail_edges_dropped_hanging_{clip_output_name}.geojson'),
#                            driver="GeoJSON")

# map od nodes to the rail network
rail_network = Network(
    nodes=gpd.read_file(output_folder.joinpath(f'rail_nodes_dropped_hanging_{clip_output_name}.geojson')),
    edges=gpd.read_file(output_folder.joinpath(f'rail_edges_dropped_hanging_{clip_output_name}.geojson'))
)

graph = _network_to_nx(rail_network)
od_gdf, graph = add_od_nodes(od=gpd.read_file(od_file), graph=graph, crs=pyproj.CRS("EPSG:4326"))

od_gdf.to_file(od_file, driver="GeoJSON")
with open(output_folder.joinpath(f'rail_complex_graph_origin_destinations_mapped_{clip_output_name}.pkl'), 'wb') as f:
    pickle.dump(graph, f)

rail_network = _nx_to_network(graph)

rail_network.nodes.to_file(output_folder.joinpath(
    f'rail_nodes_complex_graph_origin_destinations_mapped_{clip_output_name}.geojson'), driver="GeoJSON")
rail_network.edges.to_file(output_folder.joinpath(
    f'rail_edges_complex_graph_origin_destinations_mapped_{clip_output_name}.geojson'), driver="GeoJSON")

a = 1
