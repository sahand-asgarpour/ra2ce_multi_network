import geopandas as gpd
import pickle

import pyproj
from networkx import MultiDiGraph
from pathlib import Path
from snkit import Network

from ra2ce.graph.exporters.geodataframe_network_exporter import GeoDataFrameNetworkExporter
from ra2ce.graph.exporters.multi_graph_network_exporter import MultiGraphNetworkExporter
from ra2ce.graph.exporters.network_exporter_base import NetworkExporterBase
from ra2ce.graph.network_config_data.network_config_data import NetworkConfigData, NetworkSection
from ra2ce.graph.network_wrappers.osm_network_wrapper.osm_network_wrapper import OsmNetworkWrapper
from ra2ce.graph.origins_destinations import add_od_nodes
from ra2ce_multi_network.simplify_rail import _nx_to_network, _network_to_nx
import ra2ce.graph.networks_utils as nut

# Defining ini variables
root_folder = Path(
    r'C:\Users\asgarpou\OneDrive - Stichting Deltares\Documents\Projects\Long involvement\SITO_Netwerk to system\project_files\get_shortest_routes_road')
study_area_suffix = '_GR'
output_name = f'study_area{study_area_suffix}'
polygon_path = root_folder.joinpath(f'static/network/polygon{study_area_suffix}.geojson')
od_file = root_folder.joinpath('static/network/od_nodes.geojson')

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

# Download the graph from osm and save
osm_network_wrapper = OsmNetworkWrapper(config_data=config_data)
# complex_graph = osm_network_wrapper.get_clean_graph_from_osm()
# complex_graph.graph["crs"] = pyproj.CRS("EPSG:4326")
# with open(root_folder.joinpath(f'static/output_graph/road_complex_graph_{output_name}.p'), 'wb') as f:
#     pickle.dump(complex_graph, f)


# Map origin-destinations and save
with open(root_folder.joinpath(f'static/output_graph/road_complex_graph_origin_destinations_mapped_{output_name}.p'), 'rb') as f:
    complex_graph = pickle.load(f)

if not isinstance(complex_graph, MultiDiGraph):
    complex_graph = MultiDiGraph(complex_graph)

# # Simplify complex graph => get_network procedure
# simple_graph, complex_graph, link_tables = nut.create_simplified_graph(complex_graph)
#
# # Save the link tables linking complex and simple IDs
# osm_network_wrapper._export_linking_tables(link_tables)
#
# if not osm_network_wrapper.is_directed and isinstance(simple_graph, MultiDiGraph):
#     simple_graph = simple_graph.to_undirected()
#
# # Check if all geometries between nodes are there, if not, add them as a straight line.
# simple_graph = nut.add_missing_geoms_graph(simple_graph, geom_name="geometry")
# simple_graph.graph["crs"] = pyproj.CRS("EPSG:4326")
## simple_graph = osm_network_wrapper._get_avg_speed(simple_graph)

# # Map Origin destinations to the complex and simple graphs
# od_gdf = gpd.read_file(od_file)
# # od_gdf, complex_graph = add_od_nodes(od=gpd.read_file(od_file), graph=complex_graph, crs=pyproj.CRS("EPSG:4326"))
# simple_graph = add_od_nodes(od=gpd.read_file(od_file), graph=simple_graph, crs=pyproj.CRS("EPSG:4326"))[1]

# od_gdf.to_file(od_file, driver="GeoJSON")
#
# with open(root_folder.joinpath(
#         f'static/output_graph/road_complex_graph_origin_destinations_mapped_{output_name}.p'), 'wb') as f:
#     pickle.dump(complex_graph, f)
#
# with open(root_folder.joinpath(
#         f'static/output_graph/road_simple_graph_origin_destinations_mapped_{output_name}.p'), 'wb') as f:
#     pickle.dump(simple_graph, f)

# Save the complex graph as snkit.Network and geojson
# road_network_complex = _nx_to_network(complex_graph)
#
# road_network_complex.nodes.to_file(root_folder.joinpath(
#     f'static/output_graph/road_nodes_complex_graph_origin_destinations_mapped_{output_name}.geojson'), driver="GeoJSON")
# road_network_complex.edges.to_file(root_folder.joinpath(
#     f'static/output_graph/road_edges_complex_graph_origin_destinations_mapped_{output_name}.geojson'), driver="GeoJSON")

with open(root_folder.joinpath(f'static/output_graph/road_simple_graph_origin_destinations_mapped_{output_name}.p'), 'rb') as f:
    simple_graph = pickle.load(f)
simple_graph.graph["crs"] = pyproj.CRS("EPSG:4326")

road_network_simple = _nx_to_network(simple_graph)

simple_graph_exporter = MultiGraphNetworkExporter(
    basename=f'road_simple_graph_origin_destinations_mapped_{output_name}',
    export_types=['pickle', 'shp'])
simple_graph_exporter.export_to_shp(output_dir=root_folder.joinpath(f'static/output_graph'),
                                    export_data=simple_graph)
simple_graph_exporter.export_to_pickle(output_dir=root_folder.joinpath(f'static/output_graph'),
                                       export_data=simple_graph)

# with open(root_folder.joinpath(f'static/output_graph/road_complex_graph_{output_name}.p'), 'rb') as f:
#     complex_graph = pickle.load(f)

complex_graph.graph["crs"] = pyproj.CRS("EPSG:4326")

complex_graph_exporter = MultiGraphNetworkExporter(
    basename=f'road_complex_graph_origin_destinations_mapped_{output_name}',
    export_types=['pickle', 'shp'])
complex_graph_exporter.export_to_shp(output_dir=root_folder.joinpath(f'static/output_graph'),
                                     export_data=complex_graph)
complex_graph_exporter.export_to_pickle(output_dir=root_folder.joinpath(f'static/output_graph'),
                                        export_data=complex_graph)
a = 1