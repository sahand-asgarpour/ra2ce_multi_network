# ToDo: visualise the results
import numpy as np
import pandas as pd
import pickle
from osm_flex.config import *
import networkx as nx

from ra2ce.analyses.analysis_config_data.readers.analysis_config_reader_factory import AnalysisConfigReaderFactory
from ra2ce.analyses.indirect.analyses_indirect import IndirectAnalyses
from ra2ce.graph.network_config_data.network_config_data import NetworkConfigData

# Defining ini variables
root_folder = Path(
    r'C:\Users\asgarpou\OneDrive - Stichting Deltares\Documents\Projects\Long involvement\SITO_Netwerk to system\project_files\get_shortest_routes')
## Defining ini variables
study_area_suffix = '_GR'  # small case study area that works: '_ROTTERDAM_PORT'
clip_output_name = f'study_area{study_area_suffix}'
rail_net_file = root_folder.joinpath(f'static/network/merged_rail_network_{clip_output_name}.geojson')

# Load rail_net
with open(rail_net_file, 'rb') as handle:
    rail_net = pickle.load(handle)

# Add Origin and Destination tags to the relevant nodes. All terminals are both origins and destinations
rail_net.nodes['o_id'] = rail_net.nodes.apply(
    lambda r: f'O_{r.id}' if r['possible_terminal'] == 1 else np.nan, axis=1)
rail_net.nodes['d_id'] = rail_net.nodes.apply(
    lambda r: f'D_{r.id}' if r['possible_terminal'] == 1 else np.nan, axis=1)
rail_net.nodes['od_id'] = rail_net.nodes.apply(
    lambda r: f'O_{r.id},D_{r.id}' if r['possible_terminal'] == 1 else str(np.nan), axis=1)

# Add length and time to the edges

rail_net.edges['length'] = rail_net.edges['geometry'].length * 111.32  # length in km
rail_net.edges['max_speed'] = pd.to_numeric(rail_net.edges['maxspeed'], errors='coerce')
rail_net.edges.fillna(rail_net.edges.max_speed.mean(), inplace=True)
rail_net.edges['time'] = rail_net.edges['length'] / rail_net.edges['max_speed']  # time iin hour
rail_net.edges['rfid'] = rail_net.edges['id']  # id of the corresponding edge in the simple graph

rail_net.edges = rail_net.edges[rail_net.edges['length'] != 0]

# od_table is made based on the ra2ce expected attributes (names and structure)
od_gdf = rail_net.nodes[rail_net.nodes.possible_terminal == 1]
od_gdf = od_gdf.drop(columns=['terminal_collection'], axis=1)
od_gdf['category'] = 'terminal'
od_gdf['flow'] = [i * 10 for i in range(1, len(od_gdf) + 1)]
od_gdf = od_gdf.rename(columns={"id": "OBJECTID"})

od_gdf.to_feather(root_folder.joinpath(f'static/output_graph/origin_destination_table.feather'))

# convert rail network to a NetworkX graph
graph = nx.MultiGraph()

for index, row in rail_net.nodes.iterrows():
    node_id = row['id']
    attributes = {k: v for k, v in row.items()}
    graph.add_node(node_id, **attributes)

for index, row in rail_net.edges.iterrows():
    u = row['from_id']
    v = row['to_id']
    # This should be AtlasView or something like that
    attributes = {k: v for k, v in row.items()}
    graph.add_edge(u, v, **attributes)

# Setting up the traffic analysis based on ra2ce

network_config = NetworkConfigData()
network_config.network.primary_file = graph
network_config.output_path = root_folder.joinpath(f'output')
network_config.static_path = root_folder.joinpath(f'static')
network_config.project.name = 'rail'
network_config_dict = vars(network_config)

analysis_ini = root_folder.joinpath(r'analysis.ini')

analysis_config = AnalysisConfigReaderFactory().read(ini_file=analysis_ini, network_config=network_config_dict)
analysis = IndirectAnalyses(config=analysis_config, graphs=graph)

# Run
optimal_routes_gdf = analysis.optimal_route_origin_destination(graph, analysis.config['indirect'][0])
optimal_routes_gdf = optimal_routes_gdf[~optimal_routes_gdf['geometry'].is_empty]

# Save
optimal_routes_gdf.to_pickle(root_folder / 'output' / f'optimal_routes{study_area_suffix}.pkl')
optimal_routes_gdf.to_csv(path_or_buf=root_folder / 'output' / f'optimal_routes{study_area_suffix}.csv', index=False)
a = 1
