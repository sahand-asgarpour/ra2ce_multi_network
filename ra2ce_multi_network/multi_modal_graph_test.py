# Example usage
from pathlib import Path
import pickle
from ra2ce_multi_network.MultiModalGraph import MultiModalGraph
import networkx as nx
from snkit import Network
from ra2ce_multi_network.simplify_rail import _nx_to_network, _network_to_nx

root_folder = Path(r'C:\Users\asgarpou\OneDrive - Stichting Deltares\Documents\Projects\Long involvement\SITO_Netwerk to system\project_files\multi_network')
od_file_path = root_folder.joinpath('static/network/od_nodes.geojson')

# Create a MultiGraph for testing (replace this with your actual graph creation logic)
with open(root_folder.joinpath(f'static/output_graph/road_simple_graph_od_mapped_study_area_GR.p'), 'rb') as road_f:
    road_graph = pickle.load(road_f)

with open(root_folder.joinpath(f'static/output_graph/merged_rail_network_od_mapped_study_area_GR.geojson'), 'rb') as rail_f:
    rail_network = pickle.load(rail_f)

rail_graph = _network_to_nx(rail_network)

# Define graph types
# graph_types = {'road': road_graph, 'rail': rail_graph}
graph_types = {
    'graph1': nx.MultiGraph([('A', 'B'), ('B', 'C')]),
    'graph2': nx.DiGraph([('C', 'D'), ('D', 'F')]),
    'graph3': nx.MultiDiGraph([('F', 'G'), ('G', 'H')]),
}

# Create an instance of MultiModalGraph
multi_modal_graph = MultiModalGraph(od_file_path, graph_types)
multi_modal_graph = multi_modal_graph.create_multi_modal_graph()

# Access the results for the 'road' graph type
print("Mapped Multi-Modal ODs for Road Graph:", multi_modal_graph.road_mapped_multi_modal_ods)
print("Mapped Single-Modal ODs for Road Graph:", multi_modal_graph.road_mapped_single_modal_ods)

# Test for the 'rail' graph type
multi_modal_graph._find_graph_mapped_ods('rail')

# Access the results for the 'rail' graph type
print("Mapped Multi-Modal ODs for Rail Graph:", multi_modal_graph.rail_mapped_multi_modal_ods)
print("Mapped Single-Modal ODs for Rail Graph:", multi_modal_graph.rail_mapped_single_modal_ods)
