# Example usage
from pathlib import Path
import pickle
from ra2ce_multi_network.MultiModalGraph import MultiModalGraph
import networkx as nx
from snkit import Network
from ra2ce_multi_network.simplify_rail import _nx_to_network, _network_to_nx

root_folder = Path(
    r'C:\Users\asgarpou\OneDrive - Stichting Deltares\Documents\Projects\Long involvement\SITO_Netwerk to system\project_files\multi_network')
od_file_path = root_folder.joinpath('static/network/od_nodes.geojson')

# Create a MultiGraph for testing (replace this with your actual graph creation logic)
with open(root_folder.joinpath(f'static/output_graph/road_simple_graph_od_mapped_study_area_GR.p'), 'rb') as road_f:
    road_graph = pickle.load(road_f)

with open(root_folder.joinpath(f'static/output_graph/merged_rail_network_od_mapped_study_area_GR.geojson'),
          'rb') as rail_f:
    rail_network = pickle.load(rail_f)

rail_graph = _network_to_nx(rail_network)

# Define graph types
graph_types = {'road': road_graph, 'rail': rail_graph}

#  ToDo: write the comment about different types and why => also other important stuff
# for approach b: given od
# Each graph can be directed or indirect ...

# graph_types = {
#     'rail': nx.MultiGraph([('A', 'B'), ('B', 'C')]),
#     'road': nx.DiGraph([('C', 'D'), ('D', 'F')]),
#     'ww': nx.MultiDiGraph([('F', 'G'), ('G', 'H')]),
# }

# Create an instance of MultiModalGraph
multi_modal_graph = MultiModalGraph(od_file_path, graph_types)
multi_modal_graph = multi_modal_graph.create_multi_modal_graph()
a = 1
