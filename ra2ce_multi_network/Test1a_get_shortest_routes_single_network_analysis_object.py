# ToDo: visualise the results
import numpy as np
import pandas as pd
import pickle
from osm_flex.config import *
import networkx as nx

from ra2ce.analyses.analysis_config_data.analysis_config_data_reader import AnalysisConfigDataReader
from ra2ce.analyses.analysis_config_data.enums.analysis_direct_enum import AnalysisDirectEnum
from ra2ce.analyses.analysis_config_data.enums.analysis_indirect_enum import AnalysisIndirectEnum
from ra2ce.analyses.analysis_config_data.enums.weighing_enum import WeighingEnum
from ra2ce.analyses.indirect.analyses_indirect import IndirectAnalyses
from ra2ce.graph.graph_files.graph_files_collection import GraphFilesCollection
from ra2ce.graph.network_config_data.network_config_data import NetworkConfigData
from ra2ce_multi_network.simplify_rail import _network_to_nx

from ra2ce.analyses.analysis_config_data.analysis_config_data import (
    AnalysisConfigData,
    AnalysisSectionBase,
    AnalysisSectionDirect,
    AnalysisSectionIndirect,
    DirectAnalysisNameList,
    IndirectAnalysisNameList,
)

# Defining ini variables
root_folder = Path(
    # r'C:\Users\asgarpou\OneDrive - Stichting Deltares\Documents\Projects\SITO_Netwerk to system\content\input_data\pbf'
    r'C:\Users\asgarpou\osm'
)
## Defining ini variables
study_area_suffix = '_GR'  # small case study area that works: '_ROTTERDAM_PORT'
clip_output_name = f'study_area{study_area_suffix}'
rail_net_file = root_folder.joinpath(f'networks/merged_rail_network_{clip_output_name}.geojson')
analysis_folder = root_folder.joinpath('analysis/Test1a_optimal_rotes_rail')

# Load rail_net
with open(rail_net_file, 'rb') as handle:
    rail_net = pickle.load(handle)

rail_net.edges = rail_net.edges.to_crs(4326)
rail_net.nodes = rail_net.nodes.to_crs(4326)

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

od_gdf.to_feather(analysis_folder.joinpath(f'static/output_graph/origin_destination_table.feather'))

# convert rail network to a NetworkX graph
graph = _network_to_nx(rail_net)


def get_analysis_sections(project: dict) -> list[AnalysisSectionBase]:
    """
    based on get_project_section in C:\repos\ra2ce\ra2ce\analyses\analysis_config_data
    """
    _analysis_sections = []

    for analysis_setting in project["analysis_settings"]:
        _analysis_type = analysis_setting["analysis"]
        if _analysis_type in DirectAnalysisNameList:
            analysis_setting["analysis"] = AnalysisDirectEnum.get_enum(_analysis_type)
            _analysis_section = AnalysisSectionDirect(**analysis_setting)
        elif _analysis_type in IndirectAnalysisNameList:
            analysis_setting["analysis"] = AnalysisIndirectEnum.get_enum(_analysis_type)
            if analysis_setting["weighing"] == "distance":
                analysis_setting["weighing"] = WeighingEnum.LENGTH
            else:
                analysis_setting["weighing"] = WeighingEnum.get_enum(analysis_setting["weighing"])
            _analysis_section = AnalysisSectionIndirect(**analysis_setting)
        else:
            raise ValueError(f"Analysis {_analysis_type} not supported.")
        _analysis_sections.append(_analysis_section)

    return _analysis_sections


_analyses_config = AnalysisConfigData(
    root_path=analysis_folder,
    input_path=analysis_folder.joinpath(Path("input")),
    static_path=analysis_folder.joinpath(Path("static")),
    output_path=analysis_folder.joinpath(Path("output"))
)
_optimal_route_setting = {
    "name": "optimal_route_od",
    "analysis": "optimal_route_origin_destination",
    "weighing": "time",
    "save_gpkg": True,
    "save_csv": True
}
project = {
    "project_name": "multi_network_ra2ce",
    "analysis_settings": [_optimal_route_setting],
    "origins_destinations_mapped_graph": graph
}
_analyses_config.analyses = get_analysis_sections(project)

_analyses_config.project.name = project["project_name"]

_analyses = IndirectAnalyses(config=_analyses_config, graph_files=GraphFilesCollection())
_analyses.graph_files.origins_destinations_graph = graph

_analyses = IndirectAnalyses(config=_analyses_config, graph_files=_analyses.graph_files)

for _analysis_config in _analyses_config.analyses:
    func = getattr(_analyses, _analysis_config.analysis.config_value)
    result = func(_analyses.graph_files.origins_destinations_graph, _analysis_config)
    result = result[~result['geometry'].is_empty]

    # Save
    result.to_pickle(root_folder / 'analysis' / 'Test1a_optimal_rotes_rail' / 'output' /
                     f'{_analysis_config.analysis.config_value}_{study_area_suffix}.pkl')
    result.to_csv(path_or_buf=root_folder / 'analysis' / 'Test1a_optimal_rotes_rail' / 'output' /
                              f'{_analysis_config.analysis.config_value}_{study_area_suffix}.csv',
                  index=False)
