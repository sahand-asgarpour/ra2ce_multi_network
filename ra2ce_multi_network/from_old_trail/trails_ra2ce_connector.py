# External imports
from pathlib import Path
import sys
import geopandas as gpd
from osgeo import gdal

gdal.SetConfigOption("OSM_CONFIG_FILE", "osmconf.ini")

# Trails import
from src.trails.extract import *
from src.trails.flow_model import load_network, load_network_ra2ce_10a
from src.trails.simplify import *


def simplified_network_to_ra2ce(osm_path, roads_to_keep):
    """
    Wrapper function to prepare simplified network from OSM PBF file to RA2CE

    Returns:

    """
    drop_hanging_nodes_run = True
    fill_attributes_run = False

    # Workflow is inspired from simplify_all.py
    df = roads_bridges(str(osm_path))  ## Kees has this locally.
    df = df.loc[df.highway.isin(roads_to_keep)].reset_index(drop=True)

    # From here it follows simplified_network
    net = Network(edges=df) ## Based on TRAILS
    net = clean_roundabouts(net)
    net = add_endpoints(net)
    net = split_edges_at_nodes(net)
    net = add_endpoints(net)
    net = add_ids(net)
    net = add_topology(net)
    if drop_hanging_nodes_run:
        net = drop_hanging_nodes(net)
    else:
        net.nodes['degree'] = calculate_degree(net)

    net = merge_edges(net, keys_not_to_split=['bridge', 'tunnel']) ## Kees has this locally.
    net.edges = drop_duplicate_geometries(net.edges, keep='first')
    net = reset_ids(net)
    net = add_distances(net)
    net = merge_multilinestrings(net)
    if fill_attributes_run:
        net = fill_attributes(net)
    net = add_travel_time(net)

    return net


def simplified_network_to_ra2ce_develop(osm_path, roads_to_keep):
    """
    Wrapper function to prepare simplified network from OSM PBF file to RA2CE

    Returns:

    """
    # drop_hanging_nodes_run = True
    # fill_attributes_run = False
    #
    # # Workflow is inspired from simplify_all.py
    # df = roads_bridges(str(osm_path))
    # df = df.loc[df.highway.isin(roads_to_keep)].reset_index(drop=True)
    #
    # # From here it follows simplified_network
    # net = Network(edges=df)
    # net = clean_roundabouts(net)
    # net = add_endpoints(net)
    # net = split_edges_at_nodes(net)
    # net = add_endpoints(net)
    # net = add_ids(net)
    # net = add_topology(net)
    # if drop_hanging_nodes_run:
    #     net = drop_hanging_nodes(net)
    # else:
    #     net.nodes['degree'] = calculate_degree(net)

    # Workaround so I don't have to load it all the time
    # edges = gpd.GeoDataFrame(net.edges)
    # edges = edges.set_crs("EPSG:4326")
    out_folder = Path(r'data/waterbom/2022_09_29_extracts')

    import pickle
    # pickle.dump(net, open(out_folder / '{}_net.p'.format('temp'), "wb"))

    # LOAD FROM HERE
    net = pickle.load(open(out_folder / '{}_net.p'.format('temp'), 'rb'))

    net = merge_edges(net, keys_not_to_split=['bridge', 'tunnel'])
    net.edges = drop_duplicate_geometries(net.edges, keep='first')
    net = reset_ids(net)
    net = add_distances(net)
    net = merge_multilinestrings(net)
    if False:
        net = fill_attributes(net)
    net = add_travel_time(net)

    return net


def simplify_categories(edges, use_col='highway', new_col='road_types'):
    """
    Simplify road categories to less categories by removing _link, and add as a new col to df

    Args:
        edges (df):
        use_col:
        new_col:

    Returns:
        edges (df): with new columns
    """
    edges[new_col] = edges[use_col]
    edges[new_col] = edges[new_col].apply(lambda x: x[:-5] if x.endswith('_link') else x)
    return edges


# THE SIMPLEST WAY TO DRAW THE RAW DATAFRAME
# df = mainRoads(str(zh_pbf))
# df = motorwayRoads(str(zh_pbf))
# gdf = gpd.GeoDataFrame(df)


# gdf['id'] = gdf.index #add an index.
# gdf = gdf.set_crs("epsg:4326")

# gdf.to_file(r'data/waterbom/zh_0_test_id.shp')
# gdf.to_feather('data/waterbom/zh_0_test.feather')
# gdf.to_pickle(r'data/waterbom/zh_0_test_id.pkl')

# gdf.to_file(r'data/waterbom/zh_0_mw_id.shp')


# NOW USE THE SIMPLIFICATION PACKAGE TO ALSO CREATE THE EDGES FILES ULTIMATELY

# network = load_network(str(zh_pbf),mainroad=True)
#
# edges = gpd.GeoDataFrame(network.edges)
# edges = edges.set_crs("EPSG:4326")
# edges.to_file(r'data/waterbom/zh_01_cleaned_edges.shp')
# edges.to_pickle(r'data/waterbom/zh_01_cleaned_edges.pkl')
# nodes = gpd.GeoDataFrame(network.nodes)
# nodes = nodes.set_crs("EPSG:4326")
# nodes.to_file(r'data/waterbom/zh_01_cleaned_nodes.shp')
# nodes.to_pickle(r'data/waterbom/zh_01_cleaned_nodes.pkl')
# #Todo: would be neather through feather, that is probably built-in somewhere
#
# ###############     REPEAT WITH EXTENDED BOUNDARIES #########################
# ###############           ONLY MAIN ROADS           #########################


if __name__ == "__main__":
    osm_pbf = Path(r'data/waterbom/zh_1.osm.pbf')
    assert osm_pbf.exists()

    _roads_to_keep = ['primary', 'primary_link', 'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 'trunk',
                      'trunk_link', 'motorway', 'motorway_link']
    # _roads_to_keep = ['motorway','motorway_link']

    network = simplified_network_to_ra2ce(osm_path=osm_pbf, roads_to_keep=_roads_to_keep)

    out_name = "zh_1_5_nobridgemergetest"

    edges = gpd.GeoDataFrame(network.edges)
    edges = simplify_categories(edges)

    edges = edges.set_crs("EPSG:4326")

    out_folder = Path(r'data/waterbom/2022_09_29_extracts')
    edges.to_file(out_folder / '{}_edges.shp'.format(out_name))
    edges.to_feather(out_folder / '{}_edges.feather'.format(out_name))
    nodes = gpd.GeoDataFrame(network.nodes)
    nodes = nodes.set_crs("EPSG:4326")
    nodes.to_file(out_folder / '{}_nodes.shp'.format(out_name))
    nodes.to_feather(out_folder / '{}_nodes.feather'.format(out_name))

###############     REPEAT WITH EXTENDED BOUNDARIES #########################
###############              ALL ROADS              #########################
# zh_pbf = Path(r'data/waterbom/zh_1.osm.pbf')
# print(zh_pbf.exists())
# network = load_network(str(zh_pbf),mainroad=False)
#
# edges = gpd.GeoDataFrame(network.edges)
# edges = edges.set_crs("EPSG:4326")
# edges.to_file(r'data/waterbom/zh_11_cleaned_edges.shp')
# edges.to_pickle(r'data/waterbom/zh_11_cleaned_edges.pkl')
# nodes = gpd.GeoDataFrame(network.nodes)
# nodes = nodes.set_crs("EPSG:4326")
# nodes.to_file(r'data/waterbom/zh_11_cleaned_nodes.shp')
# nodes.to_pickle(r'data/waterbom/zh_11_cleaned_nodes.pkl')
