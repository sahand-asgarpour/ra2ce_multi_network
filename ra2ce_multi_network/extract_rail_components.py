from osm_flex.download import *
from osm_flex.extract import *
from osm_flex.config import *
import osm_flex.clip as cp

import geojson
from shapely.geometry import shape
from damagescanner.vector import *

from ra2ce_multi_network.deeper_extraction import filter_on_other_tags
from ra2ce_multi_network.simplify_rail import *

## Defining ini variables
dump_folder = Path(
    # r'C:\Users\asgarpou\OneDrive - Stichting Deltares\Documents\Projects\SITO_Netwerk to system\content\input_data\pbf'
    r'C:\Users\asgarpou\osm\osm_bpf'
)
# dump-related
country_iso3 = "NLD"
country = DICT_GEOFABRIK[country_iso3][1]
study_area_suffix = ''

# Clipping-related
clip_polygon_path = Path(
    rf'C:\Users\asgarpou\osm\osm_bpf\POLYGON{study_area_suffix}.geojson'
)
clip_output_name = f'study_area{study_area_suffix}'
study_area_dump_path = dump_folder.joinpath(f'{clip_output_name}.osm.pbf')

# Extraction-related
default_osm_keys = DICT_CIS_OSM['rail']['osm_keys']

# source: https://taginfo.openstreetmap.org/tags/railway=rail#combinations
rail_track_attributes = {
    'osm_keys': [
        'railway', 'name', 'gauge', 'electrified', 'voltage', 'bridge', 'maxspeed', 'service', 'tunnel', 'other_tags'
    ],
    'other_tags': ['"railway:traffic_mode"=>', '"usage"=>']
}

station_attributes = {
    'osm_keys': ['railway', 'name', 'other_tags'],
    'other_tags': ['"train"=>']
}
rail_track_osm_query = """railway='rail'"""
station_osm_query = """railway='station'"""

## Get the country or region dump
# get_country_geofabrik(iso3=country_iso3, save_path=dump_folder)

## Clip to the study area
with open(clip_polygon_path) as f:
    clip_polygon = geojson.load(f)

polygon_feature = clip_polygon['features'][0]
polygon_geom = shape(polygon_feature['geometry'])
# cp.clip_from_shapes([polygon_geom],
#                     osmpbf_output=OSM_DATA_DIR.joinpath(f'{clip_output_name}.osm.pbf'),
#                     osmpbf_clip_from=OSM_DATA_DIR.joinpath(f'{country}-latest.osm.pbf'),
#                     kernel='osmconvert', overwrite=True)

## Extract required system components

raw_rail_track_gdf = extract(osm_path=study_area_dump_path, geo_type='lines',
                             osm_keys=rail_track_attributes['osm_keys'], osm_query=rail_track_osm_query)

rail_track_gdf = filter_on_other_tags(
    attributes=rail_track_attributes, other_tags_keys=rail_track_attributes['other_tags'], gdf=raw_rail_track_gdf
)

# raw_station_gdf = extract(osm_path=study_area_dump_path, geo_type='points',
#                           osm_keys=station_attributes['osm_keys'], osm_query=station_osm_query)
#
# station_gdf = filter_on_other_tags(
#     attributes=station_attributes, other_tags_keys=station_attributes['other_tags'], gdf=raw_station_gdf
# )

## Alternative extraction package and method: Use damagescanner to retrieve railway
# all_rail = retrieve(osm_path=str(study_area_dump_path), geo_type='lines',
#                     key_col=rail_osm_keys, **{"service": [" IS NOT NULL"]})
# rail_gdf = all_rail[all_rail['railway'] == 'rail']

## Find possible terminals
# rail_network = detect_possible_terminals(network_gdf=rail_track_gdf)
a = 1

# ToDO: See why some links miss in the rail_track extraction.
# ToDO: Use osmnx and add polygone to extract network which is complete.