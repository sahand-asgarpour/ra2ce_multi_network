from osm_flex.download import *
from osm_flex.extract import *
from osm_flex.config import *
import osm_flex.clip as cp

import geojson
from shapely.geometry import shape
from damagescanner.vector import *

## Defining ini variables
dump_folder = Path(
    # r'C:\Users\asgarpou\OneDrive - Stichting Deltares\Documents\Projects\SITO_Netwerk to system\content\input_data\pbf'
    r'C:\Users\asgarpou\osm\osm_bpf'
)
country_iso3 = "NLD"
country = DICT_GEOFABRIK[country_iso3][1]
study_area_suffix = ''
clip_polygon_path = Path(
    rf'C:\Users\asgarpou\osm\osm_bpf\POLYGON{study_area_suffix}.geojson'
)
clip_output_name = f'study_area{study_area_suffix}'
study_area_dump_path = dump_folder.joinpath(f'{clip_output_name}.osm.pbf')

default_osm_keys = DICT_CIS_OSM['rail']['osm_keys']

additional_rail_keys = [
    'bridge', 'maxspeed', 'service', 'tunnel'
]
additional_station_keys = [
    # 'train'
]
rail_osm_query = """railway='rail'"""
station_osm_query = """railway='station'"""
additional_osm_keys = {  # source: https://taginfo.openstreetmap.org/tags/railway=rail#combinations
    rail_osm_query: additional_rail_keys,
    station_osm_query: additional_station_keys
}
rail_osm_keys = default_osm_keys + additional_osm_keys[rail_osm_query]
station_osm_keys = [k for k in default_osm_keys if k not in ['gauge', 'electrified', 'voltage']] + \
                   additional_osm_keys[station_osm_query]
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

rail_gdf = extract(osm_path=study_area_dump_path, geo_type='lines',
                   osm_keys=rail_osm_keys, osm_query=rail_osm_query)

station_gdf = extract(osm_path=study_area_dump_path, geo_type='points',
                      osm_keys=station_osm_keys, osm_query=station_osm_query)

## Use damagescanner to retrieve railway
# all_rail = retrieve(osm_path=str(study_area_dump_path), geo_type='lines',
#                     key_col=rail_osm_keys, **{"service": [" IS NOT NULL"]})
# rail_gdf = all_rail[all_rail['railway'] == 'rail']

a = 1
