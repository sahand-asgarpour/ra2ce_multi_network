import osm_flex.download as dl
from osm_flex.extract import *
from osm_flex.config import *
import osm_flex.clip as cp

import geojson

from ra2ce_multi_network.deeper_extraction import filter_on_other_tags
from ra2ce_multi_network.simplify_rail import *

### Defining ini variables
root_folder = OSM_DATA_DIR.parent

## dump-related
iso3_code = "DEU"
region_code = "Europe"

# dump_region = DICT_GEOFABRIK[iso3_code][1]
dump_region = region_code.lower()

dump_folder = root_folder / "osm_bpf"

## Clipping-related
study_area_suffix = '_GR'
clip_polygon_path = Path(
    rf'C:\Users\asgarpou\osm\osm_bpf\polygon{study_area_suffix}.geojson'
)
clip_output_name = f'study_area{study_area_suffix}'
study_area_dump_path = root_folder.joinpath('osm_bpf', f'{clip_output_name}.osm.pbf')

## Save gdfs
raw_rail_track_file = root_folder.joinpath(f'extracts/raw_rail_track_{clip_output_name}.geojson')
rail_track_file = root_folder.joinpath(f'extracts/rail_track_{clip_output_name}.geojson')

## Extraction-related
default_osm_keys = DICT_CIS_OSM['rail']['osm_keys']
extract_path = root_folder.joinpath('extracts')

# source: https://taginfo.openstreetmap.org/tags/railway=rail#combinations
# 'other_tags' key is a string chain of 'tags' => 'keys',
# where relevant information is stored. e.g., whether traffic mode is freight or mixed
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
rail_track_osm_query = """railway='rail'"""  # or railway='light_rail'
station_osm_query = """railway='station'"""

### Run the functions

## Get the country or region dump
# dl.get_country_geofabrik(iso3=iso3_code, save_path=dump_folder)
# dl.get_region_geofabrik(region=region_code.lower(), save_path=dump_folder)

## Clip to the study area
with open(clip_polygon_path) as f:
    clip_polygon = geojson.load(f)

# Check the osmpbf_clip_from file size (due to osmconvert limitation)
osmpbf_clip_from = OSM_DATA_DIR.joinpath(f'{dump_region}-latest.osm.pbf')
size_threshold = 2 * 1024 ** 3

file_size_bytes = os.path.getsize(osmpbf_clip_from)

if file_size_bytes > size_threshold:
    size_threshold_passed = True
else:
    size_threshold_passed = False

polygon_feature = clip_polygon['features'][0]
polygon_geom = shape(polygon_feature['geometry'])
cp.clip_from_shapes([polygon_geom],
                    osmpbf_output=OSM_DATA_DIR.joinpath(f'{clip_output_name}.osm.pbf'),
                    osmpbf_clip_from=OSM_DATA_DIR.joinpath(f'{dump_region}-latest.osm.pbf'),
                    size_threshold_passed=size_threshold_passed, kernel='osmconvert', overwrite=True)

## Extract required system components
raw_rail_track_gdf = extract(osm_path=study_area_dump_path, geo_type='lines',
                             osm_keys=rail_track_attributes['osm_keys'], osm_query=rail_track_osm_query)

raw_rail_track_gdf.to_file(raw_rail_track_file, driver='GeoJSON')

rail_track_gdf = filter_on_other_tags(
    attributes=rail_track_attributes, other_tags_keys=rail_track_attributes['other_tags'], gdf=raw_rail_track_gdf)

rail_track_gdf.to_file(rail_track_file, driver='GeoJSON')
