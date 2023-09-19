import osm_flex.download as dl
import osm_flex.config
import osm_flex.clip as cp
from shapely import unary_union

# Downloading a larger "parent" file to clip from: central america, from geofabrik.de
# --> saved to OSM_DATA_DIR/central-america-latest.osm.pbf
dl.get_region_geofabrik('central-america')

# Obtain a country polygon (honduras) and clip the central america file to it:
admin_1_hnd = cp.get_admin1_shapes('HND')
admin_1_hnd = unary_union([poly for poly in list(admin_1_hnd.values())]) # make a single valid polygon!

# CLIP
cp.clip_from_shapes([admin_1_hnd],
                    osmpbf_output=osm_flex.config.OSM_DATA_DIR.joinpath('hnd_clipped.osm.pbf'),
                    osmpbf_clip_from=osm_flex.config.OSM_DATA_DIR.joinpath('central-america-latest.osm.pbf'),
                    kernel='osmconvert', overwrite=True)
a = 1