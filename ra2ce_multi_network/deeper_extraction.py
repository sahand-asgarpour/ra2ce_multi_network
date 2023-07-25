import geopandas as gpd


def filter_on_other_tags(attributes: dict, other_tags_keys: list, gdf: gpd.GeoDataFrame):
    if 'other_tags' in attributes:
        for flt in other_tags_keys:
            flt_attribute_name = flt.split('=>')[0]
            gdf[flt_attribute_name] = None
            gdf[flt_attribute_name] = gdf[['other_tags', flt_attribute_name]].apply(
                lambda x: _filter(_flt=flt, _attr=x['other_tags']), axis=1)
        return gdf
    else:
        return gdf


def _filter(_flt: str, _attr: str):
    if _flt in _attr:
        if _flt.split('=>')[1] == "":
            return next((tag_kay.split('=>')[1] for tag_kay in _attr.split(",") if _flt in tag_kay), None)
        else:
            return _flt.split('=>')[1]