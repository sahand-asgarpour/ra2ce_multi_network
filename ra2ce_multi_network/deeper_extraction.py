import geopandas as gpd


def filter_on_other_tags(attributes: dict, other_tags_keys: list, gdf: gpd.GeoDataFrame, dropna=None):
    if dropna is None:
        dropna = []
    if not isinstance(dropna, list):
        raise ValueError("""drop na should be the list of column names for which each row should be dropped 
        when the column's value is null""")
    if gdf.empty:
        return gdf
    if 'other_tags' in attributes:
        for flt in other_tags_keys:
            flt_attribute_name = flt.split('=>')[0]
            gdf[flt_attribute_name] = None
            gdf[flt_attribute_name] = gdf[['other_tags', flt_attribute_name]].apply(
                lambda x: _filter(_flt=flt, _attr=x['other_tags']), axis=1)
            gdf.rename(columns={flt_attribute_name: flt_attribute_name.strip('"')}, inplace=True)
        if bool(dropna):
            for column_name in dropna:
                gdf.dropna(subset=[column_name], inplace=True)
        return gdf
    else:
        return gdf


def _filter(_flt: str, _attr: str):
    if _attr is not None:
        if _flt in _attr:
            if _flt.split('=>')[1] == "":
                return next((tag_kay.split('=>')[1] for tag_kay in _attr.split(",") if _flt in tag_kay), None)
            else:
                return _flt.split('=>')[1]
