import os
import numpy as np
from shapely import wkt, Point, Polygon
import pandas as pd
import geopandas as gpd

from s2geometry import S2LatLng, S2Loop, S2Polygon
import s2geometry as s2
import s2sphere

from utils.utils import load_data


# project locations into s2 cells
def get_loc_id(row, level=10):
    cell = s2sphere.Cell.from_lat_lng(s2sphere.LatLng.from_degrees(row.y, row.x))
    return cell.id().parent(level).id()


def s2_to_geo_boundary(cell_id):
    cell = s2sphere.Cell(cell_id)
    boundary = []
    for k in range(4):
        ll = s2sphere.LatLng.from_point(cell.get_vertex(k))
        boundary.append(Point(ll.lng().degrees, ll.lat().degrees))
    return boundary


def get_swiss_s2():
    # read and simplify swiss boundary -> we only need a rough estimate of coverage
    swissBoundary = gpd.read_file(os.path.join(".", "data", "swiss", "swiss_1903+.shp")).to_crs("EPSG:4326")

    # get the geometry
    swiss_polygon = swissBoundary.geometry.iloc[0]

    # simplify
    swiss_polygon_simplify = swiss_polygon.simplify(0.01, preserve_topology=False)

    # coords sequence
    coords = swiss_polygon_simplify.exterior.coords

    # construct the S2Polygon object (for Swiss)
    point_ls = []

    for xx, yy, _ in coords:
        point_ls.append(S2LatLng.FromDegrees(yy, xx).ToPoint())

    loop = S2Loop(point_ls)
    if ~loop.IsNormalized():
        loop.Normalize()
    swiss_region = S2Polygon(loop)

    return swiss_region


def get_single_layer_covering_s2(swiss_region_s2, min_level=10, max_level=14):
    single_layer_covering_s2 = {}
    # get level from 10 to 14
    for level in range(min_level, max_level + 1):
        coverer = s2.S2RegionCoverer()
        coverer.set_min_level(level)
        coverer.set_max_level(level)

        single_layer_covering_s2[level] = coverer.GetCovering(swiss_region_s2)
    return single_layer_covering_s2


def get_loc_s2_id(locs, min_level=10, max_level=14):
    processing_locs = locs.copy()
    valid_locs_ls = []
    # loop until the last level
    for level in range(min_level, max_level):
        processing_locs["s2_id"] = processing_locs["center"].apply(get_loc_id, level=level)
        processing_locs["level"] = level
        s2_grid_occurance = processing_locs.groupby("s2_id").size()

        unique_grids = s2_grid_occurance[s2_grid_occurance == 1]
        duplicate_grids = s2_grid_occurance[s2_grid_occurance > 1]

        valid_locs_ls.append(processing_locs.loc[processing_locs["s2_id"].isin(unique_grids.index.values)])
        processing_locs = processing_locs.loc[processing_locs["s2_id"].isin(duplicate_grids.index.values)].copy()

    # the rest projected to the finest level
    processing_locs["s2_id"] = processing_locs["center"].apply(get_loc_id, level=max_level)
    processing_locs["level"] = max_level
    valid_locs_ls.append(processing_locs)

    # final locations
    projected_locs = pd.concat(valid_locs_ls).sort_values(by="id")

    print(projected_locs.sort_values(by="id")["level"].value_counts())
    # level
    # 14    157665
    # 13      3486
    # 12       980
    # 11       147
    # 10        23
    # Name: count, dtype: int64
    return projected_locs


def get_s2_loc_all_multi(single_layer_covering_s2, locs, min_level=10, max_level=14):
    locs = locs.copy()
    # get the all location gdf at the coarsest level (min_level)
    row_ls = []
    for i in range(len(single_layer_covering_s2[min_level])):
        row = {}
        row["loc_id"] = single_layer_covering_s2[min_level][i].id()
        row["level"] = min_level
        row["geometry"] = Point(
            single_layer_covering_s2[min_level][i].ToLatLng().lng().degrees(),
            single_layer_covering_s2[min_level][i].ToLatLng().lat().degrees(),
        )
        row_ls.append(row)

    base_locations = gpd.GeoDataFrame(row_ls, geometry="geometry", crs="EPSG:4326")
    # initialize flag for splitting
    base_locations["need_split"] = True

    # not visited locations - stays at the coarsest level
    locs["s2_id"] = locs["center"].apply(get_loc_id, level=min_level)
    locs["level"] = min_level

    not_visited_locs = base_locations.loc[~base_locations["loc_id"].isin(locs["s2_id"].unique()), "loc_id"].values
    base_locations.loc[base_locations["loc_id"].isin(not_visited_locs), "need_split"] = False

    # locations that are only visited once empirically - stays at the current level
    current_locs = base_locations.copy()
    valid_loc_ls = []

    for level in range(min_level, max_level):

        current_level_locs = projected_locs.loc[projected_locs["level"] == level, "s2_id"].values
        current_locs.loc[current_locs["loc_id"].isin(current_level_locs), "need_split"] = False

        # save the valid locations and split the to be processed locations
        valid_loc_ls.append(current_locs.loc[current_locs["need_split"] == False])
        processing_locs = current_locs.loc[current_locs["need_split"] == True].copy()

        # get the child grids of the processing_locs, and assign them to a new processing_locs for the next loop
        row_ls = []
        for loc_id in processing_locs["loc_id"].values:

            # checked!
            for children in s2sphere.CellId(id_=int(loc_id)).children():
                ll = s2sphere.LatLng.from_point(s2sphere.Cell(children).get_center())

                row = {}
                row["loc_id"] = children.id()
                row["level"] = level + 1
                row["geometry"] = Point(ll.lng().degrees, ll.lat().degrees)
                row_ls.append(row)

        current_locs = gpd.GeoDataFrame(row_ls, geometry="geometry", crs="EPSG:4326")
        current_locs["need_split"] = True

    # the last current_locs contains all locations at the lowest level
    valid_loc_ls.append(current_locs)

    s2_locs = pd.concat(valid_loc_ls).sort_values(by="loc_id").drop(columns={"need_split"}).reset_index(drop=True)

    print(s2_locs["level"].value_counts(), len(s2_locs))
    # (level
    # 14    137864
    # 13      3486
    # 12       980
    # 11       147
    # 10        98
    # Name: count, dtype: int64,
    # 142575)
    return s2_locs


def get_empirical_visit(sp, s2_locs):

    sp = sp[["location_id"]].merge(projected_locs, left_on="location_id", right_on="id")
    loc_freq = sp.groupby("s2_id").size()

    s2_locs_freq = s2_locs.merge(loc_freq.to_frame(name="freq"), left_on="loc_id", right_on="s2_id", how="left")
    # not matched locations (not visited) receive a 0)
    s2_locs_freq["freq"] = s2_locs_freq["freq"].fillna(0)
    s2_locs_freq["freq"] = s2_locs_freq["freq"].astype(int)
    # norm
    s2_locs_freq["freq"] = s2_locs_freq["freq"] / s2_locs_freq["freq"].sum()

    # Get Polygon geometry for s2 Covering:
    geometry_rows = []
    for loc_id in s2_locs["loc_id"].values:
        row = {}
        row["geometry"] = Polygon(s2_to_geo_boundary(s2sphere.CellId(id_=int(loc_id))))
        geometry_rows.append(row)

    s2_locs_polygon = gpd.GeoDataFrame(geometry_rows, geometry="geometry", crs="EPSG:4326")

    s2_locs_freq["area"] = s2_locs_polygon["geometry"]

    return s2_locs_freq


def get_visited_loc(sp, projected_locs, s2_locs):
    sp = sp.copy()
    projected_locs = projected_locs.copy()
    s2_locs = s2_locs.copy()

    sp = load_data(sp, projected_locs)
    visited_loc = s2_locs.loc[s2_locs["loc_id"].isin(sp["location_id"].unique())].copy()

    # final clearning
    visited_loc.index = np.arange(len(visited_loc))
    visited_loc.index.name = "id"

    return visited_loc


if __name__ == "__main__":
    MIN_LEVEL = 10
    MAX_LEVEL = 14

    SP_NAME = "sp_mobis_all"
    LOC_NAME = "loc"

    LOC_SAVE_NAME = "loc_s2_level10_14"
    VISITED_LOC_SAVE_NAME = "s2_loc_visited_level10_14"
    SWISS_LOC_SAVE_NAME = "s2_loc_all_level10_14"

    # original staypoints
    sp = pd.read_csv(os.path.join(".", "data", f"{SP_NAME}.csv"))

    # original locations
    locs = pd.read_csv(os.path.join(".", "data", f"{LOC_NAME}.csv"))
    locs["center"] = locs["center"].apply(wkt.loads)
    locs = gpd.GeoDataFrame(locs, geometry="center", crs="EPSG:4326")

    # get swiss region in s2 format
    swiss_region_s2 = get_swiss_s2()

    # get the s2 Covering (single resolution)
    single_layer_covering_s2 = get_single_layer_covering_s2(swiss_region_s2, min_level=MIN_LEVEL, max_level=MAX_LEVEL)

    # Construct hierarchical grid for user locations
    projected_locs = get_loc_s2_id(locs, min_level=MIN_LEVEL, max_level=MAX_LEVEL)

    # Construct the s2 Covering (multi resolution)
    s2_locs = get_s2_loc_all_multi(single_layer_covering_s2, locs, min_level=MIN_LEVEL, max_level=MAX_LEVEL)

    # get the empirical visit frequency
    s2_locs_freq = get_empirical_visit(sp, s2_locs)

    # all location covering swiss
    s2_locs_freq.index.name = "id"
    s2_locs_freq.to_csv(os.path.join(".", "data", f"{SWISS_LOC_SAVE_NAME}.csv"), index=True)
    # original location with s2 cell ids
    projected_locs.to_csv(os.path.join(".", "data", f"{LOC_SAVE_NAME}.csv"), index=False)

    # get visited loc from all location
    visited_loc = get_visited_loc(sp, projected_locs, s2_locs_freq)
    visited_loc.to_csv(os.path.join(".", "data", f"{VISITED_LOC_SAVE_NAME}.csv"))
