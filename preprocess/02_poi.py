import os
import geopandas as gpd
import pandas as pd
import numpy as np
import pickle as pickle
from tqdm import tqdm

from shapely import wkt

from gensim.corpora import Dictionary
from gensim.models import LdaModel
import gensim

import trackintel as ti


def _read_poi_files():
    # get all the pofws, 1
    pofw = gpd.read_file(os.path.join(".", "data", "poi", "ori", "gis_osm_pofw_free_1.shp"))
    # area pofw transformed into points
    pofw_a = gpd.read_file(os.path.join(".", "data", "poi", "ori", "gis_osm_pofw_a_free_1.shp"))
    pofw_a["geometry"] = pofw_a.to_crs("EPSG:2056").geometry.centroid.to_crs("EPSG:4326")
    #
    pofw = pd.concat([pofw, pofw_a])
    pofw = pofw.drop_duplicates(subset="osm_id")

    # get all the transport
    transport = gpd.read_file(os.path.join(".", "data", "poi", "ori", "gis_osm_transport_free_1.shp"))
    # area transport transformed into points
    transport_a = gpd.read_file(os.path.join(".", "data", "poi", "ori", "gis_osm_transport_a_free_1.shp"))
    transport_a["geometry"] = transport_a.to_crs("EPSG:2056").geometry.centroid.to_crs("EPSG:4326")
    #
    transport = pd.concat([transport, transport_a])
    transport = transport.drop_duplicates(subset="osm_id")

    # drop the trees, 1
    #
    natural = gpd.read_file(os.path.join(".", "data", "poi", "ori", "gis_osm_natural_free_1.shp"))

    # drop the trees: code = 4121
    natural = natural.loc[natural["code"] != 4121]

    # area natural transformed into points
    natural_a = gpd.read_file(os.path.join(".", "data", "poi", "ori", "gis_osm_natural_a_free_1.shp"))
    natural_a = natural_a.loc[natural_a["code"] != 4121]
    natural_a["geometry"] = natural_a.to_crs("EPSG:2056").geometry.centroid.to_crs("EPSG:4326")

    natural = pd.concat([natural, natural_a])
    natural = natural.drop_duplicates(subset="osm_id")

    # get all the pois, 11
    poi = gpd.read_file(os.path.join(".", "data", "poi", "ori", "gis_osm_pois_free_1.shp"))
    # area poi transformed into points
    poi_a = gpd.read_file(os.path.join(".", "data", "poi", "ori", "gis_osm_pois_a_free_1.shp"))
    poi_a["geometry"] = poi_a.to_crs("EPSG:2056").geometry.centroid.to_crs("EPSG:4326")
    #
    poi = pd.concat([poi, poi_a])
    poi = poi.drop_duplicates(subset="osm_id")

    # get the parking slots of traffic point file
    traffic = gpd.read_file(os.path.join(".", "data", "poi", "ori", "gis_osm_traffic_free_1.shp"))

    # drop the crossing, turning_circle, street_lamp: code = 5204, 5207, 5209
    traffic = traffic.loc[~traffic["code"].isin([5204, 5207, 5209])]

    # area traffic transformed into points
    traffic_a = gpd.read_file(os.path.join(".", "data", "poi", "ori", "gis_osm_traffic_a_free_1.shp"))
    traffic_a["geometry"] = traffic_a.to_crs("EPSG:2056").geometry.centroid.to_crs("EPSG:4326")

    traffic = pd.concat([traffic, traffic_a])
    traffic = traffic.drop_duplicates(subset="osm_id")

    # building
    buildings = gpd.read_file(os.path.join(".", "data", "poi", "ori", "gis_osm_buildings_a_free_1.shp"))
    # all building get the same code 1500
    buildings["geometry"] = buildings.to_crs("EPSG:2056").geometry.centroid.to_crs("EPSG:4326")

    poi_gdf = pd.concat([pofw, transport, natural, poi, traffic, buildings])
    poi_gdf.drop(columns={"type"}, inplace=True)

    return poi_gdf


def _assign_category(df):
    # 2018 Hong: Life services, Office building/space, Other facilities, Medical/Education, Entertainment, Government, Residence communities, Financial services
    # 2021 Yin : Residential, Hotel/resort, Mixed-use, K–12 schools, University/college, Office/workplace, Services, Civic/religious, Shopping/retail, Recreation/entertainment, Transportation, Others

    ### osm code -> 2018 Hong -> 2021 Yin
    # public 20xx  -> Residence communities ->  Residential
    #    university + school + kindergarten + college (208x) -> Medical/Education -> K–12 schools/University/college
    # health 21xx -> Medical/Education -> Services
    # leisure 22xx -> Entertainment -> Recreation/entertainment
    # catering 23xx -> Life services -> Residential
    # accommodation 24xx -> Entertainment -> Hotel/resort
    # shopping 25xx -> Life services -> Shopping/retail
    # money 26xx -> Financial services -> Services
    # tourism 27xx -> Entertainment -> Recreation/entertainment
    # pofw 3xxx -> Life services -> Civic/religious
    # natural 41xx -> Entertainment -> Recreation/entertainment
    # transport 56xx -> Other facilities -> Transportation
    # miscpoi 29xx -> Other facilities -> Others

    # note: miscpoi contains "bench" or "drinking_water" that might not reveal any landuse info

    # init
    df["category"] = "Unknown"

    # public 20xx  -> Residence communities ->  Residential
    #    university + school + kindergarten + college (208x) -> Medical/Education -> K–12 schools/University/college
    df.loc[(df["code"] > 2000) & (df["code"] < 2100), "category"] = "Residential"
    df.loc[(df["code"] > 2080) & (df["code"] < 2090), "category"] = "Schools"

    # health 21xx -> Medical/Education -> Services
    df.loc[(df["code"] > 2100) & (df["code"] < 2200), "category"] = "Services"

    # leisure 22xx -> Entertainment -> Recreation/entertainment
    df.loc[(df["code"] > 2200) & (df["code"] < 2300), "category"] = "Entertainment"

    # catering 23xx -> Life services -> Residential
    df.loc[(df["code"] > 2300) & (df["code"] < 2400), "category"] = "Residential"

    # accommodation 24xx -> Entertainment -> Hotel/resort
    df.loc[(df["code"] > 2400) & (df["code"] < 2500), "category"] = "Entertainment"

    # shopping 25xx -> Life services -> Shopping/retail
    df.loc[(df["code"] > 2500) & (df["code"] < 2600), "category"] = "Shopping"

    # money 26xx -> Financial services -> Services
    df.loc[(df["code"] > 2600) & (df["code"] < 2700), "category"] = "Services"

    # tourism 27xx -> Entertainment -> Recreation/entertainment
    df.loc[(df["code"] > 2700) & (df["code"] < 2800), "category"] = "Entertainment"

    # miscpoi 29xx -> Other facilities -> Others
    df.loc[(df["code"] > 2900) & (df["code"] < 3000), "category"] = "Others"
    df.loc[(df["code"] == 1500), "category"] = "Others"

    # pofw 3xxx -> Life services -> Civic/religious
    df.loc[(df["code"] > 3000) & (df["code"] < 4000), "category"] = "Civic"

    # natural 41xx -> Entertainment -> Recreation/entertainment
    df.loc[(df["code"] > 4000) & (df["code"] < 5000), "category"] = "Entertainment"

    # transport 56xx -> Other facilities -> Transportation
    df.loc[(df["code"] > 5600) & (df["code"] < 5700), "category"] = "Transportation"
    # traffic 54xx -> Other facilities -> Transportation
    df.loc[(df["code"] > 5200) & (df["code"] < 5400), "category"] = "Transportation"

    # Unknown           2737932
    # Others             127119
    # Entertainment       93521
    # Shopping            48116
    # Residential         42271
    # Transportation      39290
    # Services             9010
    # Schools              2850
    # Civic                 765

    print(df["category"].value_counts())
    return df


def preprocess(poi_save_name):
    gdf = _read_poi_files()

    # assign category for tf-idf calculation
    gdf = _assign_category(gdf)

    # final cleaning
    gdf.drop(columns=["osm_id", "fclass"], inplace=True)
    # reindex
    gdf.reset_index(drop=True, inplace=True)
    gdf.index.name = "id"
    gdf.reset_index(inplace=True)

    # change the projection and save
    gdf = gdf.to_crs("EPSG:2056")
    gdf.to_file(os.path.join(".", "data", "poi", f"{poi_save_name}.shp"))


def get_poi_representation(poi_save_name, final_save_name, locs, categories=16):
    # checked: buffer method; transform to final_poi; vector values are different

    # read poi file
    poi = gpd.read_file(os.path.join(".", "data", "poi", f"{poi_save_name}.shp"))
    spatial_index = poi.sindex

    tqdm.pandas(desc="Generating poi within")
    locs["poi_within"] = locs["geometry"].progress_apply(_get_inside_pois, poi=poi, spatial_index=spatial_index)

    # cleaning and expanding to location_id-poi_id pair
    loc_no_geo = locs.drop(columns="geometry")

    # explode preserves nan - preserves locs with no poi
    locs_poi = loc_no_geo.explode(column="poi_within")

    # get the poi info from original poi df
    locs_poi = locs_poi.merge(poi[["id", "category", "code"]], left_on="poi_within", right_on="id", how="left")
    locs_poi.drop(columns=["id"], inplace=True)

    valid_pairs = locs_poi.dropna(subset=["poi_within"]).copy()
    valid_pairs["code"] = valid_pairs["code"].astype(int).astype(str)

    poiValues = _lda(valid_pairs, categories=categories)
    locs_rep = locs.merge(poiValues, on="loc_id", how="left")

    # create all 0 array for no-poi-locations
    locs_rep.loc[locs_rep["poiValues"].isna(), "poiValues"] = locs_rep.loc[
        locs_rep["poiValues"].isna(), "poiValues"
    ].apply(lambda x: np.zeros(categories))

    locs_rep["poiValues"] = locs_rep["poiValues"].apply(lambda x: np.array(x, dtype=np.float32))

    ## save to disk
    locs_rep.to_csv(os.path.join(".", "data", f"{final_save_name}.csv"), index=False)
    data = {}
    data["poiValues"] = np.vstack(locs_rep["poiValues"].values)
    data["loc_id"] = locs_rep["loc_id"].values
    with open(os.path.join(".", "data", f"{final_save_name}.pk"), "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _get_inside_pois(df, poi, spatial_index):
    """
    Given one extent (df), return the poi within this extent.
    spatial_index is obtained from poi.sindex to speed up the process.
    """
    possible_matches_index = list(spatial_index.intersection(df.bounds))
    possible_matches = poi.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.within(df)]["id"].values

    return precise_matches


def _lda(df, categories=16):
    """Note: deal with the osm assigned "code" field."""
    texts = df.groupby("loc_id")["code"].apply(list).to_list()

    dct = Dictionary(texts)
    corpus = [dct.doc2bow(line) for line in texts]

    lda = LdaModel(corpus, num_topics=categories)
    vector = lda[corpus]

    # the lda array
    dense_ldavector = gensim.matutils.corpus2dense(vector, num_terms=categories).T
    # the index arr
    index_arr = df.groupby("loc_id", as_index=False).count()["loc_id"].values

    poiValues = pd.Series(list(dense_ldavector))
    poiValues.index = index_arr

    poiValues.name = "poiValues"
    poiValues.index.name = "loc_id"

    return poiValues.reset_index()


if __name__ == "__main__":
    ALL_LOCATION_NAME = "s2_loc_all_level10_14"
    POI_SAVE_NAME = "final_pois"
    FINAL_SAVE_NAME = "s2_loc_poi_level10_14"

    preprocess(POI_SAVE_NAME)

    # read location and change the geometry columns
    locs = pd.read_csv(os.path.join(".", "data", f"{ALL_LOCATION_NAME}.csv"), index_col="id")
    locs = locs.drop(columns="geometry").rename(columns={"area": "geometry"})
    locs["geometry"] = locs["geometry"].apply(wkt.loads)

    locs = gpd.GeoDataFrame(locs, geometry="geometry", crs="EPSG:4326")
    locs = locs.to_crs("EPSG:2056")

    get_poi_representation(POI_SAVE_NAME, FINAL_SAVE_NAME, locs=locs, categories=32)
