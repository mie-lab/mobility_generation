import json
import os
from pathlib import Path
import pandas as pd
import geopandas as gpd
import argparse

# trackintel
import trackintel as ti


def get_dataset(config, epsilon=50):
    """Construct the raw staypoint with location id dataset."""
    ## read
    pfs, mode_labels = ti.io.read_geolife(os.path.join(config["raw_geolife"], "Data"), print_progress=True)
    # generate staypoints, triplegs and trips
    pfs, sp = pfs.generate_staypoints(time_threshold=5.0, gap_threshold=1e6, print_progress=True, n_jobs=-1)
    sp["duration"] = (sp["finished_at"] - sp["started_at"]).dt.total_seconds()

    pfs, tpls = pfs.generate_triplegs(sp)
    tpls = ti.io.geolife_add_modes_to_triplegs(tpls, mode_labels)

    sp = ti.analysis.create_activity_flag(sp, time_threshold=15)

    sp, tpls, trips = ti.preprocessing.triplegs.generate_trips(sp, tpls, gap_threshold=15, add_geometry=False)

    # assign mode
    tpls["pred_mode"] = ti.analysis.predict_transport_mode(tpls)["mode"]
    tpls.loc[tpls["mode"].isna(), "mode"] = tpls.loc[tpls["mode"].isna(), "pred_mode"]
    tpls.drop(columns={"pred_mode"}, inplace=True)

    # get the length
    tpls["length"] = ti.geogr.calculate_haversine_length(tpls)

    groupsize = tpls.groupby("trip_id").size().to_frame(name="triplegNum").reset_index()
    tpls_group = tpls.merge(groupsize, on="trip_id")

    # trips only with 1 triplegs
    res1 = tpls_group.loc[tpls_group["triplegNum"] == 1][["trip_id", "length", "mode"]].copy()

    # get the mode and length of remaining trips
    remain = tpls_group.loc[tpls_group["triplegNum"] != 1].copy()
    remain.sort_values(by="length", inplace=True, ascending=False)
    mode = remain.groupby("trip_id").head(1).reset_index(drop=True)[["mode", "trip_id"]]

    length = remain.groupby("trip_id")["length"].sum().reset_index()
    res2 = mode.merge(length, on="trip_id")
    # concat the results
    res = pd.concat([res1, res2])
    res.rename(columns={"trip_id": "id"}, inplace=True)
    res.set_index("id", inplace=True)

    trips_with_main_mode = trips.join(res, how="left")
    trips_with_main_mode = trips_with_main_mode[~trips_with_main_mode["mode"].isna()]
    trips_with_main_mode_cate = _get_mode(trips_with_main_mode)

    print(trips_with_main_mode_cate["mode"].value_counts())

    # filter activity staypoints
    sp = sp.loc[sp["is_activity"] == True].drop(columns=["is_activity", "trip_id", "next_trip_id"])

    # generate locations
    sp, locs = sp.generate_locations(
        epsilon=epsilon, num_samples=1, distance_metric="haversine", agg_level="dataset", n_jobs=-1
    )
    # filter noise staypoints
    sp = sp.loc[~sp["location_id"].isna()].copy()
    print("After filter non-location staypoints: ", sp.shape[0])

    # save locations
    locs = locs[~locs.index.duplicated(keep="first")]
    filtered_locs = locs.loc[locs.index.isin(sp["location_id"].unique())]

    path = Path(os.path.join(".", "data"))
    if not os.path.exists(path):
        os.makedirs(path)
    filtered_locs.rename(columns={"center": "geometry"}).to_csv(os.path.join(".", "data", "loc_geolife.csv"))
    print("Location size: ", sp["location_id"].unique().shape[0], filtered_locs.shape[0])

    # merge staypoint with trips info, sp with mode
    sp = sp.loc[~sp["prev_trip_id"].isna()].reset_index().copy()
    trips = (
        trips_with_main_mode_cate.drop(columns=["started_at", "finished_at", "user_id"])
        .reset_index()
        .rename(columns={"id": "trip_id"})
        .copy()
    )
    sp["prev_trip_id"] = sp["prev_trip_id"].astype(float)
    trips["trip_id"] = trips["trip_id"].astype(float)

    merged_sp = sp.merge(trips, left_on="prev_trip_id", right_on="trip_id", how="left")
    sp = merged_sp.loc[~merged_sp["trip_id"].isna()].drop(
        columns=["origin_staypoint_id", "prev_trip_id", "destination_staypoint_id", "elevation", "trip_id"]
    )

    # sp_time = sp.groupby("user_id").apply(_get_time, include_groups=False).reset_index()

    sp = gpd.GeoDataFrame(sp.rename(columns={"geom": "geometry"}), crs="EPSG:4326", geometry="geometry")
    sp = sp[["user_id", "started_at", "finished_at", "geometry", "length", "mode", "location_id"]].reset_index(
        drop=True
    )
    sp.index.name = "id"

    sp_merged = sp.as_staypoints.merge_staypoints(
        triplegs=pd.DataFrame([]),
        max_time_gap="1min",
        agg={"location_id": "first", "mode": "first", "length": "sum", "geometry": "first"},
    )
    print("After staypoints merging: ", sp_merged.shape[0])

    # recalculate staypoint duration
    sp_merged.sort_values(by=["user_id", "started_at"], inplace=True)
    sp_merged["duration"] = ((sp_merged["finished_at"] - sp_merged["started_at"]).dt.total_seconds() / 60).round()

    # get the time info
    def get_act_duration(df):
        df["act_duration"] = pd.NA
        df["act_duration"] = (
            (df["finished_at"].shift(-1) - df["finished_at"]).dt.total_seconds().shift(1) / 60
        ).round()

        df.iloc[0, df.columns.get_loc("act_duration")] = df["duration"].iloc[0]

        return df["act_duration"]

    sp_merged["act_duration"] = sp_merged.groupby("user_id").apply(get_act_duration, include_groups=False).values

    # get the time info
    print("User size: ", len(sp_merged["user_id"].unique()))

    sp_merged.to_csv(os.path.join(".", "data", "sp_geolife_all.csv"))


def _get_mode(df):
    # slow_mobility
    df.loc[df["mode"] == "slow_mobility", "mode"] = "slow"
    df.loc[df["mode"] == "bike", "mode"] = "slow"
    df.loc[df["mode"] == "walk", "mode"] = "slow"
    df.loc[df["mode"] == "run", "mode"] = "slow"

    # motorized_mobility
    df.loc[df["mode"] == "motorized_mobility", "mode"] = "motorized"
    df.loc[df["mode"] == "bus", "mode"] = "motorized"
    df.loc[df["mode"] == "car", "mode"] = "motorized"
    df.loc[df["mode"] == "subway", "mode"] = "motorized"
    df.loc[df["mode"] == "taxi", "mode"] = "motorized"
    df.loc[df["mode"] == "train", "mode"] = "motorized"
    df.loc[df["mode"] == "boat", "mode"] = "motorized"

    # fast_mobility
    df.loc[df["mode"] == "fast_mobility", "mode"] = "fast"
    df.loc[df["mode"] == "airplane", "mode"] = "fast"
    return df


if __name__ == "__main__":
    DBLOGIN_FILE = os.path.join(".", "paths.json")
    with open(DBLOGIN_FILE) as json_file:
        CONFIG = json.load(json_file)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=int, nargs="?", help="epsilon for dbscan to detect locations", default=20)
    args = parser.parse_args()

    get_dataset(epsilon=args.epsilon, config=CONFIG)
