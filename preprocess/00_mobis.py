import json
import os
import pickle as pickle
from pathlib import Path

import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm import tqdm
import argparse
import datetime

from shapely.geometry import LineString

from joblib import Parallel, delayed
import multiprocessing

# trackintel
import trackintel as ti
from trackintel.analysis.tracking_quality import temporal_tracking_quality


def get_dataset(CONFIG, epsilon=50):
    """Construct the raw staypoint with location id dataset from GC data."""
    # read file storage
    ## read and change name to trackintel format
    sp = pd.read_csv(os.path.join(CONFIG["raw_mobis"], "sps.csv"))
    # geometry
    sp["geometry"] = gpd.GeoSeries.from_wkt(sp["geometry"])
    sp = gpd.GeoDataFrame(sp, crs="EPSG:4326", geometry="geometry")

    sp["started_at"] = pd.to_datetime(sp["started_at"], format="mixed", yearfirst=True, utc=True)
    sp["finished_at"] = pd.to_datetime(sp["finished_at"], format="mixed", yearfirst=True, utc=True)
    sp = ti.io.read_staypoints_gpd(sp)

    print(type(sp))

    # tpls = pd.read_csv(os.path.join(CONFIG["raw_mobis"], "legs.csv"), nrows=100000)
    tpls = pd.read_csv(os.path.join(CONFIG["raw_mobis"], "legs.csv"), usecols=[0, 1, 3, 4, 6])
    tpls["mode"] = tpls["mode"].apply(lambda x: x[6:])

    # geometry
    tpls["geometry"] = gpd.GeoSeries.from_wkt(tpls["geometry"])
    tpls = gpd.GeoDataFrame(tpls, crs="EPSG:4326", geometry="geometry")

    # construct linestring from multilinestring
    def get_simple_line(multi):
        # multi = wkt.loads(str)
        multicoords = [list(line.coords) for line in multi.geoms]
        simple = LineString([item for sublist in multicoords for item in sublist])
        return simple

    MultiLSFlag = tpls.geometry.type == "MultiLineString"
    tpls.loc[MultiLSFlag, "geometry"] = tpls.loc[MultiLSFlag, "geometry"].apply(get_simple_line)

    tpls["started_at"] = pd.to_datetime(tpls["started_at"], format="mixed", yearfirst=True, utc=True)
    tpls["finished_at"] = pd.to_datetime(tpls["finished_at"], format="mixed", yearfirst=True, utc=True)

    # to trackintel, filter invalid geometry
    tpls = ti.io.read_triplegs_gpd(tpls[tpls.geometry.is_valid])

    print(type(tpls))

    # initial cleaning
    # negative duration records have already been dropped
    sp["duration"] = (sp["finished_at"] - sp["started_at"]).dt.total_seconds()
    tpls["duration"] = (tpls["finished_at"] - tpls["started_at"]).dt.total_seconds()

    sp = sp.sort_values(by="started_at").reset_index(drop=True)
    tpls = tpls.sort_values(by="started_at").reset_index(drop=True)

    sp.index.name = "id"
    tpls.index.name = "id"

    # ensure the timeline of sp and tpls does not overlap
    sp, tpls = filter_duplicates(sp.copy().reset_index(), tpls.reset_index())

    # define activity
    sp["is_activity"] = True

    # wait is not an activity
    sp.loc[sp["purpose"] == "wait", "is_activity"] = False

    # shorter than 25min
    sp.loc[(sp["purpose"] == "unknown") & (sp["duration"] < 25 * 60), "is_activity"] = False

    # the trackintel trip generation
    sp, tpls, trips = ti.preprocessing.triplegs.generate_trips(sp, tpls, gap_threshold=25, add_geometry=False)

    ## select valid user
    quality_path = os.path.join(".", "data", "quality")
    quality_file = os.path.join(quality_path, "mobis_filtered.csv")
    if Path(quality_file).is_file():
        valid_users = pd.read_csv(quality_file)["user_id"].values
    else:
        if not os.path.exists(quality_path):
            os.makedirs(quality_path)
        quality_filter = {"day_filter": 50, "window_size": 5, "min_thres": 0.5, "mean_thres": 0.6}
        valid_users = calculate_user_quality(sp.copy(), trips.copy(), quality_file, quality_filter)

    # filter
    sp = sp.loc[sp["user_id"].isin(valid_users)]
    tpls = tpls.loc[tpls["user_id"].isin(valid_users)]
    trips = trips.loc[trips["user_id"].isin(valid_users)]

    ## select only switzerland records
    swissBoundary = gpd.read_file(os.path.join(".", "data", "swiss", "swiss.shp"))
    print("Before spatial filtering: ", sp.shape[0])
    sp_swiss = _filter_within_swiss(sp, swissBoundary)
    print("After spatial filtering: ", sp_swiss.shape[0])

    # select activity
    sp_swiss_act = sp_swiss.loc[sp_swiss["is_activity"] == True]

    # assign travel mode
    sp_trip_fill_mode = assign_travel_mode(sp_swiss_act, tpls, trips)

    # generate locations
    sp, locs = sp_trip_fill_mode.as_staypoints.generate_locations(
        epsilon=epsilon, num_samples=1, distance_metric="haversine", agg_level="dataset", n_jobs=-1
    )
    # filter noise staypoints
    sp = sp.loc[~sp["location_id"].isna()].copy()
    print("After filter non-location staypoints: ", sp.shape[0])

    # save locations
    locs = locs[~locs.index.duplicated(keep="first")]
    filtered_locs = locs.loc[locs.index.isin(sp["location_id"].unique())]
    filtered_locs.as_locations.to_csv(os.path.join(".", "data", "loc.csv"))
    print("Location size: ", sp["location_id"].unique().shape[0], filtered_locs.shape[0])

    sp = sp[["user_id", "started_at", "finished_at", "geometry", "length", "mode", "location_id"]].reset_index(
        drop=True
    )
    sp.index.name = "id"
    # merge staypoints
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

    sp_merged["act_duration"] = sp_merged.groupby("user_id").apply(get_act_duration).values

    print("User size: ", len(sp_merged["user_id"].unique()))

    sp_merged.to_csv(os.path.join(".", "data", "sp_mobis_all.csv"))


def assign_travel_mode(sp, tpls, trips):
    ## assign travel mode
    tpls["length"] = tpls.to_crs("EPSG:2056").length
    #  get the number of triplegs for each trip
    groupsize = tpls.groupby("trip_id").size().to_frame(name="triplegNum").reset_index()
    tpls_num = tpls.merge(groupsize, on="trip_id")

    # trips only with 1 triplegs
    res1 = tpls_num.loc[tpls_num["triplegNum"] == 1][["trip_id", "length", "mode"]].copy()

    # get the mode and length of remaining trips
    remain = tpls_num.loc[tpls_num["triplegNum"] != 1].copy()

    remain.sort_values(by="length", inplace=True, ascending=False)
    mode = remain.groupby("trip_id").head(1).reset_index(drop=True)[["mode", "trip_id"]]

    length = remain.groupby("trip_id")["length"].sum().reset_index()
    res2 = mode.merge(length, on="trip_id")

    # merge
    res = pd.concat([res1, res2])

    # cleaning
    res.rename(columns={"trip_id": "id"}, inplace=True)
    res.set_index("id", inplace=True)

    # merge to trip df
    trips_mode = trips.join(res, how="left")

    encode_dict = {
        "Car": "Car",
        "Walk": "Walk",
        "Bicycle": "Bicycle",
        "Bus": "Bus",
        "LightRail": "Train",
        "Train": "Train",
        "Tram": "Tram",
        "RegionalTrain": "Train",
        "Ebicycle": "Bicycle",
        "MotorbikeScooter": "Car",
        "Motorbike": "Car",
        "Subway": "Tram",
        "Airplane": "Other",
        "Boat": "Other",
        "Ski": "Other",
        "TaxiUber": "Car",
        "CarsharingMobility": "Car",
        "Scooter": "Bicycle",
        "Cablecar": "Bus",
        "RidepoolingPikmi": "Car",
        "Etrottinett": "Bicycle",
        "Bikesharing": "Bicycle",
        "Escooter": "Bicycle",
        "Ferry": "Other",
    }
    trips_mode["mode"] = trips_mode["mode"].apply(lambda x: encode_dict[x])

    # combine with sp df
    with_pre_trip = sp.loc[~sp["prev_trip_id"].isna()].copy()

    with_pre_res = with_pre_trip.merge(
        trips_mode.reset_index()[["length", "mode", "id"]], how="left", left_on="prev_trip_id", right_on="id"
    )

    no_pre_trip = sp.loc[sp["prev_trip_id"].isna()].copy()
    no_pre_trip["length"] = 0
    no_pre_trip["mode"] = "None"

    # concat result
    sp_trip = pd.concat([with_pre_res, no_pre_trip]).drop(columns=["prev_trip_id", "next_trip_id", "trip_id", "id"])

    sp_trip.sort_values(by=["user_id", "started_at"], inplace=True)
    sp_trip.reset_index(drop=True, inplace=True)
    sp_trip.index.name = "id"

    def assign_unknown_modes(df):
        df.loc[df["mode"] == "None", "mode"] = pd.NA
        df["mode"] = df["mode"].ffill(axis=0).bfill(axis=0)
        return df

    sp_trip_fill_mode = sp_trip.groupby("user_id").apply(assign_unknown_modes, include_groups=False).reset_index()
    sp_trip_fill_mode.index.name = "id"

    return sp_trip_fill_mode


def _filter_within_swiss(stps, swissBound):
    """Spatial filtering of staypoints."""
    # save a copy of the original projection
    init_crs = stps.crs
    # project to projected system
    stps = stps.to_crs(swissBound.crs)

    ## parallel for speeding up
    stps["within"] = _apply_parallel(stps["geometry"], _apply_extract, swissBound)
    sp_swiss = stps[stps["within"] == True].copy()
    sp_swiss.drop(columns=["within"], inplace=True)

    return sp_swiss.to_crs(init_crs)


def _apply_extract(df, swissBound):
    """The func for _apply_parallel: judge whether inside a shp."""
    tqdm.pandas(desc="pandas bar")
    shp = swissBound["geometry"].to_numpy()[0]
    return df.progress_apply(lambda x: shp.contains(x))


def _apply_parallel(df, func, other, n=-1):
    """parallel apply for spending up."""
    if n is None:
        n = -1
    dflength = len(df)
    cpunum = multiprocessing.cpu_count()
    if dflength < cpunum:
        spnum = dflength
    if n < 0:
        spnum = cpunum + n + 1
    else:
        spnum = n or 1

    sp = list(range(dflength)[:: int(dflength / spnum + 0.5)])
    sp.append(dflength)
    slice_gen = (slice(*idx) for idx in zip(sp[:-1], sp[1:]))
    results = Parallel(n_jobs=n, verbose=0)(delayed(func)(df.iloc[slc], other) for slc in slice_gen)
    return pd.concat(results)


def _alter_diff(df):
    df.sort_values(by="started_at", inplace=True)
    df["diff"] = pd.NA
    # for correct dtype
    df["st_next"] = df["started_at"]

    diff = df["started_at"].iloc[1:].reset_index(drop=True) - df["finished_at"].iloc[:-1].reset_index(drop=True)
    df.iloc[:-1, df.columns.get_loc("diff")] = diff.dt.total_seconds()
    df.iloc[:-1, df.columns.get_loc("st_next")] = df["started_at"].iloc[1:].reset_index(drop=True)

    df.loc[df["diff"] < 0, "finished_at"] = df.loc[df["diff"] < 0, "st_next"]

    df["started_at"], df["finished_at"] = pd.to_datetime(df["started_at"]), pd.to_datetime(df["finished_at"])
    df["duration"] = (df["finished_at"] - df["started_at"]).dt.total_seconds()

    # print(df.loc[df["diff"] < 0])
    df.drop(columns=["diff", "st_next"], inplace=True)
    df.drop(index=df[df["duration"] <= 0].index, inplace=True)

    return df


def filter_duplicates(sp, tpls):

    # merge trips and staypoints
    sp["type"] = "sp"
    tpls["type"] = "tpl"
    df_all = pd.merge(sp, tpls, how="outer")

    df_all = df_all.groupby("user_id").apply(_alter_diff, include_groups=False).reset_index()
    sp = df_all.loc[df_all["type"] == "sp"].drop(columns=["type"])
    tpls = df_all.loc[df_all["type"] == "tpl"].drop(columns=["type"])

    sp = sp[
        [
            "id",
            "user_id",
            "started_at",
            "finished_at",
            "geometry",
            "duration",
            "purpose",
            "detected_purpose",
            "overseas",
        ]
    ]
    tpls = tpls[["id", "user_id", "started_at", "finished_at", "duration", "mode", "geometry"]]

    return sp.set_index("id"), tpls.set_index("id")


def _split_overlaps(source, granularity="day", max_iter=60):
    if granularity == "hour":
        # every split over hour splits also over day
        # this way to split of an entry over a month takes 30+24 iterations instead of 30*24.
        df = _split_overlaps(source, granularity="day", max_iter=max_iter)
    else:
        df = source.copy()

    change_flag = _get_split_index(df, granularity=granularity)
    iter_count = 0

    freq = "D" if granularity == "day" else "H"
    # Iteratively split one day/hour from multi day/hour entries until no entry spans over multiple days/hours
    while change_flag.sum() > 0:
        # calculate new finished_at timestamp (00:00 midnight)
        new_df = df.loc[change_flag].copy()
        # print(change_flag)
        # print(new_df)
        df.loc[change_flag, "finished_at"] = (df.loc[change_flag, "started_at"] + pd.Timestamp.resolution).dt.ceil(freq)

        # create new entries with remaining timestamp
        new_df["started_at"] = df.loc[change_flag, "finished_at"]

        df = pd.concat((df, new_df), ignore_index=True, sort=True)

        change_flag = _get_split_index(df, granularity=granularity)
        iter_count += 1
        if iter_count >= max_iter:
            break

    if "duration" in df.columns:
        df["duration"] = df["finished_at"] - df["started_at"]
    return df


def _get_split_index(df, granularity="day"):
    freq = "D" if granularity == "day" else "H"
    cond1 = df["started_at"].dt.floor(freq) != (df["finished_at"] - pd.Timedelta.resolution).dt.floor(freq)
    # catch corner case where both on same border and subtracting would lead to error
    cond2 = df["started_at"] != df["finished_at"]
    return cond1 & cond2


def _filter_user(df, min_thres, mean_thres):
    consider = df.loc[df["quality"] != 0]
    if (consider["quality"].min() > min_thres) and (consider["quality"].mean() > mean_thres):
        return df


def _get_tracking_quality(df, window_size):

    weeks = (df["finished_at"].max() - df["started_at"].min()).days // 7
    start_date = df["started_at"].min().date()

    quality_list = []
    # construct the sliding week gdf
    for i in range(0, weeks - window_size + 1):
        curr_start = datetime.datetime.combine(start_date + datetime.timedelta(weeks=i), datetime.time())
        curr_end = datetime.datetime.combine(curr_start + datetime.timedelta(weeks=window_size), datetime.time())

        # the total df for this time window
        cAll_gdf = df.loc[(df["started_at"] >= curr_start) & (df["finished_at"] < curr_end)]
        if cAll_gdf.shape[0] == 0:
            continue
        total_sec = (curr_end - curr_start).total_seconds()

        quality_list.append([i, cAll_gdf["duration"].sum() / total_sec])
    ret = pd.DataFrame(quality_list, columns=["timestep", "quality"])
    ret["user_id"] = df["user_id"].unique()[0]
    return ret


def calculate_user_quality(sp, trips, file_path, quality_filter):

    trips["started_at"] = pd.to_datetime(trips["started_at"]).dt.tz_localize(None)
    trips["finished_at"] = pd.to_datetime(trips["finished_at"]).dt.tz_localize(None)
    sp["started_at"] = pd.to_datetime(sp["started_at"]).dt.tz_localize(None)
    sp["finished_at"] = pd.to_datetime(sp["finished_at"]).dt.tz_localize(None)

    # merge trips and staypoints
    print("starting merge", sp.shape, trips.shape)
    sp["type"] = "sp"
    trips["type"] = "tpl"
    all_df = pd.concat([sp, trips])
    print("finished merge", all_df.shape)
    print("*" * 50)
    all_df = _split_overlaps(all_df, granularity="day")
    all_df["duration"] = (all_df["finished_at"] - all_df["started_at"]).dt.total_seconds()

    print(len(all_df["user_id"].unique()))

    # get quality
    total_quality = temporal_tracking_quality(all_df, granularity="all")
    # get tracking days
    total_quality["days"] = (
        all_df.groupby("user_id").apply(lambda x: (x["finished_at"].max() - x["started_at"].min()).days).values
    )
    # filter based on days
    user_filter_day = (
        total_quality.loc[(total_quality["days"] > quality_filter["day_filter"])]
        .reset_index(drop=True)["user_id"]
        .unique()
    )
    # filter based on sliding quality
    sliding_quality = (
        all_df.groupby("user_id")
        .apply(_get_tracking_quality, window_size=quality_filter["window_size"])
        .reset_index(drop=True)
    )

    filter_after_day = sliding_quality.loc[sliding_quality["user_id"].isin(user_filter_day)]

    if "min_thres" in quality_filter:
        # filter based on quanlity
        filter_after_day = (
            filter_after_day.groupby("user_id")
            .apply(_filter_user, min_thres=quality_filter["min_thres"], mean_thres=quality_filter["mean_thres"])
            .reset_index(drop=True)
            .dropna()
        )

    filter_after_user_quality = filter_after_day.groupby("user_id", as_index=False)["quality"].mean()

    print("final selected user", filter_after_user_quality.shape[0])
    filter_after_user_quality.to_csv(file_path, index=False)
    return filter_after_user_quality["user_id"].values


if __name__ == "__main__":
    # read file storage
    DBLOGIN_FILE = os.path.join(".", "paths.json")
    with open(DBLOGIN_FILE) as json_file:
        CONFIG = json.load(json_file)

    parser = argparse.ArgumentParser()
    parser.add_argument("epsilon", type=int, nargs="?", help="epsilon for dbscan to detect locations", default=20)
    args = parser.parse_args()

    get_dataset(epsilon=args.epsilon, CONFIG=CONFIG)
