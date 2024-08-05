import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from tqdm import tqdm

from networkx.algorithms import isomorphism
import networkx as nx

from trackintel.geogr.point_distances import haversine_dist
from trackintel.analysis.tracking_quality import _split_overlaps


def random_entropy(sp, print_progress=False):
    """Random entropy of individual visited locations.

    Parameters
    ----------
    sp : Geodataframe
        Staypoints with column "location_id".

    print_progress: boolen, default False
        Show per-user progress if set to True.

    Returns
    -------
    s: pd.Series
        the random entropy calculated at the individual level.

    References
    ----------
    [1] Song, C., Qu, Z., Blumm, N. and Barabási, A.L., 2010. Limits of predictability in human mobility. Science, 327(5968), pp.1018-1021.

    """
    if print_progress:
        tqdm.pandas(desc="User random entropy calculation")
        s = sp.groupby("user_id").progress_apply(lambda x: _random_entropy_user(x))
    else:
        s = sp.groupby("user_id").apply(lambda x: _random_entropy_user(x))

    s.rename("randomEntropy", inplace=True)
    return s


def uncorrelated_entropy(sp, print_progress=False):
    """
    Uncorrelated entropy of individual visited locations.

    Parameters
    ----------
    stps : Geodataframe
        Staypoints with column "location_id".

    print_progress: boolen, default False
        Show per-user progress if set to True.

    Returns
    -------
    pandas DataFrame
        the temporal-uncorrelated entropy of the individuals.

    References
    ----------
    [1] Song, C., Qu, Z., Blumm, N. and Barabási, A.L., 2010. Limits of predictability in human mobility. Science, 327(5968), pp.1018-1021.

    """
    if print_progress:
        tqdm.pandas(desc="User uncorrelated entropy calculation")
        s = sp.groupby("user_id").progress_apply(lambda x: _uncorrelated_entropy_user(x))
    else:
        s = sp.groupby("user_id").apply(lambda x: _uncorrelated_entropy_user(x))

    s.rename("uncorrelatedEntropy", inplace=True)
    return s


def real_entropy(sp, print_progress=False, n_jobs=-1):
    """
    Real entropy of individual visited locations.

    Parameters
    ----------
    stps : Geodataframe
        Staypoints with column "location_id".

    print_progress: boolen, default False
        Show per-user progress if set to True.

    Returns
    -------
    pandas DataFrame
        the real entropy of the individuals.

    References
    ----------
    [1] Song, C., Qu, Z., Blumm, N. and Barabási, A.L., 2010. Limits of predictability in human mobility. Science, 327(5968), pp.1018-1021.

    """
    s = applyParallel(sp.groupby("user_id"), _real_entropy_user, print_progress=print_progress, n_jobs=n_jobs)
    s.index.name = "user_id"
    s.rename("realEntropy", inplace=True)
    return s


def _random_entropy_user(sp_user):
    """
    User level random entropy calculation, see random_entropy() for details.

    Parameters
    ----------
    stps_user : Geodataframe
        The staypoints from an individual, should contain column "location_id".

    Returns
    -------
    float
        the random entropy of the individual
    """
    return np.log(len(sp_user["location_id"].unique()))


def _uncorrelated_entropy_user(sp_user):
    """
    User level uncorrelated entropy calculation, see uncorrelated_entropy() for details.

    Parameters
    ----------
    stps_user : Geodataframe
        The staypoints from an individual, should contain column "location_id".

    Returns
    -------
    float
        the temporal-uncorrelated entropy of the individual
    """
    locs_prob = sp_user["location_id"].value_counts(normalize=True, sort=False).values
    return -(locs_prob * np.log(locs_prob)).sum()


def _real_entropy_user(sp_user):
    """
    User level real entropy calculation, see real_entropy() for details.

    Parameters
    ----------
    stps_user : Geodataframe
        The staypoints from an individual, should contain column "location_id".

    Returns
    -------
    float
        the real entropy of the individual
    """
    locs_series = sp_user["location_id"].values

    n = len(locs_series)

    # 1 to ensure to consider the first situation from where
    # locs_series[i:j] = [] and locs_series[i:j] = locs_series[0:1]
    sum_lambda = 1

    for i in range(1, n - 1):
        j = i + 1

        while True:
            # if the locs_series[i:j] is longer than locs_series[:i],
            # we can no longer find it locs_series[i:j] in locs_series[:i]
            if j - i > i:
                break

            # if locs_series[i:j] exist in locs_series[:i], we increase j by 1
            # sliding_window_view creates sublist of length len(locs_series[i:j]) from locs_series[:i]
            ls = np.lib.stride_tricks.sliding_window_view(locs_series[:i], j - i).tolist()
            if tuple(locs_series[i:j]) in list(map(tuple, ls)):
                # if the subsequence already exist, we increase the sequence by 1, and check again
                j += 1
            else:
                # we find the "shortest substring" that does not exist in locs_series[:i]
                break

        # length of the substring
        sum_lambda += j - i

    # the function S5 from the suppl. material
    return 1.0 / (sum_lambda * 1 / n) * np.log(n)


def applyParallel(dfGrouped, func, n_jobs, print_progress, **kwargs):
    df_ls = Parallel(n_jobs=n_jobs)(
        delayed(func)(group, **kwargs) for _, group in tqdm(dfGrouped, disable=not print_progress)
    )
    return pd.Series(df_ls)


def radius_gyration(sp, print_progress=False, method="count"):
    """
    Radius of gyration for individuals.

    Parameters
    ----------
    sp : Geodataframe
        Staypoints with column "user_id" and geometry.

    print_progress: boolen, default False
        Show per-user progress if set to True.

    method: string, {"duration", "count"}, default "count"
        method to calculate rg. Duration additionally weights each sp with the activity duration.

    Returns
    -------
    pandas Series
        the radius of gyration for individuals.

    References
    ----------
    [1] Gonzalez, M. C., Hidalgo, C. A., & Barabasi, A. L. (2008). Understanding individual human mobility patterns. Nature, 453(7196), 779-782.

    """
    if print_progress:
        tqdm.pandas(desc="User radius of gyration calculation")
        s = sp.groupby("user_id").progress_apply(lambda x: _radius_gyration_user(x, method))
    else:
        s = sp.groupby("user_id").apply(lambda x: _radius_gyration_user(x, method))

    s.rename("radiusGyration", inplace=True)
    return s


def jump_length(sp):
    """
    Jump length between consecutive locations.

    Parameters
    ----------
    sp : Geodataframe
        Staypoints with geometry in latitude and longitude.

    Returns
    -------
    np.array
        Array containing the jump lengths.

    References
    ----------
    [1] Brockmann, D., Hufnagel, L., & Geisel, T. (2006). The scaling laws of human travel. Nature, 439(7075), 462-465.

    """
    pts = sp.geometry.values
    return np.array([haversine_dist(pts[i - 1].x, pts[i - 1].y, pts[i].x, pts[i].y)[0] for i in range(1, len(pts))])


def wait_time(df):
    """
    Wait time consecutive locations.

    Parameters
    ----------
    sp : DataFrame
        Staypoints with time information, either provided in "duration" column, or in "finished_at" and "started_at" columns.

    Returns
    -------
    np.array
        Array containing the wait time.

    References
    ----------
    [1] Brockmann, D., Hufnagel, L., & Geisel, T. (2006). The scaling laws of human travel. Nature, 439(7075), 462-465.

    """
    if "duration" in df.columns:
        return df["duration"].values
    else:
        # TODO: check
        return ((df["finished_at"] - df["started_at"]).dt.total_seconds() / 3600).values


def location_frquency(sp):
    """Location visit frquency for datasets

    Parameters
    ----------
    sp : Geodataframe
        Staypoints with column "location_id" and "user_id".

    Returns
    -------
    s: list
        the ranked visit frquency.

    References
    ----------
    [1] Gonzalez, M. C., Hidalgo, C. A., & Barabasi, A. L. (2008). Understanding individual human mobility patterns. Nature, 453(7196), 779-782.

    """

    # get visit times per user and location
    freq = sp.groupby(["user_id", "location_id"], as_index=False).size()
    # get the rank of locations per user
    freq["visitRank"] = freq.groupby("user_id")["size"].rank(ascending=False, method="first")
    # get the average visit freqency for every rank
    pLoc = freq.groupby("visitRank")["size"].mean().values

    # normalize
    pLoc = pLoc / pLoc.sum()

    return pLoc


def _radius_gyration_user(sp_user, method):
    """
    User level radius of gyration calculation, see radius_gyration() for details.

    Parameters
    ----------
    sp_user : Geodataframe
        The staypoints from an individual, should contain "geometry".

    method: string, {"duration", "count"}, default "count"
        method to calculate rg. Duration additionally weights each sp with the activity duration.
    Returns
    -------
    float
        the radius of gyration of the individual
    """
    sp_user["lat"] = sp_user["geometry"].y
    sp_user["lng"] = sp_user["geometry"].x
    lats_lngs = sp_user[["lat", "lng"]].values

    if method == "duration":
        durs = sp_user[["duration"]].values

        center_of_mass = np.sum([lat_lng * dur for lat_lng, dur in zip(lats_lngs, durs)], axis=0) / np.sum(durs)
        inside = [
            dur * (haversine_dist(lng, lat, center_of_mass[-1], center_of_mass[0]) ** 2.0)
            for (lat, lng), dur in zip(lats_lngs, durs)
        ]

        rg = np.sqrt(np.sum(inside) / sp_user["duration"].sum())
    elif method == "count":
        center_of_mass = np.mean(lats_lngs, axis=0)
        rg = np.sqrt(
            np.mean([haversine_dist(lng, lat, center_of_mass[-1], center_of_mass[0]) ** 2.0 for lat, lng in lats_lngs])
        )
    else:
        raise AttributeError(
            f"Method unknown. Please check the input arguement. We only support 'duration', 'count'. You passed {method}"
        )
    return rg


def mobility_motifs(sp, proportion_filter=0.005):
    """
    Get the mobility motifs for a input dataset (sp).

    Mobility motifs are defined as unqiue patterns of location visits per user day. Thus, location visits of each user are binned per day, and cross-compared to find valid mobility motifs. A mobility motif shall fulfil the following requirements:
    - Contains all locations visited by a user in a day.
    - Frequently occuring in the dataset (controlled by the parameter `proportion_filter`).

    Parameters
    ----------
    sp : Geodataframe
        Staypoints with user and time information ("user_id", "started_at", "finished_at"), and "location_id".

    proportion_filter: boolen, default 0.005
        Filter to control how frequent a pattern coulf be considered a motifs, e.g., 0.005 means patterns occuring more than 0.5% of all the patterns are considered motifs.

    Returns
    -------
    pandas DataFrame
        User day dataframe containing the motifs information with columns "visits", "uniq_visits", and "class". "visits" and "uniq_visits" represent the number of locations and number of unique location visits during the day, repectively. "class" is the unique type of motifs of the day. "uniq_visits" and "class" together uniquely define a motif. Non motif days receive NaN value.

    References
    ----------
    [1] Schneider, C. M., Belik, V., Couronné, T., Smoreda, Z., & González, M. C. (2013). Unravelling daily human mobility motifs. Journal of The Royal Society Interface, 10(84), 20130246.

    """
    # split the records based on day, such that daily motifs can be constructed
    sp = _split_overlaps(sp, granularity="day")
    sp["date"] = sp["started_at"].dt.date

    # delete the self transitions within the same day (no required for generated sequences)
    sp["loc_next"] = sp["location_id"].shift(-1)
    sp["date_next"] = sp["date"].shift(-1)

    sp = sp.loc[~((sp["loc_next"] == sp["location_id"]) & (sp["date_next"] == sp["date"]))].copy()
    sp.drop(columns=["loc_next", "date_next"], inplace=True)

    # count unique daily location visits, and merge back to sp
    user_date_loc_count = sp.groupby(["user_id", "date"]).agg({"location_id": "nunique"})
    user_date_loc_count.rename(columns={"location_id": "uniq_visits"}, inplace=True)
    sp = sp.merge(user_date_loc_count.reset_index(), on=["user_id", "date"], how="left")

    # construct possible graphs
    user_day_df = _get_user_day_graph(sp)

    # get total number of graphs for filtering
    total_graphs = len(user_day_df)

    def _get_valid_motifs(df):
        if (len(df) / total_graphs) > proportion_filter:
            return df

    # get the valid motifs per user days
    motifs_user_days = (
        user_day_df.groupby(["uniq_visits", "class"], as_index=False).apply(_get_valid_motifs).reset_index(drop=True)
    )

    # merge back to all user days
    return_df = (
        sp.groupby(["user_id", "date"])
        .size()
        .rename("visits")
        .reset_index()
        .merge(motifs_user_days, on=["user_id", "date"], how="left")
    )

    return return_df


def _get_user_day_graph(sp):
    """
    Construct network patterns from user daily location visits. The return of the function can be used to filter for motifs.

    Parameters
    ----------
    sp : Geodataframe
        Staypoints with columns "user_id", "date", "uniq_visits" and "location_id".

    Returns
    -------
    pandas DataFrame
        User day dataframe containing the network pattern information with columns "uniq_visits", "class". "uniq_visits" represents the number of unique location visits during the day. "class" is the unique type of network pattern of the day. "uniq_visits" and "class" together uniquely define a pattern. Non pattern days receive NaN value.

    References
    ----------
    [1] Schneider, C. M., Belik, V., Couronné, T., Smoreda, Z., & González, M. C. (2013). Unravelling daily human mobility motifs. Journal of The Royal Society Interface, 10(84), 20130246.

    """

    user_day_ls = []

    # consider up to 6 location visits per day
    for uniq_visits in tqdm(range(1, 7)):
        curr_sp = sp.loc[sp["uniq_visits"] == uniq_visits].copy()
        curr_sp["next_loc"] = curr_sp["location_id"].shift(-1)

        # for only 1 location visit, every day is the same motif
        if uniq_visits == 1:
            graph_s = curr_sp.groupby(["user_id", "date"]).size().rename("class").reset_index()
            graph_s["class"] = 0
            graph_s["uniq_visits"] = uniq_visits
            user_day_ls.append(graph_s)
            continue

        # the edge number shall be at least the node number, otherwise not a motif
        edge_num = curr_sp.groupby(["user_id", "date"]).size() - 1
        valid_user_dates = edge_num[edge_num >= uniq_visits].rename("edge_num")
        curr_sp = curr_sp.merge(valid_user_dates.reset_index(), on=["user_id", "date"], how="left")
        curr_sp = curr_sp.loc[~curr_sp["edge_num"].isna()]

        # for 2 location visits, every day that have the edge number larger than the node number is the same motif
        if uniq_visits == 2:
            graph_s = curr_sp.groupby(["user_id", "date"]).size().rename("class").reset_index()
            graph_s["class"] = 0
            graph_s["uniq_visits"] = uniq_visits
            user_day_ls.append(graph_s)
            continue

        graph_s = curr_sp.groupby(["user_id", "date"]).apply(_construct_day_graph)
        # valid motifs shall be connected: each node shall have in and our degree
        # filter graphs that do not have an in-degree and out degree
        graph_s = graph_s.loc[~graph_s.isna()]

        # check for motif groups
        unique_motif_group_ls = []
        for i in range(len(graph_s) - 1):
            # if i has already been check, we do not need to check again
            if i in [item for sublist in unique_motif_group_ls for item in sublist]:
                continue

            # check for repeated patterns as i in [i+1, max)
            possible_match_ls = [i]
            for j in range(i + 1, len(graph_s)):
                if isomorphism.GraphMatcher(graph_s.iloc[i], graph_s.iloc[j]).is_isomorphic():
                    possible_match_ls.append(j)

            # append this group of motif
            unique_motif_group_ls.append(possible_match_ls)

        # label motif class and assign back to graph_s
        graph_s = graph_s.rename("graphs").reset_index()
        class_arr = np.zeros(len(graph_s))
        for i, classes in enumerate(unique_motif_group_ls):
            class_arr[classes] = i
        graph_s["class"] = class_arr
        graph_s["class"] = graph_s["class"].astype(int)
        graph_s["uniq_visits"] = uniq_visits
        graph_s.drop(columns={"graphs"}, inplace=True)

        user_day_ls.append(graph_s)

    return pd.concat(user_day_ls)


def _construct_day_graph(df):
    """
    Construct networks from daily location visits. Shall be used after groupby ["user_id", "date"].

    Parameters
    ----------
    df : Geodataframe
        Staypoints with columns "location_id" and "next_loc".

    Returns
    -------
    networkx DiGraph
        Graph object constructed from location visits.

    """
    G = nx.DiGraph()
    G.add_nodes_from(df["location_id"])

    G.add_edges_from(df.iloc[:-1][["location_id", "next_loc"]].astype(int).values)

    in_degree = np.all([False if degree == 0 else True for _, degree in G.in_degree])
    out_degree = np.all([False if degree == 0 else True for _, degree in G.out_degree])
    # TODO: check the requirement in the original paper
    if in_degree and out_degree:
        return G


# TODO: check how to reuse the function in utils.utils
def load_data(sp, loc):
    sp = sp.merge(loc.reset_index().drop(columns={"user_id"}), how="left", left_on="location_id", right_on="id")
    sp = sp.drop(columns={"location_id", "id", "center", "extent"})
    sp = sp.rename(columns={"s2_id": "location_id"})

    sp.index.name = "id"
    sp.reset_index(inplace=True)

    sp["started_at"] = pd.to_datetime(sp["started_at"], format="mixed", yearfirst=True, utc=True).dt.tz_localize(None)
    sp["finished_at"] = pd.to_datetime(sp["finished_at"], format="mixed", yearfirst=True, utc=True).dt.tz_localize(None)

    def _get_time_info(df):
        min_day = pd.to_datetime(df["started_at"].min().date())

        # get the alighned time with act_duration
        df["temp_time"] = pd.NA
        df["temp_time"] = df["finished_at"].shift(1)
        df.loc[df.index[0], "temp_time"] = df["started_at"].iloc[0]

        df["start_day"] = (df["temp_time"] - min_day).dt.days
        df["start_min"] = df["temp_time"].dt.hour * 60 + df["temp_time"].dt.minute
        df["weekday"] = df["temp_time"].dt.weekday

        df = df.drop(columns="temp_time")
        # df["duration"] = (df["duration"] * 60).round()
        return df

    sp = sp.groupby("user_id", group_keys=False).apply(_get_time_info)
    return sp
