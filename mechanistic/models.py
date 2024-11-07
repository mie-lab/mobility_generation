import pandas as pd
import geopandas as gpd


import datetime
import operator
from collections import defaultdict, Counter
import random
import inspect

import numpy as np
from tqdm import tqdm

from sklearn.metrics import pairwise_distances
import shapely


def calculate_distance_matrix(X, Y=None, dist_metric="euclidean", n_jobs=None, **kwds):
    X = shapely.get_coordinates(X.geometry)
    Y = shapely.get_coordinates(Y.geometry) if Y is not None else X
    return pairwise_distances(X, Y, metric=dist_metric, n_jobs=n_jobs, **kwds)


def get_wait_time(parameters):
    """Wait time (duration) distribution. Emperically determined from data."""
    return np.random.lognormal(parameters["1"], parameters["2"], 1)[0]


def get_jump_length(parameters):
    """Jump length distribution. Emperically determined from data."""
    return np.random.lognormal(parameters["1"], parameters["2"], 1)[0]


class EPR:
    """Explore and preferential return model"""

    def __init__(self, all_locs, param_jump_length, param_wait_time):
        self.param_jump_length = param_jump_length
        self.param_wait_time = param_wait_time

        # precalculate distance between location pairs
        self.pair_distance = calculate_distance_matrix(all_locs, n_jobs=-1, dist_metric="euclidean")

    def simulate(self, train_seq, simulation_param, length=20):
        # get the two exploration parameter
        rho = simulation_param["p"]
        gamma = simulation_param["r"]

        train_seq = train_seq.tolist()
        dur_ls = []

        # the generation process
        for _ in range(length):
            # get wait time from distribution
            current_duration = get_wait_time(self.param_wait_time)
            # while current_duration > 24 * 2:
            #     current_duration = get_wait_time(self.param_wait_time)
            dur_ls.append(current_duration)

            next_loc = self.simulate_step(train_seq, rho, gamma)

            train_seq.append(next_loc)

        return train_seq[-length:], dur_ls

    def simulate_step(self, loc_ls, rho, gamma):
        # the prob. of exploring
        if_explore = rho * len(np.unique(loc_ls)) ** (-gamma)

        if (np.random.rand() < if_explore) or (len(loc_ls) == 1):
            # explore
            next_loc = self.explore(visited_loc=loc_ls)
        else:
            next_loc = self.pref_return(visited_loc=loc_ls)

        return next_loc

    def explore(self, visited_loc):
        """The exploration step of the epr model.

        1. get a jump distance from predefined jump length distribution.
        2. calculate the distance between current location and all the other locations.
        3. choose the location that is closest to the jump length.

        Parameters
        ----------
        curr_loc: the current location that the user is standing
        all_loc: df containing the info of all locations

        Returns
        -------
        the id of the selected location
        """
        curr_loc = int(visited_loc[-1])
        # the distance to be jumped
        jump_distance = get_jump_length(self.param_jump_length)

        # select from the 5 closest location after the jump, - 1 and + 1 for padding
        loc_dist_to_jump = np.abs(self.pair_distance[curr_loc - 1, :] - jump_distance)
        candidate_locs = np.argsort(loc_dist_to_jump)[:5] + 1

        return np.random.choice(candidate_locs)

    def pref_return(self, visited_loc):
        # not able to return to the current location
        visited_loc = np.array(visited_loc)
        curr_loc = visited_loc[-1]

        # delete the current location from the sequence
        currloc_idx = np.where(visited_loc == curr_loc)[0]
        # ensure the deleted sequence contain value
        if len(currloc_idx) != len(visited_loc):
            visited_loc = np.delete(visited_loc, currloc_idx)

        # choose next location according to emperical visits
        next_loc = np.random.choice(visited_loc)

        return next_loc


class DITRAS(EPR):

    def __init__(self, diary_generator, all_locs, param_jump_length, param_wait_time, relevance):
        self.diary_generator = diary_generator
        self.param_jump_length = param_jump_length
        self.param_wait_time = param_wait_time

        relevance[relevance == 0] = 0.1
        self.attr = relevance

        self.pair_distance = calculate_distance_matrix(all_locs, n_jobs=-1, dist_metric="euclidean")

    def simulate(self, train_seq, simulation_param, start_time, length=20):

        # get the two exploration parameter
        rho = simulation_param["p"]
        gamma = simulation_param["r"]

        train_seq = train_seq.tolist()
        most_frequent = max(set(train_seq), key=train_seq.count)

        diary_df = self.diary_generator.run(length, start_time)

        for i, row in diary_df.iterrows():
            if row.abstract_location == 0:  # the agent is at home
                train_seq.append(most_frequent)

            else:  # the agent is not at home
                next_loc = self.simulate_step(train_seq, rho, gamma)
                train_seq.append(next_loc)

        # loc and time

        return train_seq[-len(diary_df) :], diary_df.datetime

    def simulate_step(self, loc_ls, rho, gamma):
        # the prob. of exploring
        if_explore = rho * len(np.unique(loc_ls)) ** (-gamma)

        if (np.random.rand() < if_explore) or (len(loc_ls) == 1):
            # explore
            next_loc = self.explore(visited_loc=loc_ls)
        else:
            next_loc = self.pref_return(visited_loc=loc_ls)

        return next_loc

    def explore(self, visited_loc):
        curr_loc = int(visited_loc[-1])
        curr_attr = self.attr[curr_loc - 1]

        # delete the already visited locations
        uniq_visited_loc = set(np.array(visited_loc) - 1)
        remain_loc = np.arange(len(self.attr)) + 1
        remain_loc = np.delete(remain_loc, list(uniq_visited_loc))

        # slight modification, original **2, and we changed to 1.7
        r = (self.pair_distance[curr_loc - 1, remain_loc - 1] / 1000) ** (1.7)
        # we also take the square root to reduce the density effect, otherwise too strong
        attr = np.power((self.attr[remain_loc - 1] * curr_attr), 0.5).astype(float)

        # the density attraction + inverse distance
        attr = np.divide(attr, r, out=np.zeros_like(attr), where=r != 0)

        # norm
        attr = attr / attr.sum()
        selected_loc = np.random.choice(len(attr), p=attr)
        return remain_loc[selected_loc]

    def pref_return(self, visited_loc):
        # not able to return to the current location
        visited_loc = np.array(visited_loc)
        curr_loc = visited_loc[-1]

        # delete the current location from the sequence
        currloc_idx = np.where(visited_loc == curr_loc)[0]
        # ensure the deleted sequence contain value
        if len(currloc_idx) != len(visited_loc):
            visited_loc = np.delete(visited_loc, currloc_idx)

        # choose next location according to emperical visits
        next_loc = np.random.choice(visited_loc)

        return next_loc


class MarkovDiaryGenerator:
    """Markov Diary Learner and Generator.

    A diary generator :math:`G` produces a mobility diary, :math:`D(t)`, containing the sequence of trips made by an agent during a time period divided in time slots of :math:`t` seconds. For example, :math:`G(3600)` and :math:`G(60)` produce mobility diaries with temporal resolutions of one hour and one minute, respectively [PS2018]_.

    A Mobility Diary Learner (MDL) is a data-driven algorithm to compute a mobility diary :math:`MD` from the mobility trajectories of a set of real individuals. We use a Markov model to describe the probability that an individual follows her routine and visits a typical location at the usual time, or she breaks the routine and visits another location. First, MDL translates mobility trajectory data of real individuals into abstract mobility trajectories. Second, it uses the obtained abstract trajectory data to compute the transition probabilities of the Markov model :math:`MD(t)` [PS2018]_.

    Parameters
    ----------
    name : str, optional
        name of the instantiation of the class. The default is "Markov diary".

    Attributes
    ----------
    name : str
        name of the instantiation of the class.

    markov_chain_ : dict
        the trained markov chain.

    time_slot_length : str
        length of the time slot (1h).

    Examples
    --------
    >>> import skmob
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> from skmob.models.epr import Ditras
    >>> from skmob.models.markov_diary_generator import MarkovDiaryGenerator
    >>> from skmob.preprocessing import filtering, compression, detection, clustering
    >>> url = skmob.utils.constants.GEOLIFE_SAMPLE
    >>>
    >>> df = pd.read_csv(url, sep=',', compression='gzip')
    >>> tdf = skmob.TrajDataFrame(df, latitude='lat', longitude='lon', user_id='user', datetime='datetime')
    >>>
    >>> ctdf = compression.compress(tdf)
    >>> stdf = detection.stops(ctdf)
    >>> cstdf = clustering.cluster(stdf)
    >>>
    >>> mdg = MarkovDiaryGenerator()
    >>> mdg.fit(cstdf, 2, lid='cluster')
    >>>
    >>> start_time = pd.to_datetime('2019/01/01 08:00:00')
    >>> diary = mdg.generate(100, start_time)
    >>> print(diary)
                 datetime  abstract_location
    0 2019-01-01 08:00:00                  0
    1 2019-01-02 19:00:00                  1
    2 2019-01-02 20:00:00                  0
    3 2019-01-03 17:00:00                  1
    4 2019-01-03 18:00:00                  2
    5 2019-01-04 08:00:00                  0
    6 2019-01-05 03:00:00                  1

    References
    ----------
    .. [PS2018] Pappalardo, L. & Simini, F. (2018) Data-driven generation of spatio-temporal routines in human mobility. Data Mining and Knowledge Discovery 32, 787-829, https://link.springer.com/article/10.1007/s10618-017-0548-4

    See Also
    --------
    Ditras
    """

    def __init__(self, name="Markov diary"):
        self._markov_chain_ = None
        self._time_slot_length = "1h"
        self._name = name

    @property
    def markov_chain_(self):
        return self._markov_chain_

    @property
    def time_slot_length(self):
        return self._time_slot_length

    @property
    def name(self):
        return self._name

    def _create_empty_markov_chain(self):
        """
        Create an empty Markov chain, i.e., a matrix 48 * 48 where an element M(i,j) is a pair of pairs
        ((h_i, b_i), (h_j, b_j)), h_i, h_j in {0, ..., 23} and b_i, b_j in {0, 1}
        """
        self._markov_chain_ = defaultdict(lambda: defaultdict(float))
        for h1 in range(0, 24):
            for r1 in [0, 1]:
                for h2 in range(0, 24):
                    for r2 in [0, 1]:
                        self._markov_chain_[(h1, r1)][(h2, r2)] = 0.0

    @staticmethod
    def _select_loc(individual_df, location2frequency, location_column="location"):

        if isinstance(individual_df[location_column], str):  # if there is at least a location in the time slot
            locations = individual_df[location_column].split(",")

            if len(locations) == 1:  # if there just one location, then assign that location to the time slot
                return locations[0]

            elif len(set(Counter(locations).values())) == 1:  # if there are multiple locations, with the same frequency
                return sorted(
                    {k: location2frequency[k] for k in locations}.items(), key=operator.itemgetter(1), reverse=True
                )[0][
                    0
                ]  # return the locations with the highest overall frequency

            else:  # if there multiple location in the same slot, with different frequencies
                sorted_l = sorted(Counter(locations).items(), key=operator.itemgetter(1), reverse=True)
                return sorted_l[0][0]

        # if there is no location in the time slot, return NaN
        return np.nan

    @staticmethod
    def _weighted_random_selection(weights):
        return np.searchsorted(np.cumsum(weights), random.random())

    @staticmethod
    def _get_location2frequency(traj, location_column="location_id"):
        """
        Compute the visitation frequency and rank of each location of an individual

        Parameters
        ----------
        traj : pandas DataFrame
            the trajectories of the individuals.

        location_column : str, optional
            the name of the column containing the location identifier. The default is "location".

        Returns
        -------
        tuple
            a tuple of two dictionaries of locations to the corresponding visitation frequency and rank, respectively.
        """
        location2frequency, location2rank = defaultdict(int), defaultdict(int)
        for i, row in traj.iterrows():
            if isinstance(row[location_column], str):  # if it is not NaN
                for location in row[location_column].split(","):
                    # we can have more than one location in a time slot
                    # so, every slot has a comma separated list of locations
                    location2frequency[location] += 1

        # compute location2rank
        rank = 1
        for loc in sorted(location2frequency.items(), key=operator.itemgetter(1), reverse=True):
            location, frequency = loc
            location2rank[location] = rank
            rank += 1
        return location2frequency, location2rank

    def _create_time_series(self, df):  # start_date, end_date, lid='location'):
        """
        Returns
        -------
        pandas DataFrame
            the time series of the abstract locations visited by the individual.
        """

        shift = df["started_at"].min().hour
        traj = df[["started_at", "location_id"]].set_index("started_at")
        traj["location_id"] = traj["location_id"].astype("str")
        # enlarge (eventually) the time series with the specified freq (replace empty time slots with NaN)
        traj = (
            traj.groupby(pd.Grouper(freq=self._time_slot_length, closed="left"))
            .aggregate(lambda x: ",".join(x))
            .replace("", np.nan)
        )

        # compute the frequency of every location visited by the individual
        location2frequency, location2rank = self._get_location2frequency(traj, location_column="location_id")

        # select the location for every slot
        # ix = pd.DatetimeIndex(start=start_date, end=end_date, freq=self._time_slot_length)
        # ix = pd.date_range(start=start_date, end=end_date, freq=self._time_slot_length)
        time_series = traj.apply(
            lambda x: self._select_loc(x, location2frequency, location_column="location_id"), axis=1
        )

        # fill the slots with NaN with the previous element or the next element ###
        time_series.ffill(inplace=True)
        time_series.bfill(inplace=True)

        # you can use location2frequency to assign a number to every location
        time_series = time_series.apply(lambda x: location2rank[x])

        self._update_markov_chain(time_series, shift=shift)

    def _update_markov_chain(self, time_series, shift=0):
        """
        Update the Markov Chain by including the behavior of an individual

        Parameters
        ----------
        time_series: pandas DataFrame
            time series of abstract locations visisted by an individual.
        """
        HOME = 1
        TYPICAL, NON_TYPICAL = 1, 0

        n = len(time_series)  # n is the length of the time series of the individual
        slot = 0  # it starts from the first slot in the time series

        while slot < n - 1:  # scan the time series of the individual, time slot by time slot

            # h = (slot % 24)
            h = (slot + shift) % 24  # h, the hour of the day
            next_h = (h + 1) % 24  # next_h, the next hour of the day

            loc_h = time_series.iloc[slot]  # loc_h  ,   abstract location at the current slot
            next_loc_h = time_series.iloc[slot + 1]  # d_{h+1},   abstract location at the next slot

            if loc_h == HOME:  # if \delta(loc_h, t_h) == 1, i.e., she stays at home

                # we have two cases
                if next_loc_h == HOME:  # if \delta(d_{h + 1}, t_{h + 1}) == 1

                    # we are in Type1: (h, 1) --> (h + 1, 1)
                    self._markov_chain_[(h, TYPICAL)][(next_h, TYPICAL)] += 1

                else:  # she will be not in the typical location

                    # we are in Type2: (h, 1) --> (h + tau, 0)
                    tau = 1
                    if slot + 2 < n:  # if slot is the second last in the time series

                        for j in range(slot + 2, n):  # in slot + 1 we do not have HOME so we start from slot + 2
                            loc_hh = time_series.iloc[j]
                            if loc_hh == next_loc_h:  # if \delta(d_{h + j}, d_{h + 1}) == 1
                                tau += 1
                            else:
                                break

                        h_tau = (h + tau) % 24
                        # update the state of edge (h, 1) --> (h + tau, 0)
                        self._markov_chain_[(h, TYPICAL)][(h_tau, NON_TYPICAL)] += 1
                        slot = j - 2  # 1

                    else:  # terminate the while cycle
                        slot = n

            else:  # loc_h != HOME

                if next_loc_h == HOME:  # if \delta(d_{h + 1}, t_{h + 1}) == 1, i.e., she will stay at home

                    # we are in Type3: (h, 0) --> (h + 1, 1)
                    self._markov_chain_[(h, NON_TYPICAL)][(next_h, TYPICAL)] += 1

                else:

                    # we are in Type 4: (h, 0) --> (h + tau, 0)
                    tau = 1
                    if slot + 2 < n:

                        for j in range(slot + 2, n):
                            loc_hh = time_series.iloc[j]
                            if loc_hh == next_loc_h:  # if \delta(d_{h + j}, d_{h + 1}) == 1
                                tau += 1
                            else:
                                break

                        h_tau = (h + tau) % 24

                        # update the state of edge (h, 0) --> (h + tau, 0)
                        self._markov_chain_[(h, NON_TYPICAL)][(h_tau, NON_TYPICAL)] += 1
                        slot = j - 2  # 1

                    else:
                        slot = n

            slot += 1

    def _normalize_markov_chain(self):
        """
        Transform the dictionary into a proper Markov chain, i.e., normalize by row in order
        to obtain transition probabilities.
        """
        # compute the probabilities of the Markov chain, i.e. normalize by row
        for state1 in self._markov_chain_:
            tot = sum([prob for prob in self._markov_chain_[state1].values()])
            for state2 in self._markov_chain_[state1]:
                if tot != 0.0:
                    self._markov_chain_[state1][state2] /= tot

    def fit(self, sps):
        """
        Train the markov mobility diary from real trajectories.

        Parameters
        ----------
        traj : TrajDataFrame
            the trajectories of the individuals.

        lid : string, optional
            the name of the column containing the identifier of the location. The default is "location".
        """

        self._create_empty_markov_chain()  # initialize the markov chain

        tqdm.pandas(desc="create time series")
        sps.groupby("user_id").progress_apply(self._create_time_series)

        # normalize the markov chain, i.e., normalize the transitions by row
        self._normalize_markov_chain()

    def run(self, diary_length, start_date):
        """
        Start the generation of the mobility diary.

        Parameters
        ----------
        diary_length : int
            the length of the diary in hours.

        start_date : datetime
            the starting date of the generation.

        Returns
        -------
        pandas DataFrame
            the generated mobility diary.
        """
        current_date = start_date
        V = []
        prev_state = (current_date.hour, 1)  # it starts from the typical location at midnight
        V.append(prev_state)

        i = current_date.hour
        while i < diary_length:

            h = i % 24  # the hour of the day

            # select the next state in the Markov chain
            p = list(self._markov_chain_[prev_state].values())
            if sum(p) == 0.0:
                hh, rr = prev_state
                next_state = ((hh + 1) % 24, rr)
            else:
                index = self._weighted_random_selection(p)
                next_state = list(self._markov_chain_[prev_state].keys())[index]
            V.append(next_state)

            j = next_state[0]
            if j > h:  # we are in the same day
                i += j - h
            else:  # we are in the next day
                i += 24 - h + j

            prev_state = next_state

        # now we translate the temporal diary into the the mobility diary
        prev, diary, other_count = V[0], [], 1
        diary.append([current_date, 0])

        for v in V[1:]:  # scan all the states obtained and create the synthetic time series
            h, s = v
            h_prev, s_prev = prev

            if s == 1:  # if in that hour she visits home
                current_date += datetime.timedelta(hours=1)
                diary.append([current_date, 0])
                other_count = 1
            else:  # if in that hour she does NOT visit home

                if h > h_prev:  # we are in the same day
                    j = h - h_prev
                else:  # we are in the next day
                    j = 24 - h_prev + h

                for i in range(0, j):
                    current_date += datetime.timedelta(hours=1)
                    diary.append([current_date, other_count])
                other_count += 1

            prev = v

        short_diary = []
        prev_location = -1
        for visit_date, abstract_location in diary[0:diary_length]:
            if abstract_location != prev_location:
                short_diary.append([visit_date, abstract_location])
            prev_location = abstract_location

        diary_df = pd.DataFrame(short_diary, columns=["datetime", "abstract_location"])
        return diary_df
