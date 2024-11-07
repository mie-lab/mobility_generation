import pandas as pd
import numpy as np
import math

import datetime
from tqdm import tqdm


import shapely
from sklearn.metrics import pairwise_distances

"""
TimeGeo Model
"""


def calculate_distance_matrix(X, Y=None, dist_metric="euclidean", n_jobs=None, **kwds):
    X = shapely.get_coordinates(X.geometry)
    Y = shapely.get_coordinates(Y.geometry) if Y is not None else X
    return pairwise_distances(X, Y, metric=dist_metric, n_jobs=n_jobs, **kwds)


class Time_geo(object):

    def __init__(
        self,
        p_t,
        all_locs,
        alpha=1.86,
        n_w=6.1,
        beta1=3.67,
        beta2=10,
    ):

        super().__init__()
        self.alpha = alpha  # it controls the exploration depth
        self.n_w = n_w  # it is the average number of tour based on home a week.
        self.beta1 = beta1  # dwell rate
        self.beta2 = beta2  # burst rate

        self.pair_distance = calculate_distance_matrix(all_locs, n_jobs=-1, dist_metric="euclidean")

        self.p_t = p_t

    def predict_next_place_time(self, p_t_value, current_location_type):
        p1 = 1 - self.n_w * p_t_value
        p2 = 1 - self.beta1 * self.n_w * p_t_value
        p3 = self.beta2 * self.n_w * p_t_value

        location_changed = False
        if current_location_type == 0:  # at home
            if np.random.rand() <= p1:
                new_location_type = 0
                location_changed = False
            else:
                new_location_type = 1
                location_changed = True
        elif current_location_type == 1:  # not at home
            if np.random.rand() <= p2:
                new_location_type = 1
                location_changed = False
            elif np.random.rand() <= p3:
                new_location_type = 1
                location_changed = True
            else:
                new_location_type = 0
                location_changed = True

        return new_location_type, location_changed

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

        # delete the already visited locations
        uniq_visited_loc = set(np.array(visited_loc) - 1)
        remain_loc = np.arange(len(self.pair_distance)) + 1
        remain_loc = np.delete(remain_loc, list(uniq_visited_loc))

        # slight modification, original **2, and we changed to 1.7
        r = (self.pair_distance[curr_loc - 1, remain_loc - 1] / 1000) ** (1.7)
        rank = r.argsort().argsort() + 1
        rank = 1 / rank

        return np.random.choice(len(rank), p=rank / rank.sum()) + 1

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

    def simulate(self, train_seq, simulation_param, length, start_time):
        # get the two exploration parameter
        rho = simulation_param["p"]
        gamma = simulation_param["r"]

        train_seq = train_seq.tolist()
        most_frequent = max(set(train_seq), key=train_seq.count)

        current_location_type = 0
        diary = pd.DataFrame([start_time + datetime.timedelta(hours=i) for i in range(length)], columns=["time"])

        for i, row in diary.iterrows():
            now_time = row.dt.hour.values[0]
            p_t_value = self.p_t[now_time]
            now_type, location_change = self.predict_next_place_time(p_t_value, current_location_type)

            if location_change:
                if now_type == 0:
                    next_location = most_frequent
                else:
                    next_location = self.simulate_step(train_seq, rho, gamma)
            else:
                next_location = train_seq[-1]

            train_seq.append(next_location)

        diary["loc"] = train_seq[-length:]

        diary = diary.loc[diary["loc"].shift() != diary["loc"]].reset_index(drop=True)

        return diary
