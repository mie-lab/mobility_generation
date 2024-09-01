import pandas as pd
import geopandas as gpd
import math

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
