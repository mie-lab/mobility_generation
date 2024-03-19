# encoding: utf-8

import os
import shutil
import argparse
import scipy.stats
import numpy as np
from scipy.spatial import distance

import torch

from math import radians, cos, sin, asin, sqrt

# from utils import get_gps, read_data_from_file, read_logs_from_file


class Metric(object):
    def __init__(self, config, locations, input_data, valid_start_end_idx):
        self.config = config
        self.max_locs = config.total_loc_num - 1  # for padding
        self.max_distance = 400

        self.geo_x = locations.sort_values(by="loc_id")["geometry"].x.values
        self.geo_y = locations.sort_values(by="loc_id")["geometry"].y.values

        self.reference_data = self.extract_reference_dataset(input_data, valid_start_end_idx)

        # metrics to calculate
        self.ref_dist_p = None
        self.ref_rg_p = None
        self.ref_period_p = None
        self.ref_overall_topk_freq_p = None
        self.ref_topk_freq_p = None

        self.get_reference_metrics()

    def extract_reference_dataset(self, input_data, valid_start_end_idx):
        input = input_data.copy()
        input.set_index("id", inplace=True)
        return [input.iloc[idx[0] : idx[1]]["location_id"].values for idx in valid_start_end_idx]

    def get_reference_metrics(self):
        ref_dist = self.get_distances(self.reference_data) / self.max_distance
        self.ref_dist_p, _ = np.histogram(ref_dist, bins=np.arange(0, 1, 1 / 1000))

        ref_rg = self.get_rg(self.reference_data) / self.max_distance
        self.ref_rg_p, _ = np.histogram(ref_rg, bins=np.arange(0, 1, 1 / 1000))

        ref_period = self.get_periodicity(self.reference_data)
        self.ref_period_p, _ = np.histogram(ref_period, bins=np.arange(0, 1, 1 / 1000))

        ref_overall_topk_freq = self.get_overall_topk_freq(self.reference_data, K=20)
        self.ref_overall_topk_freq_p, _ = np.histogram(ref_overall_topk_freq, bins=np.arange(0, 1, 1 / 1000))

        ref_topk_freq = self.get_topk_freq(self.reference_data, K=10)
        self.ref_topk_freq_p, _ = np.histogram(ref_topk_freq, bins=np.arange(0, 1, 1 / 1000))

    def get_distances(self, trajs):
        dists = []

        for traj in trajs:
            traj = traj - 1  # for padding
            # to km
            xs = np.take(self.geo_x, traj) / 1000
            ys = np.take(self.geo_y, traj) / 1000

            # jump length
            square_jump = (xs[1:] - xs[:-1]) ** 2 + (ys[1:] - ys[:-1]) ** 2
            dists.append(np.sqrt(square_jump))
        return np.concatenate(dists, axis=0)

    def get_durations(self, trajs):
        pass

    def get_rg(self, trajs):
        """
        get the std of the distances of all points away from center as `gyration radius`
        :param trajs:
        :return:
        """
        rgs = []
        for traj in trajs:
            traj = traj - 1  # for padding

            # to km
            xs = np.take(self.geo_x, traj) / 1000
            ys = np.take(self.geo_y, traj) / 1000

            x_center = np.average(xs)
            y_center = np.average(ys)

            square_rg = np.average((xs - x_center) ** 2 + (ys - y_center) ** 2)

            rgs.append(np.sqrt(square_rg))
        return np.array(rgs, dtype=float)

    def get_periodicity(self, trajs):
        """
        stat how many repetitions within a single trajectory
        :param trajs:
        :return:
        """
        return np.array([len(set(traj)) / len(traj) for traj in trajs], dtype=float)

    def get_overall_topk_freq(self, trajs, K=100):
        """
        get probability distribution of visiting all locations
        :param trajs:
        :return:
        """
        visits = np.zeros(shape=(self.max_locs), dtype=float)
        for traj in trajs:
            traj = traj - 1  # for padding
            visits[traj] += 1

        # norm
        visits = visits / np.sum(visits)

        # loc and freq
        topk_locs = visits.argsort()[K:][::-1]
        topk_probs = visits[topk_locs]
        return topk_probs

    def get_topk_freq(self, trajs, K=20):
        topk_prob = []
        for traj in trajs:
            traj = traj - 1  # for padding

            _, counts = np.unique(traj, return_counts=True)
            counts.sort()
            counts = counts[::-1]
            prob = counts / counts.sum()

            # accounting for insufficient location visits
            topk_prob.append(np.pad(prob[:K], [(0, K - len(prob[:K]))], mode="constant"))

        return np.average(topk_prob, axis=0)

    def get_individual_jsds(self, gene_data):
        """
        get jsd scores of individual evaluation metrics
        :param t1: test_data
        :param t2: gene_data
        :return:
        """
        # gene_data = gene_data.detach().cpu().numpy()
        loc_seq = gene_data["locs"]

        gene_dist = self.get_distances(loc_seq) / self.max_distance
        gene_rg = self.get_rg(loc_seq) / self.max_distance
        gene_period = self.get_periodicity(loc_seq)
        gene_overall_topk_freq = self.get_overall_topk_freq(loc_seq, K=20)
        gene_topk_freq = self.get_topk_freq(loc_seq, K=10)

        gene_dist_p, _ = np.histogram(gene_dist, bins=np.arange(0, 1, 1 / 1000))
        dist_jsd = distance.jensenshannon(gene_dist_p, self.ref_dist_p)

        gene_rg_p, _ = np.histogram(gene_rg, bins=np.arange(0, 1, 1 / 1000))
        rg_jsd = distance.jensenshannon(gene_rg_p, self.ref_rg_p)

        gene_period_p, _ = np.histogram(gene_period, bins=np.arange(0, 1, 1 / 1000))
        period_jsd = distance.jensenshannon(gene_period_p, self.ref_period_p)

        gene_overall_topk_freq_p, _ = np.histogram(gene_overall_topk_freq, bins=np.arange(0, 1, 1 / 1000))
        overall_topk_freq_jsd = distance.jensenshannon(gene_overall_topk_freq_p, self.ref_overall_topk_freq_p)

        gene_topk_freq_p, _ = np.histogram(gene_topk_freq, bins=np.arange(0, 1, 1 / 1000))
        topk_freq_jsd = distance.jensenshannon(gene_topk_freq_p, self.ref_topk_freq_p)

        return dist_jsd, rg_jsd, period_jsd, overall_topk_freq_jsd, topk_freq_jsd
