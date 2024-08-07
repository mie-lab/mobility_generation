import numpy as np
from collections import defaultdict
from copy import deepcopy as dc

from mechanistic import scale_fitter_no_grid


from bisect import bisect
from scipy import stats
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from joblib import Parallel, delayed
from multiprocessing import Pool

import matplotlib.pyplot as plt

import itertools
from collections import Counter


def total_likelihood(source_target, cell_attractiveness, cell_p_change, return_all_values=False):
    """
    Compute the total likelihood given all parameters.

    source_target: (list of lists)
        List of transitions.
    cell_attractiveness: (dict)
        Assigning attractiveness to each container
    cell_p_change: (list)
        Dictionary containing the probability to travel at any given level-distance for each location.
    return_all_values: (bool)
        If true, returns the likelihood associated to each transition.

    """

    n_scales = len(source_target[0][0])
    unique, counts = np.unique(source_target, return_counts=True, axis=0)

    # Create sequence of origin-destinations
    change = unique[:, 0] != unique[:, 1]
    change_indexes = np.argmax([list(i) + [True] for i in change], axis=1)

    # Compute adjusted bernullis from dict
    unique_cell_p_change = np.array([cell_p_change[tuple(cell)] for cell in unique[:, 0]])

    # Compute cell attractiveness
    attractiveness_source = np.array(
        [[cell_attractiveness[tuple(a[: n + 1])] for n in range(n_scales)] for a in unique[:, 0]]
    )
    attractiveness_target = np.array(
        [[cell_attractiveness[tuple(a[: n + 1])] for n in range(n_scales)] for a in unique[:, 1]]
    )

    normalized_attr = attractiveness_target / (1 - attractiveness_source)
    normalized_attr = np.array([list(i) + [1] for i in normalized_attr])

    # Probabilities of scale changes

    p_s = np.clip(np.choose(change_indexes, unique_cell_p_change.T), 0.01, 0.99)

    # Compute prob of selecting a cell
    cell_probabilities_1 = np.choose(change_indexes, normalized_attr.T)

    # Prob of selecting all other cells
    cell_probabilities_2 = np.array([np.prod(k[change_indexes[n] + 1 :]) for n, k in enumerate(attractiveness_target)])

    # Compute total likelihood
    if not return_all_values:
        r = p_s * cell_probabilities_1 * cell_probabilities_2
        r = np.concatenate([[i] * c for i, c in zip(r, counts)])
        return sum(-np.log(r))

    else:
        r = p_s * cell_probabilities_1 * cell_probabilities_2

        return (
            np.concatenate(
                [[i] * c for i, c in zip(r, counts)]
            ),  # array([6.02322201, 6.02322201, ..., 8.23421385, 4.28297013])  # probabilities
            np.concatenate(
                [[i] * c for i, c in zip(unique, counts)]
            ),  # array([[[0, 3], [390, 2]], [[390, 2], [3, 21]], ... ])        # transitions
        )


def find_cell_p_change(source_target, n_scales, series):
    """
    Find the probability of transitioning at given scale for all locations.

    Input
    -----
        source_target: numpy 2d-array (N, n_scales, n_scales)
            The original sequence of cell-ids
        n_scales: (int)
            Number of levels

    Output
    ------
        cell_p_change_dict: (dict)
            Dictionary with locations as keys, and prob of transitioning at any distance-level as values

        cell_p_change_dict_by_d: (dict)
           Dictionary with distance from home as key, prob of transitioning at any distance-level as value


    """

    # Find where location changes
    change = source_target[:, 0] != source_target[:, 1]

    # Find  most important location
    home = sorted(Counter([tuple(i) for i in series]).items(), key=lambda x: x[1], reverse=True)[0][0]

    # for each location find how many levels it is far from home
    locations = np.unique(series, axis=0)
    distance_from_home = np.argmax(locations != np.array([home] * len(locations)), axis=1)
    dictionary = dict(zip([tuple(i) for i in locations], distance_from_home))
    dictionary[tuple(home)] = n_scales

    # group all transitions by distance from home (of the source)
    source_target = sorted(source_target, key=lambda x: dictionary[tuple(x[0])])
    groups = itertools.groupby(source_target, key=lambda x: dictionary[tuple(x[0])])

    # Create a dictionary of cell_p_change for each location.
    cell_p_change_dict = {}
    cell_p_change_dict_by_d = {}
    for key, group in groups:
        group = np.array(list(group))

        # At which index the transition occurs
        change = group[:, 0] != group[:, 1]
        change_indexes = np.argmax([list(i) + [True] for i in list(change)], axis=1)
        d = Counter(change_indexes)

        # Compute values and update dictionary
        cell_p_change = [d.get(n, 0.001) for n in range(n_scales + 1)]
        cell_p_change = [i / float(sum(cell_p_change)) for i in cell_p_change]

        cell_p_change_dict_by_d[n_scales - key] = cell_p_change

        for location in group[:, 0]:
            cell_p_change_dict[tuple(location)] = cell_p_change

    # Fix eventually the last location in the series (it could be that it did not appear elsewhere)
    last_location = tuple(series[-1])
    cell_p_change_dict[last_location] = cell_p_change_dict_by_d[n_scales - dictionary[last_location]]

    return cell_p_change_dict, cell_p_change_dict_by_d


def compute_cell_attractiveness(series, cell_attractiveness, n=0):
    """
    Recursive function to populate the `cell_attractiveness` dictionary.

    Input
    -----
        series: list
            Sequence of locations
        cell_attractiveness: dict
            Dictionary to populate
        n: int
            Level index
    """
    series = sorted(series)
    n_scales = len(series[0])

    # group the elements by id (up to scale n) and count the number of occurances
    groups = itertools.groupby(series, key=lambda x: tuple(x[: n + 1]))
    key_groups = [(key, len(list(group))) for key, group in groups]

    # update the dictionary: keys are cell ids, and values are attractiveness
    sum_values = sum([v for k, v in key_groups])
    key_groups = [(k, min(v / sum_values, 0.99)) for k, v in key_groups]

    cell_attractiveness.update(key_groups)

    # if we have not completed for all scales, apply this function recursively
    if n < n_scales - 1:
        for key, group in itertools.groupby(series, key=lambda x: tuple(x[: n + 1])):
            compute_cell_attractiveness(list(group), cell_attractiveness, n + 1)


def compute_likelihood(source_target, return_all_values=False):
    """Given a series of transitions (described as a hierarchy) compute the likelihood of the hiearchical partitioning.

    Input
    -----
        source_target : list of lists
            The series of trips in hierarchical description.
            The largest scale is the first value, the smallest scale is the last value.
    Output
    ------
        L: float
            The value of the likelihood.
        cell_attractiveness: dict
            Dictionary containing the attractiveness of cell_ids at all scales.
        cell_p_change: dict
            Dictionary containing the out-transition probabilities for each container.
        cell_p_change_by_d: list of floats
            Dictionary containing the out-transition probabilities for each distance-from-home.
    """

    n_scales = len(source_target[0][0])
    series = source_target[:, 0].tolist() + [source_target[-1, 1].tolist()]

    # Estimate the cell attractiveness, like {(2, ): 0.8, (20, ): 0.1, ..., (2, 15): 0.9, (2, 4): 0.05, ...}
    cell_attractiveness = {}
    compute_cell_attractiveness(series, cell_attractiveness, 0)

    # Find cell p change from data
    cell_p_change, cell_p_change_by_d = find_cell_p_change(source_target, n_scales, series)

    # Compute likelihood
    L = total_likelihood(source_target, cell_attractiveness, cell_p_change, return_all_values)
    return L, cell_attractiveness, cell_p_change, cell_p_change_by_d


def recover_parameters_from_fitted_trace(trace):
    """Given a trace, recover parameters a and p.

    Input
    -----
    trace: list of lists (e.g.[ [1,2,3] , [1,2,1], ...., [2,1,1]])
           Sequence of locations in hierarchical form.


    Output
    ------
    nested_dictionary: (dict)
        Gives the attractiveness of each container.

    cell_p_change: (dict)
        Gives the probability of changing at any level-distance for each cell.

    """

    # Create the source_target_list and compute the parameters of the model
    source_target = np.stack([trace[:-1], trace[1:]], axis=1)
    (proba_dist, proba_dist_counts), cell_attractiveness, cell_p_change, _ = compute_likelihood(
        source_target, return_all_values=True
    )

    # Create nested dictionary
    nested_dictionary = []
    items = sorted(cell_attractiveness.items(), key=lambda x: len(x[0]))
    for group1 in itertools.groupby(items, lambda x: len(x[0])):
        scale = group1[0]
        new_group = sorted(list(group1[1]), key=lambda x: x[0][: scale - 1])
        new_dict = dict()
        for group2 in itertools.groupby(new_group, lambda x: x[0][: scale - 1]):
            new_dict[group2[0]] = dict(group2[1])
        nested_dictionary.append(new_dict)
    return nested_dictionary, cell_p_change


def generate_trace(nested_dictionary, cell_p_change, size, initial_position=None):
    """
    Generate a synthetic trace starting from a sequence of locations with the corresponding scale structure

    Input
    -----
    nested_dictionary: (dict)
        Gives the attractiveness of each container.
    cell_p_change: (dict)
        Gives the probability of changing at any level-distance for each cell.
    size: (int)
        Length of the sythethic sequence.
    initial position: (list)
        Initial position


    Output
    ------
    synthetic_trace: list of lists (e.g.[ [1,2,3] , [1,2,1], ...., [2,1,1]])

    """

    # Recover parameters
    traces_len = int(size)
    n_scales = len(list(cell_p_change.values())[0]) - 1

    # Initialize synthetic trace
    if initial_position is None:
        locs = range(len(cell_p_change.keys()))
        initial_position = list(cell_p_change.keys())[np.random.choice(locs)]

    L = tuple(initial_position)  # current cell
    synthetic_series = [L[-1]]  # sequence of cells
    scale_change = cell_p_change[L]  # current p_change

    while len(synthetic_series) < traces_len:
        # Iterate through steps

        # Select level
        change = np.random.choice(range(n_scales + 1), p=scale_change)

        if change == n_scales:
            new_cell = L

        else:
            # Move
            attractiveness = nested_dictionary[change][L[:change]]
            new_cell = L[: change + 1]

            # Select new_cell
            possible_cells = [i for i in attractiveness.items() if i[0] != new_cell]
            if len(possible_cells) == 0:
                continue

            k, v = list(zip(*list(possible_cells)))
            new_cell = k[np.random.choice(range(len(v)), p=np.array(list(v)) / sum(list(v)))]

            scale = change + 1
            while scale < n_scales:
                attractiveness = nested_dictionary[scale][new_cell]
                k, v = list(zip(*list(attractiveness.items())))
                new_cell = k[np.random.choice(range(len(v)), p=np.array(list(v)) / sum(list(v)))]
                scale += 1

        # Update values
        synthetic_series.append(new_cell[-1])
        L = new_cell
        scale_change = cell_p_change[L]

    return synthetic_series


def haversine(points_a, points_b, radians=False):
    """
    Calculate the great-circle distance bewteen points_a and points_b
    points_a and points_b can be a single points or lists of points.

    Author: Piotr Sapiezynski
    Source: https://github.com/sapiezynski/haversinevec

    Using this because it is vectorized (stupid fast).
    """

    def _split_columns(array):
        if array.ndim == 1:
            return array[0], array[1]  # just a single row
        else:
            return array[:, 0], array[:, 1]

    if radians:
        lat1, lon1 = _split_columns(points_a)
        lat2, lon2 = _split_columns(points_b)

    else:
        # convert all latitudes/longitudes from decimal degrees to radians
        lat1, lon1 = _split_columns(np.radians(points_a))
        lat2, lon2 = _split_columns(np.radians(points_b))

    # calculate haversine
    lat = lat2 - lat1
    lon = lon2 - lon1

    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lon * 0.5) ** 2
    h = 2 * 6371e3 * np.arcsin(np.sqrt(d))
    return h  # in meters


class ScalesOptim:
    """Scales optimizer class.

    Input
    -----
        labels : list
            Sequence of stops. Example [0, 1, 1, 0, 10, 2, ...]
        stop_locations : np.array (shape=(len(set(labels))), 2))
            (lat, lon) Coordinates of locations. Row index must correspond to label index.
        distance_func : callable
            Function to compute distances
        min_dist : float (> 1)
            Consider linkage solutions such that distance[i]>min_distance*distance[i-1]
        min_diff : int

        siglvl : float (0 < siglvl
            Significance value for tests.
        stat_test: callable
            Function that takes as input two lists of values. Statistical test.
        bootstrp: bool
            Run the bootstrap routine or not.
        information_criterion : str (None, 'AIC' or 'BIC')
            If different than None, choose between AIC or BIC.
        n_procs: int
            Number of processes
        linkage_method: str
        verbose: bool
        bootstrap_iter: (int)
            Number of bootstrap iterations.


    """

    def __init__(
        self,
        labels,
        stop_locations,
        distance_func=haversine,
        min_dist=1.2,
        min_diff=0,
        siglvl=0.05,
        stat_test=stats.ks_2samp,
        bootstrap=False,
        repeated_trips=True,
        information_criterion="AIC",
        nprocs=1,
        linkage_method="complete",
        verbose=True,
        bootstrap_iter=200,
    ):

        self.stop_locations = stop_locations
        self.distance_func = distance_func
        self.labels = labels
        self.min_dist = min_dist
        self.min_diff = min_diff
        self.siglvl = siglvl
        self.stat_test = stat_test
        self.bootstrap = bootstrap
        self.repeated_trips = repeated_trips
        self.information_criterion = information_criterion
        self.nprocs = nprocs
        self.verbose = verbose
        self.num_iter = bootstrap_iter
        self.alphas = []

        # Find possible clusterings
        pdistance = general_pdist(self.stop_locations, distance_function=distance_func)
        # pdistance = pdist(self.stop_locations)
        self.Z = linkage(pdistance, method=linkage_method)

        # Merge distances for all branches to follow
        self.merge_d = []
        for i, d in enumerate(self.Z[:-1, 2]):
            if i == 0 or d > self.min_dist * self.merge_d[-1]:
                self.merge_d.append(d)

        self.max_d = self.Z[-1, 2]

    def _worker(self, inputs):

        n, max_Rs, label_to_cell_map, final_series, final_scales, n_params_prev = inputs

        # Update series
        candidate_series = []
        scale_index = bisect(sorted(final_scales.values()), n)
        for element in final_series:
            a = dc(element)
            a.insert(scale_index, label_to_cell_map[element[-1]].item())  # `.item()` yields native python int
            candidate_series.append(a)

        # Compute likelihood
        source_target = np.stack([candidate_series[:-1], candidate_series[1:]], axis=1)
        (proba_dist, proba_dist_counts), _, _, alphas = scale_fitter_no_grid.compute_likelihood(
            source_target, return_all_values=True
        )
        LL = sum(-np.log(proba_dist))

        if not self.repeated_trips:
            proba_dist_proba_dist_counts = set(zip(proba_dist, map(str, proba_dist_counts)))
            proba_dist = np.array([v0 for v0, v1 in proba_dist_proba_dist_counts])

        # Compute the criterion
        n_params = n_params_prev + len(set([i[scale_index] for i in candidate_series])) + 1

        if self.information_criterion == "AIC":
            criterion = 2 * n_params + 2 * LL
        elif self.information_criterion == "BIC":
            criterion = np.log(len(self.labels) - 1) * n_params + 2 * LL
        else:
            criterion = LL

        return candidate_series, n_params, LL, proba_dist, criterion, n, max_Rs, label_to_cell_map, alphas

    def find_best_scale(self):
        """
        Run the optimization routine and find the best combinations of the scales.

        """
        # Find L_min
        series = [[c] for c in self.labels]
        source_target = np.stack([series[:-1], series[1:]], axis=1)
        (proba_dist_min, _), _, _, alphas_min = scale_fitter_no_grid.compute_likelihood(
            source_target, return_all_values=True
        )
        L_min = sum(-np.log(proba_dist_min))

        # Initialize all values
        final_series = dc(series)
        final_proba_dist = proba_dist_min

        scales = dict()
        sizes = dict()

        final_scales = dc(scales)
        final_sizes = dc(sizes)
        final_series = dc(series)
        final_alphas = dc(alphas_min)

        improvement = True
        scale = 2
        likelihoods = defaultdict(list)
        criterion_s = defaultdict(list)

        n_params_min = n_params_prev = len(set(self.labels)) + 1

        if self.information_criterion == "AIC":
            criterion_min = 2 * n_params_min + 2 * L_min

        elif self.information_criterion == "BIC":
            criterion_min = np.log(len(self.labels) - 1) * n_params_min + 2 * L_min

        else:
            criterion_min = L_min

        # Add a scale unntil there is no more improvement
        if self.verbose:
            print("Searching for minimum at scale {}:\n".format(scale))
        while improvement:
            improvement = False

            # Try all possible clusterings
            inputs = []
            for n, max_Rs in enumerate(reversed(self.merge_d)):
                label_to_cell_map = dict(enumerate(fcluster(self.Z, max_Rs, criterion="distance")))
                if (
                    len(set(scales.values()) & set(range(n - self.min_diff, n + 1 + self.min_diff))) == 0
                ):  # This obscure line just checks if n or indices adjacent to n (tunable parameter `min_diff`) are already chosen.
                    inputs.append((n, max_Rs, label_to_cell_map, final_series, final_scales, n_params_prev))

            if self.nprocs > 1:
                result = Parallel(n_jobs=self.nprocs, max_nbytes=1e6)(delayed(self._worker)(inp) for inp in inputs)
            else:
                result = map(self._worker, inputs)

            for candidate_series, n_params, LL, proba_dist, criterion, n, max_Rs, label_to_cell_map, alphas in result:
                d_criterion = criterion - criterion_min

                # Save likelihood and criterion
                likelihoods[scale].append((max_Rs, LL))
                criterion_s[scale].append((max_Rs, criterion))

                # Print user output
                if self.verbose:
                    print("    It: %d/%d" % (n, len(self.merge_d) - 1), end=" ")
                    if max_Rs < 1000:
                        print("| d: %.01f m" % max_Rs, end=" ")
                    else:
                        print("| d: %.01f km" % (max_Rs / 1000), end=" ")
                    print("| L: %.01f" % LL, end=" ")
                    print(
                        "| p: "
                        + ", ".join(
                            [
                                ": [".join([str(i), ",".join(["%.02f" % k for k in v] + ["] "])])
                                for i, v in alphas.items()
                            ]
                        ),
                        end="",
                    )
                    if self.information_criterion:
                        print("| %s: %.01f" % (self.information_criterion, criterion), end=" ")
                        print("| ∆%s: %.01f" % (self.information_criterion, d_criterion))
                    else:
                        print("| ∆L: %.01f" % d_criterion)

                # Check if Likelihood has improved and criterion is positive
                if d_criterion < 0:
                    improvement = True
                    L_min = LL
                    criterion_min = criterion
                    proba_dist_min = proba_dist
                    scales[scale] = n
                    sizes[scale] = max_Rs
                    series = candidate_series
                    n_params_min = n_params
                    alphas_min = alphas

            if improvement:
                if self.verbose:
                    print("\nFound minimum at   d:", end=" ")
                    if sizes[scale] < 1000:
                        print("%.01f m" % sizes[scale])
                    else:
                        print("%.01f km" % (sizes[scale] / 1000))
                    print("                   L: %.01f" % L_min)
                    if self.information_criterion:
                        print("                 %s: %.01f" % (self.information_criterion, criterion_min), end="\n\n")
                    else:
                        print("", end="\n\n")

                # Compute p-value
                if self.siglvl is not None:
                    if self.verbose:
                        print("Result of statistical test:")
                    if self.bootstrap:
                        pval, L_vec, L_prev_vec = bootstrap_pval(
                            final_series, series, self.stat_test, num_iter=self.num_iter, nprocs=self.nprocs
                        )
                    else:
                        pval = self.stat_test(-np.log(final_proba_dist), -np.log(proba_dist_min))[1]
                    if pval > self.siglvl:
                        del likelihoods[scale]
                        del criterion_s[scale]
                        improvement = False

                if self.verbose and self.bootstrap:
                    plt.figure(figsize=(6, 2))
                    plt.hist(L_vec, label="Candidate", alpha=0.5)
                    plt.hist(L_prev_vec, label="Previous", alpha=0.5)
                    plt.xlabel("L", fontsize=12)
                    plt.xticks(fontsize=12)
                    plt.yticks(fontsize=12)
                    plt.legend(fontsize=12)
                    plt.show()
                    print("    p =", pval)
                    if pval < self.siglvl:
                        print("    --> Rejecting null hypothesis.", end="\n\n")
                    else:
                        print("    --x Cannot reject null hypothesis.", end="\n\n")

            if improvement:
                scale += 1
                n_params_prev = n_params_min

                final_scales = dc(scales)
                final_series = dc(series)
                final_sizes = dc(sizes)
                final_proba_dist = dc(proba_dist_min)
                final_alphas = dc(alphas_min)

                if self.verbose:
                    print("Searching for minimum at scale {}:\n".format(scale))
            else:
                if self.verbose:
                    print("Could not improve beyond scale %d. Optimization ends." % (scale - 1))
                # else:
                #     print("Found %d scales" % (scale - 1))

        # Add code that sorts scale indices by size

        return final_series, final_scales, likelihoods, criterion_s, final_sizes, final_proba_dist, final_alphas


def _worker_bootstrap_pval(inputs):
    (
        seed,
        N,
        source_target,
        cell_attractiveness,
        fitted_cell_p_change,
        source_target_prev,
        cell_attractiveness_prev,
        fitted_cell_p_change_prev,
    ) = inputs
    np.random.seed(seed)
    random_indices = np.random.randint(0, N, 300)
    L = scale_fitter_no_grid.total_likelihood(source_target[random_indices], cell_attractiveness, fitted_cell_p_change)
    np.random.seed(seed + 1)
    random_indices = np.random.randint(0, N, 300)
    L_prev = scale_fitter_no_grid.total_likelihood(
        source_target_prev[random_indices], cell_attractiveness_prev, fitted_cell_p_change_prev
    )
    return L, L_prev


def bootstrap_pval(series_prev, series, stat_test, num_iter=1000, nprocs=10):
    """Compute bootstrap pvalue for a series.

    Input
    -----
        series : list of lists
        scale_index : int
            The index in the series which needs to be tested
        num_iter : int
            Number of bootstrap iterations
    """
    # Number of trips
    N = len(series) - 1
    n_scales = len(series[0])
    n_scales_prev = len(series_prev[0])

    if n_scales != n_scales_prev + 1:
        raise

    # Reshape to trips
    source_target = np.stack([series[:-1], series[1:]], axis=1)
    source_target_prev = np.stack([series_prev[:-1], series_prev[1:]], axis=1)

    # Cell attractiveness and cell p change of current
    cell_attractiveness = {}
    scale_fitter_no_grid.compute_cell_attractiveness(series, cell_attractiveness, 0)
    fitted_cell_p_change, _ = scale_fitter_no_grid.find_cell_p_change(source_target, n_scales, series)

    # Cell attractiveness and cell p change of prev
    cell_attractiveness_prev = {}
    scale_fitter_no_grid.compute_cell_attractiveness(series_prev, cell_attractiveness_prev, 0)
    fitted_cell_p_change_prev, _ = scale_fitter_no_grid.find_cell_p_change(
        source_target_prev, n_scales_prev, series_prev
    )
    # Maintain bool array of iteration test succeses and failures

    inputs = []
    for seed in range(num_iter):
        inputs.append(
            (
                seed,
                N,
                source_target,
                cell_attractiveness,
                fitted_cell_p_change,
                source_target_prev,
                cell_attractiveness_prev,
                fitted_cell_p_change_prev,
            )
        )

    if nprocs > 1:
        p = Pool(nprocs)
        result = p.map(_worker_bootstrap_pval, inputs)
        # result = Parallel(n_jobs=nprocs, max_nbytes=1e6)(delayed(_worker_bootstrap_pval)(inp) for inp in inputs)
    else:
        result = map(_worker_bootstrap_pval, inputs)

    L_vec, L_prev_vec = [], []
    for L, L_prev in result:
        L_vec.append(L)
        L_prev_vec.append(L_prev)

    if nprocs > 1:
        p.close()

    if stat_test is None:
        pval = np.mean(np.array(L_vec) >= np.array(L_prev_vec))
    else:
        pval = stat_test(L_vec, L_prev_vec)[1]

    return pval, L_vec, L_prev_vec


def general_pdist(points, distance_function=haversine):
    """
    Calculate the distance bewteen each pair in a set of points given a distance function.

    Author: Piotr Sapiezynski
    Source: https://github.com/sapiezynski/haversinevec

    Input
    -----
        points : array-like (shape=(N, 2))
            (lat, lon) in degree or radians (default is degree)

    Output
    ------
        result : array-like (shape=(N*(N-1)//2, ))
    """
    c = points.shape[0]
    result = np.zeros((c * (c - 1) // 2,), dtype=np.float64)
    vec_idx = 0

    for idx in range(0, c - 1):
        ref = points[idx]
        temp = distance_function(points[idx + 1 : c, :], ref, radians=False)
        # to be taken care of
        result[vec_idx : vec_idx + temp.shape[0]] = temp
        vec_idx += temp.shape[0]
    return result
