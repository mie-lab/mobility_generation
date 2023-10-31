import os
import pandas as pd
import geopandas as gpd

from loc_predict.processing import _split_train_test
from utils.utils import load_data

import powerlaw
import matplotlib.pyplot as plt

from metrics import radius_gyration, jump_length, location_frquency, wait_time

if __name__ == "__main__":
    data_dir = os.path.join("data", "validation")
    # read and preprocess
    sp = pd.read_csv(os.path.join(data_dir, "sp.csv"), index_col="id")
    loc = pd.read_csv(os.path.join(data_dir, "locs_s2.csv"), index_col="id")
    sp = load_data(sp, loc)

    train_data, vali_data, test_data = _split_train_test(sp)

    test_data = test_data.merge(
        loc.reset_index()[["id", "center"]].rename(columns={"id": "location_id"}), how="left", on="location_id"
    )
    test_data.rename(columns={"center": "geometry"})
    test_data = gpd.GeoDataFrame(test_data, geometry="geometry", crs="EPSG:4326")
    print(test_data)

    metric = jump_length(test_data)
    xlabel = "$\Delta r\,(m)$"
    ylabel = "$P(\Delta r)$"
    xmin = 1

    # fit power law
    fit = powerlaw.Fit(metric, xmin=xmin)

    # plotting
    powerlaw.plot_pdf(metric, label="data")
    fit.power_law.plot_pdf(linestyle="--", label="powerlaw fit")
    fit.truncated_power_law.plot_pdf(linestyle="--", label="truncated power law")
    fit.lognormal.plot_pdf(linestyle="--", label="lognormal fit")

    plt.legend(prop={"size": 13})
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)

    plt.show()
