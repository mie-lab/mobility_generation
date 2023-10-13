import pickle
import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm import tqdm
import os

from sklearn.metrics import f1_score, recall_score


def markov_transition_prob(df, n=1):
    COLUMNS = [f"loc_{i+1}" for i in range(n)]
    COLUMNS.append("toLoc")

    locSequence = pd.DataFrame(columns=COLUMNS)

    locSequence["toLoc"] = df.iloc[n:]["location_id"].values
    for i in range(n):
        locSequence[f"loc_{i+1}"] = df.iloc[i : -n + i]["location_id"].values
    return locSequence.groupby(by=COLUMNS).size().to_frame("size").reset_index()


def get_true_pred_pair(locSequence, df, n=1):
    testSeries = df["location_id"].values

    true_ls = []
    pred_ls = []
    for i in range(testSeries.shape[0] - n):
        locCurr = testSeries[i : i + n + 1]
        numbLoc = n

        # loop until finds a match
        while True:
            res_df = locSequence
            for j in range(n - numbLoc, n):
                res_df = res_df.loc[res_df[f"loc_{j+1}"] == locCurr[j]]
            res_df = res_df.sort_values(by="size", ascending=False)

            if res_df.shape[0]:  # if the dataframe contains entry, stop finding
                # choose the location which are visited most often for the matches
                pred = res_df["toLoc"].drop_duplicates().values
                break
            # decrese the number of location history considered
            numbLoc -= 1
            if numbLoc == 0:
                pred = np.zeros(10)
                # pred = locSequence.sort_values(by="size", ascending=False)["toLoc"].drop_duplicates().values
                break

        true_ls.append(locCurr[-1])
        pred_ls.append(pred)

    return true_ls, pred_ls


def get_performance_measure(true_ls, pred_ls):
    acc_ls = [1, 5, 10]

    res = []
    ndcg_ls = []
    # total number
    res.append(len(true_ls))
    for top_acc in acc_ls:
        correct = 0
        for true, pred in zip(true_ls, pred_ls):
            if true in pred[:top_acc]:
                correct += 1

            # ndcg calculation
            if top_acc == 10:
                idx = np.where(true == pred[:top_acc])[0]
                if len(idx) == 0:
                    ndcg_ls.append(0)
                else:
                    ndcg_ls.append(1 / np.log2(idx[0] + 1 + 1))

        res.append(correct)

    top1 = [pred[0] for pred in pred_ls]
    f1 = f1_score(true_ls, top1, average="weighted")
    recall = recall_score(true_ls, top1, average="weighted")

    res.append(f1)
    res.append(recall)
    res.append(np.mean(ndcg_ls))

    # rr
    rank_ls = []
    for true, pred in zip(true_ls, pred_ls):
        rank = np.where(pred == true)[0] + 1
        # (np.nonzero(pred == true)[0] + 1).astype(float)
        if len(rank):
            rank_ls.append(rank[0])
        else:
            rank_ls.append(0)
    rank = np.array(rank_ls, dtype=float)

    #
    rank = np.divide(1.0, rank, out=np.zeros_like(rank), where=rank != 0)
    # rank[rank == np.inf] = 0
    # append the result
    res.append(rank.sum())

    return pd.Series(res, index=["total", "correct@1", "correct@5", "correct@10", "f1", "recall", "ndcg", "rr"])


def get_markov_res(train, test, n=2):
    locSeq_df = markov_transition_prob(train, n=n)

    # true_ls, pred_ls = get_true_pred_pair(locSeq_df, test, n=n)

    # print(locSeq)
    return get_true_pred_pair(locSeq_df, test, n=n)
