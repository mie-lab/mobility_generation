import pandas as pd


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

        df["start_day"] = (df["started_at"] - min_day).dt.days
        df["start_min"] = df["started_at"].dt.hour * 60 + df["started_at"].dt.minute
        df["weekday"] = df["started_at"].dt.weekday
        df["duration"] = (df["duration"] * 60).round()
        return df

    sp = sp.groupby("user_id", group_keys=False).apply(_get_time_info)
    return sp
