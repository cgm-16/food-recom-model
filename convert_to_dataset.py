import logging as log
import pandas as pd

log.basicConfig(level=log.DEBUG)

WEATHER_DATA = "./results/weather.csv"
DATALAB_DATA = "./results/datalab_norm_list.csv"

def merge_datasets():
    df_weather = pd.DataFrame(pd.read_csv(WEATHER_DATA))
    df_datalab = pd.DataFrame(pd.read_csv(DATALAB_DATA))
    df_datalab_ranked = convert_dataset_by_value(df_datalab)
    df_results = multiply_rows(df_weather, df_datalab_ranked)
    
    log.debug(df_results.head(10))
    df_results.to_csv("./results/model_dataset.csv")

def convert_dataset_by_value(df : pd.DataFrame):
    result = pd.DataFrame(columns=["date", "0", "1", "2", "3", "4"])
    for row in df.itertuples():
        res_row = {"date" : row[2]}
        comp_dict = row._asdict()
        del comp_dict["Index"]
        del comp_dict["_1"]
        del comp_dict["기간"]
        for i in range(5):
            max_val = 0.0
            max_key = None
            for k, v in comp_dict.items():
                if max_val < v:
                    max_val = v
                    max_key = k
            res_row[str(i)] = max_key
            del comp_dict[max_key]
        result.loc[len(result)] = res_row
    result = result.set_axis(["date", "first", "second", "third", "fourth", "fifth"], axis=1)
    return result

def multiply_rows(df : pd.DataFrame, ranked_df : pd.DataFrame):
    number_to_repeats = {"first" : 20, "second" : 10, "third" : 6, "fourth" : 5, "fifth" : 4}
    result = pd.DataFrame(columns=["date", "precipitation", "avg_temp", "lowest_temp", "highest_temp", "is_rain", "is_snow", "is_holiday", "meal_type"])
    for row_wea, row_rank in zip(df.itertuples(), ranked_df.itertuples()):
        dict_rank = row_rank._asdict()
        del dict_rank["Index"]
        del dict_rank["date"]
        for k, v in dict_rank.items():
            weather = row_wea._asdict()
            del weather["_1"]
            weather["meal_type"] = v
            for _ in range(number_to_repeats[k]):
                result.loc[len(result)] = weather
    return result

def main():
    merge_datasets()

if __name__ == '__main__':
    main()