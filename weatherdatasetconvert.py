import string
import pandas as pd

def convert_dataset(percfilename : string, tempfilename : string):
    with open("./holidays.txt", "r", encoding="utf-8") as f:
        holidays = [str(x).replace("\n", "") for x in f.readlines()]

    df_perc = pd.DataFrame(pd.read_csv(percfilename).drop(columns=["지점"]))
    df_temp = pd.DataFrame(pd.read_csv(tempfilename).drop(columns=["지점"]))

    df = df_perc.merge(df_temp, how="outer", on="날짜")
    df = df.set_axis(["date", "precipitation", "avg_temp", "lowest_temp", "highest_temp"], axis=1)

    df["precipitation"] = pd.to_numeric(df["precipitation"])
    df["avg_temp"] = pd.to_numeric(df["avg_temp"])
    df["lowest_temp"] = pd.to_numeric(df["lowest_temp"])
    df["highest_temp"] = pd.to_numeric(df["highest_temp"])
    df["is_rain"] = df.apply(lambda x : is_rain(x["precipitation"], x["avg_temp"]), axis=1)
    df["is_snow"] = df.apply(lambda x : is_snow(x["precipitation"], x["avg_temp"]), axis=1)
    df["is_holiday"] = df.apply(lambda x : is_holiday(x["date"], holidays), axis=1)
    df.to_csv("./results/weather.csv")

def is_rain(prec, temp):
    if prec > 0.0 and temp > 0.0:
        return True
    else:
        return False

def is_snow(prec, temp):
    if prec > 0.0 and temp <= 0.0:
        return True
    else:
        return False

def is_holiday(date, holidays : list):
    dayname = pd.to_datetime(date).day_name()
    if date in holidays or dayname == "Saturday" or dayname == "Sunday":
        return True
    else:
        return False

def main():
    convert_dataset("weather_perc.csv", "weather_temp.csv")

if __name__ == '__main__':
    main()
    