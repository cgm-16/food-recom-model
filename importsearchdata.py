# 라이브러리 임포트
import os
import sys
import json
import urllib.request
import pandas as pd

# 데이터랩 API
client_id = "Tx6hCbKgYrzKjaiVLHDd"
client_secret = "ri3HDCLeUv"

# 키워드 리스트
kw_list = [
    ["해장국"],
    ["순두부찌개", "삼겹살", "짜장면", "치킨"],
    ["라면", "김치찌개", "부대찌개", "간장 게장"],
    ["떡볶이", "곱창", "삼계탕", "비빔밥"],
    ["김밥", "감자탕", "해물 파전", "짬뽕"],
    ["순대", "콩국수", "칼국수", "설렁탕"],
    ["떡국", "된장찌개", "갈비", "춘천 닭갈비"],
    ["보쌈", "아귀찜", "잡채", "두부김치"],
    ["호박죽", "계란찜", "냉면", "도토리묵"],
    ["추어탕", "불고기", "낙지 볶음", "빙수"]
]

def datalab_api(keywords : list):
    url = "https://openapi.naver.com/v1/datalab/search"
    bodyjson = {
        "startDate" : "2021-01-01",
        "endDate" : "2021-12-31",
        "timeUnit" : "date",
        "keywordGroups" : [
            {"groupName" : str(keywords[0]), "keywords" : [str(keywords[0])]},
            {"groupName" : str(keywords[1]), "keywords" : [str(keywords[1])]},
            {"groupName" : str(keywords[2]), "keywords" : [str(keywords[2])]},
            {"groupName" : str(keywords[3]), "keywords" : [str(keywords[3])]},
            {"groupName" : str(keywords[4]), "keywords" : [str(keywords[4])]}
        ]
    }

    body = json.dumps(bodyjson)

    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",client_id)
    request.add_header("X-Naver-Client-Secret",client_secret)
    request.add_header("Content-Type","application/json")
    response = urllib.request.urlopen(request, data=body.encode("utf-8"))
    res_json = json.loads(response.read().decode("utf-8"))

    # 데이터프레임
    df1 = pd.DataFrame(res_json["results"][0]["data"]); df1.columns=["기간", str(keywords[0])]
    df2 = pd.DataFrame(res_json["results"][1]["data"]); df2.columns=["기간", str(keywords[1])]
    df3 = pd.DataFrame(res_json["results"][2]["data"]); df3.columns=["기간", str(keywords[2])]
    df4 = pd.DataFrame(res_json["results"][3]["data"]); df4.columns=["기간", str(keywords[3])]
    df5 = pd.DataFrame(res_json["results"][4]["data"]); df5.columns=["기간", str(keywords[4])]

    # 데이터프레임 병합
    df2 = df1.merge(df2, how="outer", on="기간")
    df3 = df2.merge(df3, how="outer", on="기간")
    df4 = df3.merge(df4, how="outer", on="기간")
    df5 = df4.merge(df5, how="outer", on="기간")

    return df5

# 최소-최대 정규화를 적용한 데이터랩 검색량 조회 함수
def datalab_list_norm(kw_list):
    df = datalab_api(kw_list[0] + kw_list[1])
    a, b = df.iloc[:,1].min(), df.iloc[:,1].max()

    i=2
    while i < len(kw_list):
        df1 = datalab_api(kw_list[0] + kw_list[i])
        x, y = df1.iloc[:,1].min(), df1.iloc[:,1].max()
        #최소-최대 정규화
        df1.iloc[:,2] = (df1.iloc[:,2] - x) / (y - x) * (b - a) + a
        df1.iloc[:,3] = (df1.iloc[:,3] - x) / (y - x) * (b - a) + a
        df1.iloc[:,4] = (df1.iloc[:,4] - x) / (y - x) * (b - a) + a
        df1.iloc[:,5] = (df1.iloc[:,5] - x) / (y - x) * (b - a) + a
        df1 = df1.drop(columns=["기간", "해장국"])
        df = pd.concat([df, df1], axis=1)
        i += 1

    return df

def main():
    df = datalab_list_norm(kw_list)
    df.head(10)
    df.to_csv("./results/datalab_norm_list.csv")

if __name__ == '__main__':
    main()
    