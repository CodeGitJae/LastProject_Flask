from datetime import datetime
import datetime as dt
import requests
import json


# date, time, 위경도(x,y)를 각각 입력받아 해당 날짜의 시간에 대한
# 온도, 습도, 강수량, 풍속을 반환하는 함수
# date 형태 예시) 20240627
# time 형태 예시) 0500
def get_weather(date, x, y):

    one_day = dt.timedelta(days=1)
    two_day = dt.timedelta(days=2)

    now = datetime.now()
    next_one_day = now + one_day
    next_two_day = now + two_day

    now_date = now.strftime("%Y%m%d")
    next_one_date = next_one_day.strftime("%Y%m%d")
    next_two_date = next_two_day.strftime("%Y%m%d")

    x = int(float(x))
    y = int(float(y))
    
    url = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst'
    api_key = 'RXcvwxvz5SQjqoC1kUoQCZbaZF90otaOiu+10dpvoSmXMXwNqV/3BHvTSHwnBVmm5xhtCc+ifNdgonQbqqGsoA=='
    params ={
            'serviceKey' : api_key,
            'pageNo' : '1',
            'numOfRows' : '1000',
            'dataType' : 'JSON',
            'base_date' : now_date,
            'base_time' : '0500',
            'nx' : x,
            'ny' : y
            }

    response = requests.get(url, params=params)
    data = response.content
    dict_data = json.loads(data)
    dict_items = dict_data["response"]["body"]["items"]["item"]

    TMP = []
    REH = []
    WSD = []
    PCP = []

    now_date = str(now_date)
    next_one_date = str(next_one_date)
    next_two_date = str(next_two_date)
    time = '2100'
    
    for i in range(len(dict_items)):
        if (dict_items[i]["fcstDate"] == now_date or dict_items[i]["fcstDate"] == next_one_date or dict_items[i]["fcstDate"] == next_two_date) and (dict_items[i]["fcstTime"] == time):
            if (dict_items[i]["category"] == "TMP"):
                TMP.append(dict_items[i]["fcstValue"])
            elif (dict_items[i]["category"] == "REH"):
                REH.append(dict_items[i]["fcstValue"])
            elif (dict_items[i]["category"] == "WSD"):
                WSD.append(dict_items[i]["fcstValue"])
            elif (dict_items[i]["category"] == "PCP"):
                if (dict_items[i]["fcstValue"] == "강수없음"):
                    PCP.append(0)
                else:
                    PCP.append(dict_items[i]["fcstValue"])
    return TMP, REH, WSD, PCP



# 주소 address를 입력받아 api에 전송해서 위경도 x, y 값을 받아 반환한다.
def get_position(address):
    url = "https://dapi.kakao.com/v2/local/search/address.json" #요청할 url 주소
    headers = {"Authorization": 'KakaoAK 8157b78b80003b4aa367ddacfda1ae2b'} #REST API 키(유효한 키)
    query = {'query': address} #입력할 주소
    dict_data = requests.get(url,
                        headers=headers,
                        data=query).json() #카카오 API 요청
    y = dict_data["documents"][0]["address"]["x"]
    x = dict_data["documents"][0]["address"]["y"]

    return x, y


# 주소 address를 입력받아 해당 위치에 기반한 온도, 습도, 풍속, 강수량을 반환한다.
def get_tmp_reh_wsd_pcp(address):
    x, y = get_position(address)
    print(x)
    print(y)
    TMP, REH, WSD, PCP = get_weather(20240627, x, y)

    return TMP, REH, WSD, PCP