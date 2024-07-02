from datetime import datetime
import datetime as dt
import requests
import json
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torchvision.transforms as transforms
import torch
import os
import torch.nn as nn
import torch.nn.init
from torch.autograd import Variable
import pandas as pd
import requests
from bs4 import BeautifulSoup


def get_img_src(word):
    img_list = []
    url = f'https://search.naver.com/search.naver?ssc=tab.image.all&where=image&sm=tab_jum&query={word}'

    response = requests.get(url)
    question_list_page_content = response.text
    soup = BeautifulSoup(question_list_page_content, "html.parser")

    for i in range(len(soup.select("img"))):
        text = str(soup.select("img")[i])
        text = text[text.index("https"):text.index(";type")]
        img_list.append(text)

    return img_list



# 좌표기반
# date, time, 위경도(x,y)를 각각 입력받아 해당 날짜의 시간에 대한
# 온도, 습도, 강수량, 풍속을 반환하는 함수
# date 형태 예시) 20240627
# time 형태 예시) 0500

def get_position_weather(x, y):

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
                    to_cut = dict_items[i]["fcstValue"]
                    PCP.append(float(to_cut[:to_cut.index("m")]))
    return TMP, REH, WSD, PCP


# 주소기반
# date, time, 위경도(x,y)를 각각 입력받아 해당 날짜의 시간에 대한
# 온도, 습도, 강수량, 풍속을 반환하는 함수
# date 형태 예시) 20240627
# time 형태 예시) 0500
def get_address_weather(address):

    x, y = get_position(address)

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
                    to_cut = dict_items[i]["fcstValue"]
                    PCP.append(float(to_cut[:to_cut.index("m")]))
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
def get_address_tmp_reh_wsd_pcp(address):
    x, y = get_position(address)

    TMP, REH, WSD, PCP = get_position_weather(x, y)

    return TMP, REH, WSD, PCP




# 좌표 x, y를 입력받아 해당 위치에 기반한 온도, 습도, 풍속, 강수량을 반환한다.
def get_position_tmp_reh_wsd_pcp(x, y):
    TMP, REH, WSD, PCP = get_position_weather(x, y)

    return TMP, REH, WSD, PCP




# 모델을 불러와 학습시키는 함수
def call_model(data):
    ss = StandardScaler()
    ms = MinMaxScaler()

    path = os.getcwd().replace('\\', '/') + '/pybo'

    y = pd.read_csv(path+"/static/data/y_data.csv")
    ms.fit(y)

    X_act = data
    X_act_ss = ss.fit_transform(X_act)
    X_act_tensors = torch.Tensor(X_act_ss)
    X_act_tensors_f = torch.reshape(X_act_tensors, (X_act_tensors.shape[0], 1, X_act_tensors.shape[1]))

    input_size = 4
    hidden_size = 500
    num_layers = 1

    num_classes = 2
    
    model = LSTM(num_classes, input_size, hidden_size, num_layers, 365)
    model.load_state_dict(torch.load(path+"/static/models/LSTM_MODELt_1L_500h.pth"))

    model.train()

    predict = model(X_act_tensors_f)
    predict = predict.data.numpy()
    predict = ms.inverse_transform(predict)

    return predict.tolist()



# LSTM 모델 클래스
class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        self.num_classes=num_classes
        self.num_layers=num_layers
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.seq_length=seq_length
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        self.fc_1 = nn.Linear(hidden_size*num_layers, 256)
        self.fc1 = nn.Linear(256, hidden_size*num_layers)
        
        self.fc_3 = nn.Linear(hidden_size*num_layers, 256)
        self.fc3 = nn.Linear(256, num_classes) 
        
        self.relu = nn.ReLU()


    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn.view(x.size(0), -1)
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc1(out)
        
        out = self.relu(out)
        out = self.fc_3(out)
        out = self.relu(out)
        out = self.fc3(out)
        
        return out
    
