from flask import Flask, Blueprint, request
import pandas as pd
import os

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from torch.autograd import Variable

from pybo.static.modules import module

bp = Blueprint("main", __name__, url_prefix="/")

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





@bp.route("dust_info")
def dust_info():

    address = request.args.get("address")

    TMP, REH, WSD, PCP = module.get_tmp_reh_wsd_pcp(address)

    temperature = request.args.get('temperature')
    precipitation = request.args.get('precipitation')
    wind_speed = request.args.get('wind_speed')
    humidity = request.args.get('humidity')

    test = pd.DataFrame({
    '평균기온(°C)':[temperature],
    '일강수량(mm)':[precipitation],
    '평균 풍속(m/s)':[wind_speed],
    '평균 상대습도(%)':[humidity]
    })

    test = pd.DataFrame({
    '평균기온(°C)':[5.5*(-0.133)],
    '일강수량(mm)':[0.0*(-2.6)],
    '평균 풍속(m/s)':[1.9*(-0.469)],
    '평균 상대습도(%)':[63.1*(0.005)]
    })

    ss = StandardScaler()

    X_act = test[["평균기온(°C)", "일강수량(mm)", "평균 풍속(m/s)", "평균 상대습도(%)"]].values.tolist()
    X_act_ss = ss.fit_transform(X_act)
    X_act_tensors = torch.Tensor(X_act)
    X_act_tensors_f = torch.reshape(X_act_tensors, (X_act_tensors.shape[0], 1, X_act_tensors.shape[1]))

    lr = 0.0001

    input_size = 4
    hidden_size = 512
    num_layers = 2

    num_classes = 2

    path = os.getcwd().replace('\\', '/') + '/pybo/static/models/10m_test.h5'
    
    model = LSTM(num_classes, input_size, hidden_size, num_layers, 365)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.train()

    predict = model(X_act_tensors_f).tolist()
    dict = {'PM10':int(predict[0][0]*7*100), 'PM2.5':int(predict[0][1]*1.55*100)}

    return dict
