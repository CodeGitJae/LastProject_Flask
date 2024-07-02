from flask import Flask, Blueprint, request
import pandas as pd
import flask
import json


from pybo.static.modules import module

bp = Blueprint("main", __name__, url_prefix="/")
    

# 주소를 받아 오늘, 내일, 모레의 미세먼지 데이터를 dictionary 형태로 반환하는 라우터
@bp.route("dust_info")
def dust_info():

    address = request.args.get("address")

    TMP, REH, WSD, PCP = module.get_address_tmp_reh_wsd_pcp(address)

    data = pd.DataFrame({
    '평균기온':TMP,
    '강수량':PCP,
    '평균 풍속':WSD,
    '평균 상대습도':REH
    })

    predict = module.call_model(data)

    dict_return = [{"PM10":predict[0][0], "PM2.5":predict[0][1]}, {"PM10":predict[1][0], "PM2.5":predict[1][1]}, {"PM10":predict[2][0], "PM2.5":predict[2][1]}]
    json_val = json.dumps(dict_return)

    my_res = flask.Response(json_val)

    return my_res


# 좌표를 받아 오늘, 내일, 모레의 미세먼지 데이터를 dictionary 형태로 반환하는 라우터
@bp.route("position_dust_info")
def position_dust_info():

    x = request.args.get("x")
    y = request.args.get("y")

    TMP, REH, WSD, PCP = module.get_position_tmp_reh_wsd_pcp(x, y)

    data = pd.DataFrame({
    '평균기온':TMP,
    '강수량':PCP,
    '평균 풍속':WSD,
    '평균 상대습도':REH
    })

    predict = module.call_model(data)

    dict_return = [{"PM10":predict[0][0], "PM2.5":predict[0][1]}, {"PM10":predict[1][0], "PM2.5":predict[1][1]}, {"PM10":predict[2][0], "PM2.5":predict[2][1]}]
    json_val = json.dumps(dict_return)
    my_res = flask.Response(json_val)

    return my_res


# keyword 를 받아 해당 keyword를 네이버에 검색했을 때 나오는 이미지들의 주소들을
# json 형태로 반환하는 라우터 
@bp.route("get_img")
def get_img():
    keyword = request.args.get("keyword")
    img_list = module.get_img_src(keyword)

    dict_return = {"result":img_list}

    json_val = json.dumps(dict_return)

    my_res = flask.Response(json_val)
    return my_res




# 좌표를 입력받아 해당 주소에 해당하는 일기예보 결과를 반환하는 라우터
@bp.route("get_weather")
def get_weather():
    x = request.args.get("x")
    y = request.args.get("y")
    TMP, REH, WSD, PCP = module.get_position_weather(x, y)

    dict_return = {"result":{"기온":TMP, "습도":REH, "강수량":PCP}}

    json_val = json.dumps(dict_return)

    my_res = flask.Response(json_val)
    return my_res
