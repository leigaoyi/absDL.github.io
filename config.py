import json
import os
from bunch import Bunch

def get_config_from_json(json_file):
    """
    将配置文件转换为配置类
    change json file to dictionary
    """
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)  # 配置字典

    config = Bunch(config_dict)  # 将配置字典转换为类

    return config, config_dict

args, _ = get_config_from_json('./config/train.json')