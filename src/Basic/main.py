# -*- coding: utf-8 -*-
""" main.py """

from configs.module.config import CFG
from models.ModelName import ModelName


def run():
    """Builds model, loads data, trains and evaluates"""
    model = ModelName(CFG)
    model.load_data()
    model.build()
    #model.train()
    model.evaluate()


if __name__ == '__main__':
    run()