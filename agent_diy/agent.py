#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from common_python.utils.common_func import create_cls
import numpy as np
import os
from kaiwudrl.interface.agent import BaseAgent
from agent_diy.conf.conf import Config

ObsData = create_cls("ObsData", feature=None)
ActData = create_cls("ActData", act=None)


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
        super().__init__(agent_type, device, logger, monitor)

    def predict(self, list_obs_data):
        pass

    def exploit(self, list_obs_data):
        pass

    def learn(self, list_sample_data):
        pass

    def observation_process(self, raw_obs):
        return ObsData(feature=int(raw_obs[0]))

    def action_process(self, act_data):
        return act_data.act

    def save_model(self, path=None, id="1"):
        pass

    def load_model(self, path=None, id="1"):
        pass
