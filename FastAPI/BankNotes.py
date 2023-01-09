# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:51:19 2020
@author: Cecile Lecoublet
"""
from pydantic import BaseModel
# 2. Class which describes Bank Notes measurements
class BankNote(BaseModel):
    SK_ID_CURR : float
