# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 18:08:49 2021

@author: schue
"""

import pandas as pd
import numpy as np

Test1 = pd.DataFrame(np.array([[1, "a"], [3, "b"], [5, "c"]]), columns=['x', 'y'])
Test2 = pd.DataFrame(np.array([[2, "a"], [4, "b"], [6, "c"]]), columns=['x', 'y'])


Result = Test1.merge(Test2, how="outer")
# Result = Test1.merge(Test2, on = "x", how= "cross")
# Result = Result.sort_values("x")