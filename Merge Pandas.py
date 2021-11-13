# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 18:08:49 2021

@author: schue
"""

import pandas as pd
import numpy as np

Test1 = pd.DataFrame(np.array([[1, "a"], [3, "b"], [5, "c"]]), columns=['Cumsum', 'y'])
Test2 = pd.DataFrame(np.array([[2, "a"], [4, "b"], [6, "c"]]), columns=['Cumsum', 'y'])


# Result = Test1.merge(Test2, how="outer", sort = True)

# Result = pd.concat([Test1, Test2], axis=1)


Test1 = Test1.set_index("Cumsum")
Test2 = Test2.set_index("Cumsum")
Result = Test1.join(Test2, how="outer", lsuffix="1")
# Test1.set_index('Cumsum').join(Test2.set_index('Cumsum'))