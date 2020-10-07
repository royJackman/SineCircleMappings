from pyparsing import OneOrMore, nestedExpr
import pandas as pd
import numpy as np

filedata = ''
with open('chorales.lisp', 'r') as f:
    filedata = f.read()

data = OneOrMore(nestedExpr()).parseString(filedata)
data = data.asList()
retval = []

for chorale in data:
    temp = pd.concat([pd.DataFrame([dict(row).values()], columns=dict(row).keys()) for row in chorale[1:]], ignore_index=True)
    retval.append(temp.to_numpy(dtype='int32'))

with open('bach_chorales.npy', 'wb') as f:
    np.save(f, np.array(retval))