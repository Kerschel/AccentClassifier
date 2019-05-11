import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler

d = np.random.randn(3,3,3)
d -= np.min(d, axis=0)
d /= np.ptp(d, axis=0)
mms = MinMaxScaler()
mms.fit_transform(np.abs(d))
print (d)
