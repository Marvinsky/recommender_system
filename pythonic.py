
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ex = pd.DataFrame({
    "Mozart" : [0, 5, 5, 5, 3, 2],
    "Bach" :   [1, 0, 4, 5, 3, 2], 
    "Chopin"  : [2, 3, 0, 5, 4, 1],
    "Brahms" : [2, 2, 5, 0, 3, 1],
    "Wagner" : [1, 3, 3, 2, 0, 3],
    "Liszt" :  [3, 3, 2, 1, 2, 0]
}, index = ["Abel", "Baker", "Charlie", "David", "Erik", "Frank"])

e_sim = lambda ser1, ser2: 1/(1+np.linalg.norm(ser1-ser2))
p_sim = lambda ser1, ser2: .5 + (np.corrcoef(ser1, ser2)[0][1])/2
c_sim = lambda ser1, ser2: .5 + (np.dot(ser1, ser2)/(np.linalg.norm(ser1)*np.linalg.norm(ser2)))/2

print("e_sim = ", e_sim(ex.iloc[:3,3], ex.iloc[:3,4]))
print("p_sim = ", p_sim(ex.iloc[:3,3], ex.iloc[:3,4]))
print("c_sim = ", c_sim(ex.iloc[:3,3], ex.iloc[:3,4]))

drop_rows_with_zeros = lambda ex: ex.iloc[np.intersect1d(ex.iloc[:,0].to_numpy().nonzero(), ex.iloc[:,1].to_numpy().nonzero()),:]

df = drop_rows_with_zeros(ex)
print(df.head())


