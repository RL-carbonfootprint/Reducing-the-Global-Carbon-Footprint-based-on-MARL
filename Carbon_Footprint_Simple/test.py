'''
from scipy.stats import skewnorm
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)

a=0.9
data= skewnorm.rvs(a, size=10)

plt.hist(data, density=True, histtype='stepfilled', alpha=0.2)
plt.legend(loc='best', frameon=False)
plt.show()
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

NUM_SAMPLES = 100000
SKEW_PARAMS = [0.9, 0]


# literal adaption from:
# http://stackoverflow.com/questions/4643285/how-to-generate-random-numbers-that-follow-skew-normal-distribution-in-matlab
# original at:
# http://www.ozgrid.com/forum/showthread.php?t=108175
def rand_skew_norm(fAlpha, fLocation, fScale):
    sigma = fAlpha / np.sqrt(1.0 + fAlpha**2) 

    afRN = np.random.randn(2)
    u0 = afRN[0]
    v = afRN[1]
    u1 = sigma*u0 + np.sqrt(1.0 -sigma**2) * v 

    if u0 >= 0:
        return u1*fScale + fLocation 
    return (-u1)*fScale + fLocation 

def randn_skew(N, skew):
    return [rand_skew_norm(3, skew, 0.025) for x in range(N)]
'''
# lets check they at least visually match the PDF:
plt.subplots(figsize=(12,4))
for alpha_skew in SKEW_PARAMS:
    p = randn_skew(100, alpha_skew)
    sns.distplot(p)
plt.show()
'''
p = np.random.choice(randn_skew(100, 0.9), size=1)[0]

print(p)