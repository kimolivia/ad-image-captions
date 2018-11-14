""" with open('twitter.pw', 'r') as f:
    twitter = f.read()

print(twitter, type(twitter), len(twitter))
t = twitter.split('\n')
print(t, len(t)) """


#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

x = np.linspace(0, 20, 100)
plt.plot(x, np.sin(x))
plt.show() 
