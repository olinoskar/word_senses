import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

A = [0.12, 0.046, 0.22, 0, 0, 0, 0.15, 0.0031, 0.019, 0.029, 0.067, 0.3, 0.0003, 0.0068, 0.048]
B = [0.084, 0.93, 0.95, 1, 0.93, 0.61, 0.018, 0.69, 1, 0.93, 1, 0.51, 0.59, 0.85, 0.91, 0.62, 0.042, 0.31,
     0.086, 0.17,0.01, 0.44, 0.15, 0.31, 0.47, 0.54, 0.2, 0.047, 0.033, 0.085, 0.42, 0.24, 0.24, 0.055, 0.18,
    1, 0.93, 0.12, 0.052, 0.067, 0.1, 0.012, 0.11, 0.11, 0.31, 0.062, 1, 0.58, 0.13, 1, 0.076, 1, 1,
    0.133, 0.124, 0.044, 0.068, 0.79, 0.74, 0.017, 0.96, 0.92, 0.9, 0.036, 0.88, 1, 0.95]

A = 1-np.array(A)
B = 1-np.array(B)

fig, ax = plt.subplots()
sns.set_style("white")
sns.kdeplot(A, ax=ax, shade=True, color = 'red', clip=(0,1))
sns.kdeplot(B, ax=ax, shade=True, color='blue', clip=(0,1))
plt.legend(labels=["Don't reject $H_0$", "Reject $H_0$"], fontsize=17)
sns.set_style("white")
fig.savefig('kde_plot.png')

dff = pd.DataFrame(columns=['class', 'value'])
for b in B:
    dff.loc[len(dff)] = ["Reject $H_0$", b]
for a in A:
    dff.loc[len(dff)] = ["Don't reject $H_0$", a]

#sns.set(font_scale=3.0)  # crazy big
fig, ax = plt.subplots()
sns.set_style("white")

plot = sns.boxplot(x="class", y="value", data=dff)
ax.set(xlabel='', ylabel='')
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(17)

mybox_red = plot.artists[0]
mybox_blue = plot.artists[1]

# Change the appearance of that box
mybox_red.set_facecolor('blue')
mybox_red.set_alpha(0.375)
mybox_blue.set_facecolor('red')
mybox_blue.set_alpha(0.375)

fig.savefig('box_plot.png')







