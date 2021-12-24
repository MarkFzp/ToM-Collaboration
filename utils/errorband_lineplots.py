'''
Timeseries plot with error bands
================================
'''
import sys
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

paper_rc = {'lines.linewidth': 2.5}
sns.set(style="darkgrid")
sns.set(font_scale=1.5, rc = paper_rc)
filename = sys.argv[1]
# Load an example dataset with long-form data
results = pd.read_csv('%s.csv' % filename)

# Plot the responses for different events and regions
sns.lineplot(x="Training_Progress", y="Accuracy",
             hue="Method", data=results)

plt.xticks([0.1 * i for i in range(11)])
plt.yticks([0.1 * i for i in range(11)])
#plt.legend(loc='upper left')
plt.show()
