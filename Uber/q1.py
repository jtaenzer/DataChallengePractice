import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

'''
1. Perform any cleaning, exploratory analysis, and/or visualizations to use the provided data for this
analysis (a few sentences/plots describing your approach will suffice). What fraction of the driver
signups took a first trip?
'''

pd.options.display.width = 0
data = pd.read_csv("./data/data.csv", sep=",")
# Convert first_completed_date column to boolean column 'started_driving'
# Use this to calculate fraction of driver signups that took a first trip
data['started_driving'] = np.where(~data['first_completed_date'].isnull(), 1, 0)
drive_frac = np.sum(data['started_driving'])/len(data['started_driving'])
consent_frac = np.sum(data['started_driving'][~data['bgc_date'].isnull()])/len(data['started_driving'][~data['bgc_date'].isnull()])
print("Fraction of driver signups that took a first trip: {:.2f}".format(drive_frac))
print("Fraction of background check consenters that took a first trip: {:.2f}".format(consent_frac))


# Create correlation matrix for signup_os vs. started_driving
# No strong correlation between the signup OS and whether a first trip is taken
countvec = CountVectorizer()
signup_os_df = pd.DataFrame(countvec.fit_transform(data['signup_os'][~data['signup_os'].isnull()].tolist()).toarray(),
                            columns=countvec.vocabulary_)
signup_os_df['started_driving'] = np.where(data['started_driving'][~data['signup_os'].isnull()] == True, 1, 0)
corr_matrix = signup_os_df.corr()
sn.heatmap(corr_matrix, annot=True)
plt.savefig("./plots/signup_os_corr.png")
plt.close()

# Create correlation matrix for signup_channel vs started_driving
# There is some predictive power here, referral is correlated and organic is anti correlated to started_driving
signup_channel_df = pd.DataFrame(countvec.fit_transform(data['signup_channel'][~data['signup_channel'].isnull()].tolist()).toarray(),
                                 columns=countvec.vocabulary_)
signup_channel_df['started_driving'] = np.where(data['started_driving'][~data['signup_channel'].isnull()] == True, 1, 0)
corr_matrix = signup_channel_df.corr()
sn.heatmap(corr_matrix, annot=True)
plt.savefig("./plots/signup_channel_corr.png")
plt.close()

# Create correlation matrix bgc_date (exists) vs started_driving
# Consenting to a background check is correlated to started_driving
data['bgc_bool'] = ~data['bgc_date'].isnull()
corr_matrix = data[['bgc_bool', 'started_driving']].corr()
sn.heatmap(corr_matrix, annot=True)
plt.savefig("./plots/bgc_bool_corr.png")
plt.close()

# Create correlation matrix for time delay between signing up and consenting to a background check and started_driving
# Delay in consenting to background check is anti-correlated to started_driving
data['bgc_delay'] = (pd.to_datetime(data['bgc_date']) - pd.to_datetime(data['signup_date'])).dt.days
corr_matrix = data[['bgc_delay', 'started_driving']].corr()
sn.heatmap(corr_matrix, annot=True)
plt.savefig("./plots/bgc_delay_corr.png")
plt.close()
# Make a hist of the bgc_delay
plt.bar(range(len(data['bgc_delay'].value_counts())), data['bgc_delay'].value_counts(), align='center')
plt.title("Background check consent delay histogram")
plt.xlabel("Days")
plt.ylabel("Count")
plt.savefig("./plots/bgc_delay_hist.png")
plt.close()

# Create correlation matrix vehicle_info (exists) vs started_driving
# Adding vehicle info is strongly correlated to started_driving
data['vehicle_info_bool'] = np.where(~data['vehicle_added_date'].isnull(), 1, 0)
corr_matrix = data[['vehicle_info_bool', 'started_driving']].corr()
sn.heatmap(corr_matrix, annot=True)
plt.savefig("./plots/vehicle_info_corr.png")
plt.close()

data['vehicle_info_delay'] = (pd.to_datetime(data['vehicle_added_date']) - pd.to_datetime(data['signup_date'])).dt.days
corr_matrix = data[['vehicle_info_delay', 'started_driving']].corr()
sn.heatmap(corr_matrix, annot=True)
plt.savefig("./plots/vehicle_added_delay_corr.png")
plt.close()
# Make a hist of the vehicle_added_delay
plt.bar(range(len(data['vehicle_info_delay'].value_counts())), data['vehicle_info_delay'].value_counts(), align='center')
plt.title("Vehicle information delay histogram")
plt.xlabel("Days")
plt.ylabel("Count")
plt.savefig("./plots/vehicle_info_delay_hist.png")
plt.close()


data['bgc_delay'] = (pd.to_datetime(data['bgc_date']) - pd.to_datetime(data['signup_date'])).dt.days
data['vehicle_info_delay'] = (pd.to_datetime(data['vehicle_added_date']) - pd.to_datetime(data['signup_date'])).dt.days

fig, ax = plt.subplots()
ax.scatter(data['bgc_delay'][data['started_driving'] == False],
           data['vehicle_info_delay'][data['started_driving'] == False],
           c="r", label="no first trip")
ax.scatter(data['bgc_delay'][data['started_driving'] == True],
           data['vehicle_info_delay'][data['started_driving'] == True],
           c="g", label="first trip")
ax.legend()
plt.xlabel("bgc_delay")
plt.ylabel("vehicle_info_delay")
plt.savefig("./plots/delay_scatter.png")
plt.close()

