import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.options.display.width = 0
data = pd.read_csv("./data/data.csv", sep=",")

data['started_driving'] = ~data['first_completed_date'].isnull()

print(data.describe(include='all'))

