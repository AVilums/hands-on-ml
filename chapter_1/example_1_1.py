import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

data_root: str = "https://github.com/ageron/data/raw/main/"
life_sat = pd.read_csv(data_root + "lifesat/lifesat.csv")
x = life_sat[["GDP per capita (USD)"]].values
y = life_sat[["Life satisfaction"]].values

life_sat.plot(kind='scatter', grid=True,
              x="GDP per capita (USD)", y="Life satisfaction",
              title="Life satisfaction vs GDP per capita (USD)")
plt.axis([23_500, 62_500, 4, 9])
plt.show()

model = LinearRegression()
model.fit(x, y)

x_new = [[33_442.8]]
print(model.predict(x_new))