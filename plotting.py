#Build with assistance of GPT 5.5
#The code was reviewed, debugged, and verified by the student

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import csv

#initiation and plotting
x = []
y_tsw = [] #tokens/s/
y_error=[]
err=[]
#The .csv file is exported from Google Doc and given in the form of:
"""Frequency,Median power,Absolute Uncertainty,Percentage Uncertaninty,Tokens/s,Absolute Uncertainty,
   Percentage Uncertianty,Tokens/s/W,Percentage Uncertainty,Absolute Uncertainty"""

with open("data_summary.csv", "r") as f:
    data = csv.reader(f)
    for row in data:
        x.append(float(row[0]))
        y_tsw.append(float(row[7]))
        err.append(float(row[9]))
y_error = [err, err]

x_points = np.array(x)
y_points_tsw = np.array(y_tsw)
ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(60))
plt.plot(x_points, y_points_tsw, "o", color = "orange")
plt.ylabel("tokens·s⁻¹·W⁻¹")
plt.xlabel("Frequency/MHz")
plt.title("Energy Efficiency Graph of RTX 3080")
plt.errorbar(
    x,
    y_tsw,
    yerr=y_error,
    fmt="o"
)

#curve of best fit
coef = np.polyfit(x_points, y_points_tsw, deg=5)
poly = np.poly1d(coef)
x_fit = np.linspace(x_points.min(), x_points.max(), 500)
y_fit = poly(x_fit)
plt.plot(x_fit, y_fit, color="black")

#R^2 value
residuals = y_points_tsw - poly(x_points) #difference between predicted y value and acutal y value of each plotted point
ss_res = np.sum(residuals**2) #Sum of the square of residuals
ss_tot = np.sum((y_points_tsw - np.mean(y_points_tsw))**2) #Total sum of squared residuals
r_squared = 1 - (ss_res / ss_tot) #by definition, R² = 1 - (total sum of residuals/total sum of squared residuals)
print(f"R² = {r_squared:.4f}")

#Annotate the frequency with highest energy efficiency
max_idx = np.argmax(y_tsw)
plt.scatter(x[max_idx], y_tsw[max_idx])
plt.annotate(
    f"Max tokens·s⁻¹·W⁻¹: {y_tsw[max_idx]} at {x[max_idx]} MHz",
    (x[max_idx], y_tsw[max_idx]),
    xytext=(10, 10),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-")
)

plt.show()
