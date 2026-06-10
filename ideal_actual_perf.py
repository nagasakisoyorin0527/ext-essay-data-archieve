import numpy as np
import matplotlib.pyplot as plt
import csv

#initialization and plotting
x = []
y_ideal = []
y_actual = []
coeff_ideal = 855
coeff_actual = 58.01
y_error = []
err = []

with open("data_summary.csv", "r") as f:
    data = csv.reader(f)
    for row in data:
        x.append(int(row[0]))
        y_ideal.append(float(row[0])/coeff_ideal)
        y_actual.append(float(row[4])/coeff_actual)        
        y_error.append(float(row[6]))

for i in range(len(y_error)):
    y_error[i] = y_error[i]*y_actual[i]
err = [y_error, y_error]

x_points  = np.array(x)
y_actual_points = np.array(y_actual)
y_ideal_points = np.array(y_ideal)
plt.ylabel("Normalised Performance (relative to 855 MHz)")
plt.xlabel("Frequency / MHz")
plt.title("Theoretical vs. Actual Normalised Performance of the RTX 3080")



plt.plot(
    x_points,
    y_actual_points,
    "o",
    color = "blue"
)

plt.plot(
    x_points,
    y_ideal_points,
    "o",
    color = "orange"
)

plt.plot(
    [855, 1860],
    [y_ideal[0], y_ideal[-1]],
    color="orange",
    linewidth = 2,
    label="Theoretical Performance"
)

#error bar
plt.errorbar(
    x,
    y_actual,
    yerr=err,
    fmt="o"
)

#best line of fit for the actual performance
coef = np.polyfit(x_points, y_actual_points, deg=5)
poly = np.poly1d(coef)
x_fit = np.linspace(x_points.min(), x_points.max(), 500)
y_fit = poly(x_fit)
plt.plot(x_fit, y_fit, color="blue", label="Actual Performance")

#R^2 value
residuals = y_actual_points - poly(x_points)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y_actual_points - np.mean(y_actual_points))**2) 
r_squared = 1 - (ss_res / ss_tot)
print(f"R² = {r_squared:.4f}")

plt.annotate(f"Normalized performance: {y_ideal[-1]}", [1860.0, y_ideal[-1]], arrowprops=dict(arrowstyle="-"), )
plt.annotate(f"Normalized performance: {y_actual[-1]}", [1860.0, y_actual[-1]])


#labeling the lines
plt.legend()
plt.show()