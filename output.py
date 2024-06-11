import matplotlib.pyplot as plt
import numpy as np

def read_scores(filename):
    with open(filename, 'r') as file:
        scores = [int(line.split(': ')[1]) for line in file if line.startswith('Score:')]
    return scores

def fit_func(x, a, b, c):
    return a * np.exp(b * x) + c

filename = 'scores.txt'
scores = read_scores(filename)

x_data = np.arange(1, len(scores) + 1)
y_data = np.array(scores)

plt.scatter(x_data, y_data, label='Data Points')

coefficients = np.polyfit(x_data, y_data, 3)
poly_func = np.poly1d(coefficients)

x_fit = np.linspace(1, len(scores), 100)
y_fit = poly_func(x_fit)
plt.plot(x_fit, y_fit, color='red', label='Fitted Curve')

plt.title('Scores Scatter Plot with Fitted Curve')
plt.xlabel('Index')
plt.ylabel('Score')
plt.legend()
plt.show()
