import matplotlib.pyplot as plot
import numpy as np
import random

w1ActualValue = 4
w2ActualValue = 6 
actualBias = 232

x1Values = np.array([])
x2Values = np.array([])
yValues = np.array([])

for a in range(10):
    x1Values = np.append(x1Values, random.randint(1, 10))
    x2Values = np.append(x2Values, random.randint(1, 10))

for a in range(10):
    yValues = np.append(yValues, ((w1ActualValue * x1Values[a]) + (w2ActualValue * x2Values[a]) + actualBias))

def gradientDescent(x1Values, x2Values, yValues, w1, w2, b, learningRate, passes):
    m = len(yValues)
    xMatrix = np.vstack([x1Values, x2Values]).T
    yMatrix = yValues
    for abc in range(passes):
        predictions = np.dot(xMatrix, np.array([w1, w2])) + b
        costs = predictions - yMatrix
        w1PartialDerivative = (1/m) * np.dot(costs, x1Values)
        w2PartialDerivative = (1/m) * np.dot(costs, x2Values)
        bPartialDerivative = (1/m) * np.sum(costs)
        w1 -= learningRate * w1PartialDerivative
        w2 -= learningRate * w2PartialDerivative
        b -= learningRate * bPartialDerivative
    return w1, w2, b

w1 = 0
w2 = 0
b = 0
learningRate = 0.01
passes = 10000

w1, w2, b = gradientDescent(x1Values, x2Values, yValues, w1, w2, b, learningRate, passes)


print(w1, w2, b)
