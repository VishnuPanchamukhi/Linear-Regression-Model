import matplotlib.pyplot as plot
import random   

# generate training data

xValues = [i * 0.1 for i in range(100)]
trueSlope = 5
trueIntercept = 1.0
yValues = [x * trueSlope + trueIntercept + random.uniform(-3, 3) for x in xValues]

# squared error cost function 
def calculateCost(xValues, yValues, w, b):
    totalCost = 0 
    # num of data points
    m = len(yValues)
    for i in range(m):
        prediction = w * xValues[i] + b
        totalCost += (yValues[i] - prediction) ** 2
    return totalCost / (2 * m)

# gradient descent function
def gradientDescent(xValues, yValues, w, b, learningRate, passes):
    m = len(xValues)
    for abc in range(passes):
        # values for partial derivatives
        wPartialDerivative = 0
        bPartialDerivative = 0
        for i in range(m):
            prediction = w * xValues[i] + b
            cost = prediction - yValues[i]
            wPartialDerivative += cost * xValues[i]
            bPartialDerivative += cost
        wPartialDerivative /= m
        bPartialDerivative /= m
        # update w and b simulataneously
        w -= learningRate * wPartialDerivative
        b -= learningRate * bPartialDerivative
    return w, b

# starting guesses for w and b
w = 0
b = 0
learningRate = 0.01
passes = 1000

# find starting cost
initialCost = calculateCost(xValues, yValues, w, b)
print("Initial cost: " + str(round(initialCost, 2)))

# perform gradient descent
w, b = gradientDescent(xValues, yValues, w, b, learningRate, passes)

# final cost
finalCost = calculateCost(xValues, yValues, w, b)
print("Final cost: " + str(round(finalCost, 2)))

# plot training data
plot.figure(figsize=(10, 6))
plot.scatter(xValues, yValues, label="Training Data", color="blue")

# plot calculate line of best fit from gradient descent
bestFitLine = [w * x + b for x in xValues]
plot.plot(xValues, bestFitLine, label="Best Fit Line: y = " + str(round(w, 2)) + "x + " + str(round(b, 2)), color="red")

# labels + titles
plot.xlabel("x")
plot.ylabel("y")
plot.title("Linear Regression using Gradient Descent")
plot.legend()
plot.show()
