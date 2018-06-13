from numpy import *

def find_error(b, m, points):
	error = 0
	for i in range(0, len(points)):
		x = points[i, 0]
		y = points[i, 1]
		error += (y - (m*x + b)) ** 2
	return error/float(len(points))

def single_step_gradient(current_b, current_m, points, rate):
	gradient_b = 0
	gradient_m = 0
	N = float(len(points))

	for i in range(0, len(points)):
		x = points[i, 0]
		y = points[i, 1]
		gradient_b += -(2/N) * (y - ((current_m * x) + current_b))
		gradient_m += -(2/N) * x * (y - ((current_m * x) + current_b))
	new_b = current_b - (rate * gradient_b)
	new_m = current_m - (rate * gradient_m)

	return [new_b, new_m]

def gradient_descent(points, starting_b, starting_m, rate, iterations):
	b = starting_b
	m = starting_m

	for i in range(iterations):
		b, m = single_step_gradient(b, m, array(points), rate)
	return [b, m]

def test_model(final_b, final_m):
	testPoints = genfromtxt("test.csv", delimiter=",")
	c = 0
	##print("ACTUAL \t \t PREDICTION")
	for i in range(len(testPoints)):
		prediction = final_m * testPoints[i, 0] + final_b
		actual = testPoints[i, 1]
		##print(actual, "\t", prediction)
		if(abs(prediction - actual) <= 5):
			c += 1
	print("Correct prediction percentage: ", c/float(len(testPoints))*100)

def run():
	points = genfromtxt("train.csv", delimiter=",")
	learning_rate = 0.0001
	num_iterations = 1000
	initial_b = 0
	initial_m = 0
	modelError = find_error(initial_b, initial_m, points)
	print ()
	print ("Starting Gradient Descent with b = {0} and m = {1} for {2} iterations".format(initial_b, initial_b, num_iterations))
	print ("Initial Error: ",modelError)
	[final_b, final_m] = gradient_descent(points, initial_b, initial_m, learning_rate, num_iterations)
	print ("Ending Gradient Descent with final value of b = {0} and m = {1}".format(final_b, final_m))
	modelError = find_error(final_b, final_m, points)
	print ("Final Error: ",modelError)

	test_model(final_b, final_m)

if __name__ == '__main__':
	run()