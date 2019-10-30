import pybamm 
import numpy as np
import matplotlib.pyplot as plt

#1. initialise an empty model
model = pybamm.BaseModel()

#2.Define vars
x = pybamm.Variable("x") #object
y1 = pybamm.Variable("y1")

#3.State governing equations
model.rhs={x: 4*x-2*y1, y1: 3*x-y1} #creates dictionary with dx/dt and dy/dt. Index dictionary with dict["key"]


#4.State ICs
model.initial_conditions = {x: 1, y1:2}

#5.Define outputs
model.variables={"x*y1": x*y1, "x":x, "y1":y1} #dictionary of output variables

#6. Descritize
disc=pybamm.Discretisation()
disc.process_model(model)

#7. Solve the ODE
solver = pybamm.ScipySolver() #basic solver for ODEs (class)
t_eval = np.linspace(0,1,100)   #evaluation 
solution = solver.solve(model, t_eval)

print(solution.t)
print(solution.y)
myOutput=model.variables["x*y1"].evaluate(solution.t, solution.y) #solution.y is NOT the same as the variable y1, it's the states
print(myOutput)

#8. Plotting