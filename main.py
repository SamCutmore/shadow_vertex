import linprog_core as lp

# Objective: Max 3x + 2/3y
prob = lp.PyProblem([(3, 1), (2, 3)], goal="max")

# Add Constraint: 1/2x + 1y <= 4/1
prob.add_constraint([(1, 2), 1], "<=", (4, 1))

# Add Constraint: 2x + 1y <= 5
prob.add_constraint([2, 1], "<=", 5)

print(prob)

tab = prob.to_tableau()
print(tab)