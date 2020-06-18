from ortools.sat.python import cp_model


model = cp_model.CpModel()


cp_model.NewIntVar(0,100)