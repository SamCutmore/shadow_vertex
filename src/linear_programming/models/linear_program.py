from dataclasses import dataclass, field
from fractions import Fraction
from enum import Enum
from typing import Dict, List, Optional, Tuple

class ObjectiveExtremum(Enum):
    MINIMUM = "min"
    MAXIMUM = "max"

class ConstraintRelation(Enum):
    LE = "<="
    GE = ">="
    EQ = "="

@dataclass
class Variable:
    name: str
    lb: Optional[Fraction] = None
    ub: Optional[Fraction] = None

@dataclass
class Constraint:
    coeffs: Dict[str, Fraction]
    sense: ConstraintRelation
    rhs: Fraction

@dataclass
class LinearProgram:
    sense: ObjectiveExtremum = ObjectiveExtremum.MINIMUM
    variables: Dict[str, Variable] = field(default_factory=dict)
    constraints: List[Constraint] = field(default_factory=list)
    objective: Dict[str, Fraction] = field(default_factory=dict)

    def add_var(self, name: str, lb: Optional[float] = None, ub: Optional[float] = None) -> Variable:
        if name not in self.variables:
            if lb is not None:
                lb=Fraction(lb)
            if ub is not None:
                ub=Fraction(ub)
            self.variables[name] = Variable(name=name, lb=lb, ub = ub)
                
        return self.variables[name]

    def set_objective(self, coeffs: Dict[str, float], sense: ObjectiveExtremum = ObjectiveExtremum.MINIMUM) -> None:
        for v, c in coeffs.items():
            self.add_var(v)
        self.sense = sense
        self.objective = {v: Fraction(c) for v, c in coeffs.items()}

    def add_constraint(self, coeffs: Dict[str, float], sense: ConstraintRelation, rhs: float) -> None:
        for v in coeffs:
            self.add_var(v)
        self.constraints.append(
            Constraint(
                coeffs={v: Fraction(c) for v, c in coeffs.items()},
                sense=sense,
                rhs=Fraction(rhs)
            )
        )

    def build_sparse_matrix(self) -> Tuple[List[int], List[int], List[Fraction]]:
        row_indices, column_indices, coefficients = [], [], []
        var_names = list(self.variables.keys())
        var_index = {v: i + 1 for i, v in enumerate(var_names)}
        for i, cons in enumerate(self.constraints, start=1):
            for v, c in cons.coeffs.items():
                row_indices.append(i)
                column_indices.append(var_index[v])
                coefficients.append(c)
        return row_indices, column_indices, coefficients

    def _str_(self) -> str:
        sense_str = "minimize" if self.sense == ObjectiveExtremum.MINIMUM else "maximize"
        out = [f"Objective ({sense_str}): " +
               " + ".join(f"{c}*{v}" for v, c in self.objective.items())]
        for i, c in enumerate(self.constraints, start=1):
            sym = {"LE": "<=", "GE": ">=", "EQ": "="}[c.sense.name]
            expr = " + ".join(f"{coef}*{v}" for v, coef in c.coeffs.items())
            out.append(f"C{i}: {expr} {sym} {c.rhs}")
        return "\n".join(out)
    
lp = LinearProgram()

lp.set_objective({"x1": 2, "x2": 3}, sense=ObjectiveExtremum.MAXIMUM)

lp.add_constraint({"x1": 1, "x2": 2}, ConstraintRelation.LE, 5)
lp.add_constraint({"x1": 3, "x2": 1}, ConstraintRelation.EQ, 8)

print(lp._str_())

row_indices, column_indices, coefficients = lp.build_sparse_matrix()
print("\nSparse matrix (row_indices, column_indices, coefficients):")
for t in zip(row_indices, column_indices, coefficients):
    print(t)