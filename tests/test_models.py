from linear_programming.models.linear_program import Variable, Constraint, Objective, Sense, Extrema
import unittest
from fractions import Fraction


class TestLPModels(unittest.TestCase):
    
    def test_variable_defaults(self):
        v = Variable(name="x1")
        self.assertEqual(v.name, "x1")
        self.assertEqual(v.lower_bound, Fraction(0))
        self.assertEqual(v.upper_bound, Fraction(0))
    
    def test_variable_custom_bounds(self):
        v = Variable(name="x2", lower_bound=Fraction(1, 2), upper_bound=Fraction(5))
        self.assertEqual(v.lower_bound, Fraction(1, 2))
        self.assertEqual(v.upper_bound, Fraction(5))
    
    def test_constraint_creation(self):
        c = Constraint(coeffs={"x1": Fraction(1), "x2": Fraction(2)},
                       sense=Sense.Less_or_equal,
                       rhs=Fraction(10))
        self.assertEqual(c.coeffs["x2"], Fraction(2))
        self.assertEqual(c.sense, Sense.Less_or_equal)
        self.assertEqual(c.rhs, Fraction(10))
    
    def test_objective_minimise(self):
        obj = Objective(coeffs={"x1": Fraction(3), "x2": Fraction(1)})
        self.assertEqual(obj.extrema, Extrema.minimise)
        self.assertEqual(obj.coeffs["x1"], Fraction(3))
    
    def test_objective_maximise(self):
        obj = Objective(coeffs={"x1": Fraction(1)}, extrema=Extrema.maximise)
        self.assertEqual(obj.extrema, Extrema.maximise)
    
    def test_enum_values(self):
        self.assertEqual(Sense.equal.value, "=")
        self.assertEqual(Extrema.minimise.value, "minimise")
        self.assertTrue(isinstance(Sense.Less_or_equal, Sense))
        self.assertTrue(isinstance(Extrema.maximise, Extrema))


if __name__ == "__main__":
    unittest.main()
