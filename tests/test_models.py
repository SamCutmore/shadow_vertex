from fractions import Fraction



frac: Fraction = Fraction(8, 6)
result = frac.as_integer_ratio()

print(f"Result: {result}")
