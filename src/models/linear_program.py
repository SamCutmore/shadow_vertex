from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, list
from enum import enum

class Sense(enum):
    Less_or_equal = "<="
    greater_or_equal = ">="
    equal = "="

class Extrema(enum):
    minimise = "minimise"
    maximise = "maximise"

@dataclass
class Variable:
    name: str
    lower_bound: Fraction = Fraction(0)
    upper_bound: Fraction = Fraction('inf')
    
@dataclass
class Constraint:
    coeffs: Dict[str, Fraction]
    sense: Sense
    rhs: Fraction
    
@dataclass
class Objective:
    coeffs: Dict[str, Fraction]
    extrema: Extrema = Extrema.minimise