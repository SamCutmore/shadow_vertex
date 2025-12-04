import numpy as np

class RationalTableau:
    def __init__(self, num_tableau, den_tableau=None):
        """
        num_tableau: int64 numpy array of numerators
        den_tableau: int64 numpy array of denominators (optional, defaults to 1s)
        """
        self.num = num_tableau.astype(np.int64)
        if den_tableau is None:
            self.den = np.ones_like(self.num, dtype=np.int64)
        else:
            self.den = den_tableau.astype(np.int64)
        self.rows, self.cols = self.num.shape

    def normalize_row(self, i):
        """Reduce row by GCD of numerators and denominators to avoid overflow"""
        row_gcd = np.gcd.reduce(np.concatenate((self.num[i], self.den[i])))
        if row_gcd > 1:
            self.num[i] //= row_gcd
            self.den[i] //= row_gcd
            
    def find_pivot_column(self):
        """Return index of most negative reduced cost in objective row."""
        objective = self.num[-1, :-1] / self.den[-1, :-1]
    
        # If all >= 0 → optimal
        if np.all(objective >= 0):
            return None
    
        index = int(np.argmin(objective))
        print(f"Pivot row selection: {index}")
        return int(np.argmin(objective))

    def find_leaving_row(self, pivot_col):
        """Minimum ratio test"""
    
        col_num = self.num[:-1, pivot_col]
        col_den = self.den[:-1, pivot_col]
        rhs_num = self.num[:-1, -1]
        rhs_den = self.den[:-1, -1]
    
        # Compute ratios = (rhs_num / rhs_den) / (col_num / col_den)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = (rhs_num * col_den) / (rhs_den * col_num)
    
        # Disallow non-positive pivot column entries
        ratios[col_num <= 0] = np.inf
    
        # Detect unboundedness
        if np.all(np.isinf(ratios)):
            return None
        
        index = int(np.argmin(ratios))
        print(f"row index selection: {index}")
        return int(np.argmin(ratios))


    def pivot(self, pivot_row, pivot_col):
        """Integer pivot using numerator/denominator arrays"""
        
        return

    def solve(self, max_iterations=50):
        for it in range(max_iterations):
            pivot_col = self.find_entering_column()
            if pivot_col is None:
                break # Optimal
            pivot_row = self.find_leaving_row(pivot_col)
            if pivot_row is None:
                raise ValueError("LP is unbounded")
            self.pivot(pivot_row, pivot_col)
            self.print_tableau()
        return self.num, self.den

    def print_tableau(self):
        for i in range(self.rows):
            row = [f"{self.num[i,j]}/{self.den[i,j]}" for j in range(self.cols)]
            print(" | ".join(row))
        print("-"*40)

tableau = np.array([
    [1, 2, 1, 0, 8],
    [3, 1, 0, 1, 9],
    [-3, -4, 0, 0, 0]
], dtype=np.int64)

solver = RationalTableau(tableau)
print("Initial tableau:")
solver.print_tableau()
column = solver.find_pivot_column()
solver.find_leaving_row(column)
