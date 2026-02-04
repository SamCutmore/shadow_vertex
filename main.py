import linprog_core
import time

def benchmark_simplex_klee_minty(n=14):
    # Setup Klee-Minty 
    obj = [2**(n - (i + 1)) for i in range(n)]
    prob = linprog_core.PyProblem(obj, goal="max")
    
    for i in range(1, n + 1):
        coeffs = [2**(i - j + 1) for j in range(1, i)] + [1]
        coeffs.extend([0] * (n - len(coeffs)))
        prob.add_constraint(coeffs, "<=", 5**(i-1))

    # Standard simplex
    solver = linprog_core.PySimplexSolver()

    print(prob)
    
    print(f"--- Benchmarking Standard Simplex ({n}D) ---")
    print(f"Expected path length: {2**n} pivots")
    
    start_time = time.time()
    # solve_with_history
    solution, history = solver.solve_with_history(prob)
    end_time = time.time()
    
    elapsed = end_time - start_time
        
    print("-" * 40)
    print(f"Total Pivots:  {len(history)}")
    print(f"Time Taken:    {elapsed:.6f} seconds")
    print(f"Pivots/Sec:    {len(history)/elapsed:.2f}")
    print(f"Final Status:  {solution.status}")
    print(f"Final Obj:     {solution.objective}")

def run_two_phase_klee_minty_differences(start=3, stop=10):
    pivot_counts = []

    for n in range(start, stop + 1):
        # Set up Klee-Minty problem
        obj = [2**(n - (i + 1)) for i in range(n)]
        prob = linprog_core.PyProblem(obj, goal="max")
        for i in range(1, n + 1):
            coeffs = [2**(i - j + 1) for j in range(1, i)] + [1]
            coeffs.extend([0] * (n - len(coeffs)))
            prob.add_constraint(coeffs, "<=", 5**(i-1))

        solver = linprog_core.PyTwoPhaseSimplexSolver()

        print(f"--- Running {n}D Shadow Vertex ---")
        start_time = time.time()
        solution, history = solver.solve_with_history(prob)
        end_time = time.time()

        elapsed = end_time - start_time
        pivots = len(history)
        pivot_counts.append(pivots)

        print("-" * 40)
        print(f"Total Pivots:  {pivots}")
        print(f"Time Taken:    {elapsed:.6f} seconds")
        print(f"Pivots/Sec:    {pivots/elapsed:.2f}")
        print(f"Final Status:  {solution.status}")
        print(f"Final Obj:     {solution.objective}")

    # Compute differences between consecutive pivot counts
    print("\nPivot Counts & Differences:")
    print(f"{'n':>2} | {'Pivots':>10} | {'Diff':>10}")
    print("-" * 30)
    for i, pivots in enumerate(pivot_counts):
        n_dim = start + i
        diff = pivot_counts[i] - pivot_counts[i-1] if i > 0 else "-"
        print(f"{n_dim:>2} | {pivots:>10} | {diff:>10}")
    

if __name__ == "__main__":
    run_two_phase_klee_minty_differences(start=3, stop=15)
    benchmark_simplex_klee_minty(n=10)