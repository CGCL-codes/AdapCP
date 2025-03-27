from pulp import *

def solve_ilp(n, B_e, Q_e, Q_s, f, O):
    # Create the problem instance
    prob = LpProblem("CNN_Split_Scheduling", LpMinimize)

    # Define variables
    x = LpVariable.dicts("x", [(i, j) for i in range(1, n+1) for j in range(i, n+1)], cat='Binary')
    y = LpVariable.dicts("y", [(i, j) for i in range(1, n+1) for j in range(i, n+1)], cat='Binary')
    v = LpVariable.dicts("v", [(i, j) for i in range(1, n+1) for j in range(i, n+1)], lowBound=0, upBound=1, cat='Continuous')
    z = LpVariable.dicts("z", [(i, j) for i in range(1, n+1) for j in range(i, n+1)], lowBound=0, upBound=1, cat='Continuous')

    # Constraint (8): Each layer is covered exactly once
    for r in range(1, n+1):
        variables_in_r = []
        for i in range(1, r+1):
            for j in range(max(i, r), n+1):
                if (i, j) in x:
                    variables_in_r.append(x[(i, j)])
                if (i, j) in y:
                    variables_in_r.append(y[(i, j)])
        prob += lpSum(variables_in_r) == 1

    # Constraints (11) for v and z
    for i in range(1, n+1):
        for j in range(i, n+1):
            # v constraints
            prob += v[(i, j)] <= x[(i, j)]
            if j + 1 <= n:
                prob += v[(i, j)] <= lpSum(y[(j+1, k)] for k in range(j+1, n+1))
            else:
                prob += v[(i, j)] <= 0  # j+1 exceeds n, sum is 0

            # z constraints
            prob += z[(i, j)] <= y[(i, j)]
            if j + 1 <= n:
                prob += z[(i, j)] <= lpSum(x[(j+1, k)] for k in range(j+1, n+1))
            else:
                prob += z[(i, j)] <= 0  # j+1 exceeds n, sum is 0

    # Precompute sum of FLOPs for each partition (i, j)
    sum_flops = {}
    for i in range(1, n+1):
        for j in range(i, n+1):
            sum_flops[(i, j)] = sum(f[i-1:j])  # f is 0-based

    # Compute T_comp
    T_comp = lpSum([x[(i, j)] * (sum_flops[(i, j)] / Q_e) + y[(i, j)] * (sum_flops[(i, j)] / Q_s)
                   for i in range(1, n+1) for j in range(i, n+1)])

    # Compute T_trans (assuming O is 0-based index for layers)
    T_trans_upload = lpSum([v[(i, j)] * (O[j-1] / B_e) for i in range(1, n+1) for j in range(i, n+1)])
    T_trans_download = lpSum([z[(i, j)] * (O[j-1] / B_e) for i in range(1, n+1) for j in range(i, n+1)])
    T_trans = T_trans_upload + T_trans_download

    # Set objective function
    prob += T_comp + T_trans

    # Solve the problem
    prob.solve()

    # Output results
    print("Status:", LpStatus[prob.status])
    if LpStatus[prob.status] == 'Optimal':
        print("Optimal Total Time:", value(prob.objective))
        print("\nVariable values:")
        for var in prob.variables():
            if var.varValue() > 0:
                print(f"{var.name}: {var.varValue()}")
    else:
        print("No optimal solution found.")

# Example usage
if __name__ == "__main__":
    # Example parameters
    n = 3  # Number of layers
    B_e = 10  # Bandwidth in Mbps
    Q_e = 1000  # End device computing power (FLOPs/sec)
    Q_s = 5000  # Total server computing power (FLOPs/sec)
    f = [100, 200, 300]  # FLOPs per layer (0-based)
    O = [10, 20, 30]     # Output data per layer (0-based, in MB)

    solve_ilp(n, B_e, Q_e, Q_s, f, O)