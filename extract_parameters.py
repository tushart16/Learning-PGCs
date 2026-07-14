import numpy as np 
import itertools
import random
from scipy.optimize import linear_sum_assignment

def preprocess_variables(n, query_oracle, tol=1e-9):
    """
    Algorithm 4: Removes irrelevant variables and identifies all common factors.
    Time Complexity: O(n)
    Query Complexity: 2n + 1
    """
    S_not = []
    S_f = []
    common_parameters = {}
    
    baseline_query = np.ones(n)
    F = query_oracle(baseline_query)
    
    for j in range(n):
        test_query = np.ones(n)
        test_query[j] = 0.0
        F_j = query_oracle(test_query)
        
        if np.isclose(F_j, F, atol=tol):
            S_not.append(j)
        else:
            p_j = F_j / F
            if np.isclose(p_j, 1.0, atol=tol):
                continue 
                
            z_j = -p_j / (1.0 - p_j)
            
            pit_query = np.random.uniform(-100.0, 100.0, size=n)
            pit_query[j] = z_j 
            pit_result = query_oracle(pit_query)
            
            if np.isclose(pit_result, 0.0, atol=tol):
                S_f.append(j)
                common_parameters[j] = p_j
                
    return S_not, S_f, common_parameters


def partition_into_blocks(alive_vars, r):
    """
    Algorithm 3 (Setup): Partitions variables into overlapping blocks.
    Time Complexity: O(n)
    """
    blocks = []
    n_alive = len(alive_vars)
    m = 4 * r + 1
    step_size = 3 * r + 2 
    
    if n_alive <= m:
        blocks.append(alive_vars)
        return blocks
        
    start_idx = 0
    while start_idx + m <= n_alive:
        block = alive_vars[start_idx : start_idx + m]
        blocks.append(block)
        start_idx += step_size
        
    if start_idx < n_alive:
        blocks.append(alive_vars[-m:])
        
    return blocks


def extract_off_diagonal_moments(block, n, query_oracle):
    """
    Populates the strictly off-diagonal elements of M2 and M3.
    Time/Query Complexity: O(m^3) -> effectively O(r^3)
    """
    m = len(block)
    M2 = np.zeros((m, m))
    M3 = np.zeros((m, m, m))
    
    local_to_global = {local_idx: global_idx for local_idx, global_idx in enumerate(block)}
    base_template = np.ones(n)
    
    # Part A: 2nd-Order Off-Diagonal Moments M2 (j != k)
    for j in range(m):
        for k in range(j + 1, m):
            g_j = local_to_global[j]
            g_k = local_to_global[k]
            
            query_vector = base_template.copy()
            query_vector[g_j] = 0.0
            query_vector[g_k] = 0.0
            
            val = query_oracle(query_vector)
            M2[j, k] = val
            M2[k, j] = val
            
    # Part B: 3rd-Order Off-Diagonal Moments M3 (j != k != l)
    for j in range(m):
        for k in range(j + 1, m):
            for l in range(k + 1, m):
                g_j = local_to_global[j]
                g_k = local_to_global[k]
                g_l = local_to_global[l]
                
                query_vector = base_template.copy()
                query_vector[g_j] = 0.0
                query_vector[g_k] = 0.0
                query_vector[g_l] = 0.0
                
                val = query_oracle(query_vector)
                
                M3[j, k, l] = val
                M3[j, l, k] = val
                M3[k, j, l] = val
                M3[k, l, j] = val
                M3[l, j, k] = val
                M3[l, k, j] = val
                
    return M2, M3


def solve_missing_diagonal_M2(M2_partial, r, target_idx):
    """
    O(r^3). 
    Uses random sampling due to Genericity.
    """
    m = M2_partial.shape[0]
    available = [i for i in range(m) if i != target_idx]
    
    max_tries = 100
    for _ in range(max_tries):
        # Genericity guarantees a random disjoint subset will almost always work instantly
        random.shuffle(available)
        A1 = available[:r]
        A2 = available[r:2*r]
        
        rows = [target_idx] + A1
        cols = [target_idx] + A2
        
        S = M2_partial[np.ix_(rows, cols)]
        S_cofactor = S[1:, 1:]
        det_cofactor = np.linalg.det(S_cofactor)
        
        if np.abs(det_cofactor) > 1e-15:
            det_S_with_zero = np.linalg.det(S)
            return -det_S_with_zero / det_cofactor
            
    raise ValueError(f"Genericity failed: Could not find valid rank-{r} minor for M2.")


def fill_all_M2_diagonals(M2_partial, r):
    """
    Time Complexity: O(m * r^3) -> effectively O(r^4)
    """
    m = M2_partial.shape[0]
    M2_completed = M2_partial.copy()
    
    for j in range(m):
        M2_completed[j, j] = solve_missing_diagonal_M2(M2_partial, r, j)
        
    return M2_completed


def solve_missing_entry_M3(M3_partial, r, i_star, j_star, k_star):
    """
    O(r^3) per entry.
    Randomized search exploiting the Zariski-dense non-zero minors.
    """
    m = M3_partial.shape[0]
    forbidden = {i_star, j_star, k_star}
    pool = [x for x in range(m) if x not in forbidden]
    
    max_tries = 100
    for _ in range(max_tries):
        random.shuffle(pool)
        R_other = pool[:r]
        
        # We need 2r indices to form r column pairs, completely disjoint from R_other
        remaining_pool = pool[r:]
        C_other = [(remaining_pool[2*idx], remaining_pool[2*idx+1]) for idx in range(r)]
        
        S = np.zeros((r + 1, r + 1))
        R = [i_star] + R_other
        C = [(j_star, k_star)] + C_other
        
        for r_idx, row_val in enumerate(R):
            for c_idx, (c_val1, c_val2) in enumerate(C):
                S[r_idx, c_idx] = M3_partial[row_val, c_val1, c_val2]
                
        S_cofactor = S[1:, 1:]
        det_cofactor = np.linalg.det(S_cofactor)
        
        if np.abs(det_cofactor) > 1e-15:
            det_S_with_zero = np.linalg.det(S)
            return -det_S_with_zero / det_cofactor
            
    raise ValueError(f"Genericity failed for M3.")

def fill_all_M3_missing_entries(M3_partial, r):
    """
    Fills M3 sequentially. Phase 1 (two equal indices), then Phase 2 (all three equal).
    Time Complexity: O(m^2 * r^3) -> effectively O(r^5)
    """
    m = M3_partial.shape[0]
    M3_completed = M3_partial.copy()
    
    # Entries with exactly two identical indices (i, i, k)
    for i in range(m):
        for k in range(m):
            if i != k:
                val = solve_missing_entry_M3(M3_completed, r, i, i, k)
                # Apply symmetry across all permutations
                M3_completed[i, i, k] = val
                M3_completed[i, k, i] = val
                M3_completed[k, i, i] = val
                
    # Entries with all three identical indices (i, i, i)
    for i in range(m):
        val = solve_missing_entry_M3(M3_completed, r, i, i, i)
        M3_completed[i, i, i] = val
        
    return M3_completed

def compute_whitening_matrix(M2, r):
    """
    Computes the whitening matrix W from M2.
    """
    # 1. Eigendecomposition of M2
    eigenvalues, eigenvectors = np.linalg.eigh(M2)
    
    # 2. Sort eigenvalues in descending order and grab the top 'r'
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    top_eigenvalues = eigenvalues[:r]
    U = eigenvectors[:, :r]
    
    # if eigenvalues are negative or zero, the rank assumption failed
    if np.any(top_eigenvalues <= 1e-12):
        raise ValueError("M2 does not have rank r, or numerical instability occurred.")
        
    # 3. Construct D^{-1/2}
    D_inv_sqrt = np.diag(1.0 / np.sqrt(top_eigenvalues))
    
    # 4. Construct Whitening Matrix W
    W = U @ D_inv_sqrt
    
    return W

def whiten_tensor(M3, W):
    """
    Applies the whitening matrix W to M3 along all three modes efficiently.
    Implements the sequential speedup described in Appendix F.
    """
    # Adding optimize=True forces numpy to calculate T1 -> T2 -> M3_hat sequentially
    # dropping the time complexity from O(m^3 * r^3) down to O(r*m^3 + r^2*m^2 + r^3*m)
    M3_hat = np.einsum('ijl,ia,jb,lc->abc', M3, W, W, W, optimize=True)
    
    return M3_hat

def robust_tensor_power_method(M3_hat, r, L=10, max_iter=100, tol=1e-8):
    """
    Extracts the orthogonal eigenvectors and eigenvalues from the whitened tensor.
    L is the number of random restarts to guarantee hitting the global optimum.
    """
    eigenvalues = []
    eigenvectors = []
    
    T = M3_hat.copy()
    
    for component in range(r):
        best_lambda = -1
        best_theta = None
        
        # Robust restarts: Try L different random vectors and keep the strongest one
        for restart in range(L):
            # 1. Random initialization
            theta = np.random.randn(r) #random r numbers from std normal
            theta /= np.linalg.norm(theta)
            
            # 2. Power Iteration
            for step in range(max_iter):
                # Contract tensor: T(I, J, K) * theta(J) * theta(K)
                theta_new = np.einsum('ijk,j,k->i', T, theta, theta)
                
                norm = np.linalg.norm(theta_new)
                if norm == 0:
                    break
                theta_new /= norm
                
                # Check for convergence (accounting for sign flips)
                if np.linalg.norm(theta_new - theta) < tol or np.linalg.norm(theta_new + theta) < tol:
                    theta = theta_new
                    break
                theta = theta_new
                
            # 3. Calculate final eigenvalue stretch
            lam = np.einsum('ijk,i,j,k->', T, theta, theta, theta)
            
            if lam > best_lambda:
                best_lambda = lam
                best_theta = theta
                
        eigenvalues.append(best_lambda)
        eigenvectors.append(best_theta)
        
        # 4. Deflation: Subtract this component's gravity from the tensor
        # T = T - lambda * (theta \otimes theta \otimes theta)
        deflation_term = best_lambda * np.einsum('i,j,k->ijk', best_theta, best_theta, best_theta)
        T = T - deflation_term
        
    return eigenvalues, eigenvectors


def recover_true_parameters(eigenvalues, eigenvectors, W):
    """
    Un-whitens the extracted components to get the real-world bag weights and coin biases.
    """
    mu_list = []
    alpha_list = []
    
    # Calculate the Moore-Penrose pseudoinverse of W transposed
    W_T_pinv = np.linalg.pinv(W.T)
    
    for lam, theta in zip(eigenvalues, eigenvectors):
        # Recover bag weight (mu_i)
        mu = 1.0 / (lam ** 2)
        mu_list.append(mu)
        
        # Recover true coin biases (alpha_i)
        # alpha_i = (W^T)^\dagger * (theta / sqrt(mu))
        alpha = W_T_pinv @ (theta / np.sqrt(mu))
        alpha = 1.0 - alpha
        alpha_list.append(alpha)
        
    return mu_list, alpha_list


def extract_local_block_parameters(block, n, r, query_oracle):
    # 1. Build the grids and query the Oracle
    M2_off, M3_off = extract_off_diagonal_moments(block, n, query_oracle)
    
    # 2. Vanishing Minors Solver
    M2 = fill_all_M2_diagonals(M2_off, r)
    M3 = fill_all_M3_missing_entries(M3_off, r)
    
    # 3. Whitening
    W = compute_whitening_matrix(M2, r)
    M3_hat = whiten_tensor(M3, W)
    
    # 4. Power Method & Recovery
    eigenvalues, eigenvectors = robust_tensor_power_method(M3_hat, r)
    local_mus, local_alphas = recover_true_parameters(eigenvalues, eigenvectors, W)
    
    return local_mus, local_alphas

def stitch_global_parameters(blocks, n, r, query_oracle):
    """
    Executes the extraction pipeline across all blocks and stitches the random
    permutations together using the r - 1 overlapping anchor variables.
    """
    global_mus = np.zeros(r)
    global_alphas = np.zeros((r, n))
    
    for i, block in enumerate(blocks):
        print(f"Extracting Block {i} (Variables {block[0]} to {block[-1]})...")
        local_mus, local_alphas = extract_local_block_parameters(block, n, r, query_oracle)
        
        # BLOCK 0: Sets 
        if i == 0:
            global_mus[:] = local_mus
            for local_idx, global_var in enumerate(block):
                for comp in range(r):
                    global_alphas[comp, global_var] = local_alphas[comp][local_idx]
        
        # ALL OTHER BLOCKS: Must be aligned 
        else:
            prev_block = blocks[i - 1]
            overlap_vars = [v for v in block if v in prev_block]
            local_overlap_indices = [block.index(v) for v in overlap_vars]
            
            # Build a Cost Matrix instead of testing permutations
            cost_matrix = np.zeros((r, r))
            for local_c in range(r):
                for global_c in range(r):
                    # 1. Compare Bag Weights
                    mu_err = (local_mus[local_c] - global_mus[global_c]) ** 2
                    
                    # 2. Compare Fingerprints (The anchor variables)
                    alpha_err = sum((local_alphas[local_c][l_idx] - global_alphas[global_c, g_var]) ** 2 
                                    for l_idx, g_var in zip(local_overlap_indices, overlap_vars))
                    
                    cost_matrix[local_c, global_c] = mu_err + alpha_err
            
            # O(r^3) Hungarian Algorithm solves the mapping instantly
            local_indices, global_indices = linear_sum_assignment(cost_matrix)
            
            # Apply the optimal O(r^3) mapping
            for local_c, global_c in zip(local_indices, global_indices):
                for l_idx, g_var in enumerate(block):
                    if g_var not in overlap_vars:
                        global_alphas[global_c, g_var] = local_alphas[local_c][l_idx]
                        
    return global_mus, global_alphas

def reconstruct_bernoulli_mixture(n, r, query_oracle):
    """
    Reconstructs the complete mixture parameters in O(n * r^5) time.
    """
    print("Step 1: Preprocessing and Data Cleaning...")
    # S_not = dummies, S_f = common factors
    S_not, S_f, common_params = preprocess_variables(n, query_oracle)
    
    # Filter the alive variables
    alive_vars = [i for i in range(n)] # if i not in S_not and i not in S_f]
    
    # print(f"Removed {len(S_not)} dummies and {len(S_f)} common factors.")
    print(f"{len(alive_vars)} active variables remaining.")
    
    print("\nStep 2: Partitioning into Overlapping Blocks...")
    blocks = partition_into_blocks(alive_vars, r)
    print(f"Created {len(blocks)} blocks to process.")
    
    print("\nStep 3: Executing Tensor Extraction and Stitching...")
    global_mus, global_alphas_alive = stitch_global_parameters(blocks, n, r, query_oracle)
    
    print("\nStep 4: Re-inserting Common Factors...")
    final_alphas = np.zeros((r, n))
    
    # Insert the active variables
    for comp in range(r):
        for var in alive_vars:
            final_alphas[comp, var] = global_alphas_alive[comp, var]
            
    # Insert the common factors (identical across all bags)
    for var, bias in common_params.items():
        for comp in range(r):
            final_alphas[comp, var] = bias
            
    # Dummy variables remain 0.0
    print("\nRECONSTRUCTION COMPLETE.")
    return global_mus, final_alphas

