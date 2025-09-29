import logging
import torch
import numpy as np
import time
from tqdm import tqdm
from typing import Tuple

def get_progress_bar(total: int, desc: str):
    return tqdm(
        total=total,
        desc=desc,
        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
    )  

@torch.no_grad()
def objective_function(
    W: torch.Tensor, 
    H: torch.Tensor, 
    P: torch.Tensor, 
    C: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the quantization error (objective value).
    
    Args:
    W: Weight matrix (row_count * group_count, group_size)
    H: Hessian matrix (blk_num, group_size, group_size)
    P: Assignment matrix (row_count * group_count, group_size//vec_sz, n_cluster)
    C: Centroid matrix (n_cluster, vec_sz)
    
    Returns:
    Objective value (scalar)
    """

    device = torch.device("cuda")
    P, C = P.to(device), C.to(device)
    W_hat = torch.einsum('ijc,ck->ijk', P, C) # Shape: (row_count * group_count, group_size//vec_sz, vec_sz)
    W_hat = W_hat.view(W_hat.shape[0], -1) # Shape: (row_count * group_count, group_size)
    delta_w = W_hat - W

    blk_num = H.shape[0]
    blk_size = W.shape[0] // blk_num

    delta_w = delta_w.reshape(blk_num, blk_size, delta_w.shape[-1])
    objective_value = torch.einsum('nij,njk,nik->i', delta_w, H, delta_w) 
    total_error = objective_value.mean()

    return total_error


@torch.no_grad()
def parallel_objective_function_sub(
    W: torch.Tensor,  # Shape: (b, g_cd)
    quadratic: torch.Tensor,  # Shape: (g_cd, g_cd)
    linear: torch.Tensor,  # Shape: (b, g_cd)
    W_hat_options: torch.Tensor,  # Shape: (b, g_cd, num_options)
) -> torch.Tensor:
    """
    Calculate the quantization error (objective value), and return the list of errors for each options.
    W_hat is a tensor with possible options concatenated along the last dimension.
    
    Args:
    W: Weight matrix (b, g_cd)
    quadratic: torch.Tensor,  # Shape: (g_cd, g_cd)
    linear: torch.Tensor,  # Shape: (b, g_cd)
    W_hat_options: torch.Tensor,  # Shape: (b, g_cd, num_options)
    
    Returns:
    Possible objective values for each options (b, num_options)
    """
    device = torch.device("cuda")
    W_hat_options = W_hat_options.to(device)
    b, g_cd, num_options = W_hat_options.shape

    delta_w_g = W_hat_options - W.unsqueeze(2).expand(-1, -1, num_options)

    quadratic_term = torch.einsum('jk,ijp,ikp->ip', quadratic, delta_w_g, delta_w_g)
    linear_term = torch.einsum('ij,ijp->ip', linear, delta_w_g)
    total_error_quad = quadratic_term + linear_term
    return total_error_quad



def update_batch_P(
    W: torch.Tensor,  # Shape: (b, group_size)
    H: torch.Tensor,  # Shape: (blk_num, group_size, group_size)
    P: torch.Tensor,  # Shape: (b, group_size // vec_sz, n_cluster)
    C: torch.Tensor,  # Shape: (n_cluster, vec_sz)
    iteration: int,
    g_cd: int,        # Number of weights to update at a time
    cd_cycles: int,
    verbose: bool = False,
):
    device = torch.device("cuda")
    C = C.to(device)
    C_ = C.unsqueeze(0).expand(P.shape[0], -1, -1)
    assignments_prev = P.argmax(dim=-1).to(device)  # Shape: (b, group_size // vec_sz)
    b, d = assignments_prev.shape
    n_cluster, vec_sz = C_.size(1), C_.size(2)
    assert H.shape[0] == 1
    H_ = H[0]

    assignments = assignments_prev.clone()
    update_size = cd_cycles * d

    for update_start_idx in range(0, update_size, g_cd):
        start_idx = update_start_idx % d
        end_idx = min(start_idx + g_cd, d)
        indices = torch.arange(start_idx * vec_sz, end_idx * vec_sz, device=device)
        indices_assignments = torch.arange(start_idx, end_idx, device=device)
        # Generate all possible assignments for the group
        num_options = n_cluster ** g_cd
        if num_options > 1e6:
            print(f"Skipping group starting at index {start_idx} due to large number of assignments ({num_options}).")
            continue

        # Create all possible assignments for the group
        from itertools import product
        assignments_list = list(product(range(n_cluster), repeat=g_cd))
        assignments_array = torch.tensor(assignments_list, device=device).T  # Shape: (g_cd, num_options)
        assignments_array = assignments_array.unsqueeze(0).expand(b, -1, -1)  # Shape: (b, g_cd, num_options, vec_sz)

        # Creating options for g_cd weights
        C_expanded = C_.unsqueeze(1).expand(-1, g_cd, -1, -1) # Shape: (b, g_cd, n_cluster, vec_sz)
        W_g_hat_options = torch.gather(C_expanded, dim=2, index=assignments_array.unsqueeze(-1).expand(-1, -1, -1, vec_sz)) # Shape: (b, g_cd, num_options, vec_sz)

        # Gathering original quantized weights and compute linear & quadratic terms
        # Expand C and gather original weights
        C_expanded_org = C_.unsqueeze(1).expand(-1, d, -1, -1) # Shape: (b, d, n_cluster, vec_sz)
        W_hat_org = torch.gather(C_expanded_org, dim=2, index=assignments.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, vec_sz)).squeeze(2) # Shape: (b, d, vec_sz)
        
        # Compute deltas
        delta_w_org = W_hat_org.view(b, -1) - W # Shape: (b, group_size)
        
        # Get indices and slices
        notg_indices = torch.cat([
            torch.arange(0, start_idx * vec_sz, device=device),
            torch.arange(end_idx * vec_sz, d * vec_sz, device=device)
        ])

        H_g_notg = H_[indices, :][:, notg_indices] # Shape: (g_cd * vec_sz, group_size - g_cd * vec_sz)

        delta_w_org_notg = delta_w_org[:, notg_indices].to(device) # Shape: (b, group_size - g_cd * vec_sz)

        # Compute quadratic and linear terms
        quadratic = H_[indices, :][:, indices] # Shape: (g_cd * vec_sz, g_cd * vec_sz)
        linear = 2 * torch.einsum('gd,id->ig', H_g_notg, delta_w_org_notg) # Shape: (b, g_cd * vec_sz)
        W_g = W[:, indices] # Shape: (b, g_cd * vec_sz)

        W_g_hat_options = W_g_hat_options.permute(0, 1, 3, 2).view(b, g_cd * vec_sz, num_options) # Shape: (b, g_cd * vec_sz, num_options)
        # Objective function computation
        cur_obj_value = parallel_objective_function_sub(W_g, quadratic, linear, W_g_hat_options) # Shape: (b, num_options)

        # Update assignments
        min_obj, argmin_obj = cur_obj_value.min(dim=1, keepdim=True)
        expanded_argmin_obj = argmin_obj.unsqueeze(1).expand(-1, g_cd, -1).to(device)
        assignments[:, indices_assignments] = assignments_array.gather(dim=2, index=expanded_argmin_obj).squeeze(-1) # Shape: (row_count * group_count, g_cd)

    num_changed = (assignments_prev != assignments).sum().item()
    total_assignments = assignments_prev.numel()
    percentage_changed = num_changed / total_assignments * 100
    if verbose:
        logging.info(f"Percentage of assignments changed: {percentage_changed:.2f}%%")

    # Convert assignments to one-hot encoding to create new P
    P = torch.zeros((b, d, n_cluster), dtype=torch.float32, device=assignments.device)
    P.scatter_(2, assignments.long().unsqueeze(-1), 1.0)

    return P


def update_P(
    W: torch.Tensor,  # Shape: (row_count * group_count, group_size)
    H: torch.Tensor,  # Shape: (blk_num, group_size, group_size)
    P: torch.Tensor,  # Shape: (row_count * group_count, group_size//vec_sz, n_cluster)
    C: torch.Tensor,  # Shape: (n_cluster, vec_sz)
    iteration: int,
    g_cd: int = 1,
    cd_cycles: int = 4,
):
    n_cluster = C.shape[0]
    batch_output_size = 4096 # * 32 // max(32, n_cluster) 
    device = torch.device("cuda")
    updated_P_list = []

    pb = get_progress_bar((W.size(0) - 1) // batch_output_size + 1, f"Updating P (cd_cycles={cd_cycles})")
    for out_idx in range(0, W.size(0), batch_output_size):
        torch.cuda.reset_peak_memory_stats()  # Reset memory stats at start of iteration

        W_batch = W[out_idx:out_idx+batch_output_size].to(device)
        P_batch = P[out_idx:out_idx+batch_output_size].to(device)
        C_batch = C.to(device)

        verbose = False # (out_idx == 0)
        
        updated_P_batch = update_batch_P(W_batch, H, P_batch, C_batch, iteration, g_cd=g_cd, cd_cycles=cd_cycles, verbose=verbose).cpu()
        updated_P_list.append(updated_P_batch)
        pb.update(1)
    pb.close()

    # Log max CUDA memory usage
    P = torch.cat(updated_P_list, dim=0)
    return P

def project_to_pd(H, eps=1e-2):
    H_sym = (H + H.T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(H_sym)
    eigenvalues = torch.clamp(eigenvalues, min=eps)
    H_spd = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
    H_spd = (H_spd + H_spd.T) / 2
    H_spd = H_spd.to(H.dtype)
    return H_spd

import torch
def kron_with_identity_vec(P: torch.Tensor, vec_sz: int) -> torch.Tensor:
    B, d, c = P.shape
    I_vec_sz = torch.eye(vec_sz, vec_sz).to(P.device)
    P_expanded = P.unsqueeze(-1).unsqueeze(-1)             # (B, d, c, 1, 1)
    I_expanded = I_vec_sz.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, vec_sz, vec_sz)

    out = P_expanded * I_expanded  # (B, d, c, vec_sz, vec_sz)
    out = out.permute(0, 1, 3, 2, 4)  # (B, d, vec_sz, c, vec_sz)
    out = out.reshape(B, d * vec_sz, c * vec_sz)
    return out

def update_C(
    W: torch.Tensor,  # (row, gs)
    H: torch.Tensor,  # (1, gs, gs)
    P: torch.Tensor,  # (row, gs//vec_sz, n_cluster)
    C: torch.Tensor,  # (n_cluster, vec_sz)
    batch_size: int = 256
):
    device = W.device
    dtype  = W.dtype

    L = torch.linalg.cholesky(H[0])      # (gs, gs)
    LT = L.transpose(0, 1)               # (gs, gs)

    row, gs = W.shape
    n_cluster, vec_sz = C.shape
    
    A = torch.zeros(n_cluster * vec_sz, n_cluster * vec_sz, device=device, dtype=dtype)
    b = torch.zeros(n_cluster * vec_sz,    device=device, dtype=dtype)

    for start in range(0, row, batch_size):
        end = min(start + batch_size, row)

        # (B, gs // vec_sz, n_cluster)
        P_chunk = P[start:end].to(device)
        # (B, gs)
        W_chunk = W[start:end].to(device)
        B = P_chunk.shape[0]

        # kronecker product with identity.
        P_chunk_expanded = kron_with_identity_vec(P_chunk, vec_sz) # Shape: (B, gs, n_cluster * vec_sz)

        X_temp = torch.einsum('ij,bjk->bik', LT, P_chunk_expanded) # Shape: (B, gs, n_cluster * vec_sz)
        W_temp = torch.einsum('ij,bj->bi', LT, W_chunk) # Shape: (B, gs)

        A += torch.einsum('bik,bil->kl', X_temp, X_temp) # Shape: (n_cluster * vec_sz, n_cluster * vec_sz)
        b += torch.einsum('bik,bi->k', X_temp, W_temp) # Shape: (n_cluster * vec_sz)

    C_flat = torch.linalg.solve(A, b) # Shape: (n_cluster * vec_sz)
    C = C_flat.view(n_cluster, vec_sz) # Shape: (n_cluster, vec_sz)
    return C

def train_least_squares(
    W: np.ndarray, # Shape: (row_count * group_count, group_size)
    init_P: np.ndarray, # Shape: (row_count * group_count, group_size//vec_sz, n_cluster)
    init_centroids: np.ndarray, # Shape: (n_cluster, vec_sz)
    H: np.ndarray, # Shape: (blk_num, group_size, group_size)
    num_iterations: int = 3,
    cd_cycles: int = 4,
    eig_threshold: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda")

    P = torch.tensor(init_P, dtype=torch.float32, device="cpu")
    C = torch.tensor(init_centroids, dtype=torch.float32, device="cpu")
    W = torch.tensor(W, dtype=torch.float32).to(device)
    H = torch.tensor(H, dtype=torch.float32).to(device)

    # eigenvalues = torch.linalg.eigvalsh(H)
    # for i in range(eigenvalues.shape[0]):
    #     top_3_and_bottom_3 = [round(eig.item(), 2) for eig in torch.cat([eigenvalues[i][:3], eigenvalues[i][-3:]])]
    #     logging.info(f"{i+1}-th H has Eigenvalues (top 3 and bottom 3): {top_3_and_bottom_3}, Projecting to PD with eps=1e-6 for numerical stability")
    #     H[i] = project_to_pd(H[i], eps=1e-6)

    #     eps = eig_threshold * 10
    #     while not torch.all(eigenvalues[i] > eig_threshold):
    #         top_3_and_bottom_3 = [round(eig.item(), 2) for eig in torch.cat([eigenvalues[i][:3], eigenvalues[i][-3:]])]
    #         logging.info(f"{i+1}-th H not PD, Eigenvalues (top 3 and bottom 3): {top_3_and_bottom_3}, Projecting to PD with eps={eps}")
    #         H[i] = project_to_pd(H[i], eps=eps)
    #         eigenvalues = torch.linalg.eigvalsh(H)
    #         eps *= 10
    #     top_3_and_bottom_3 = [round(eig.item(), 2) for eig in torch.cat([eigenvalues[i][:3], eigenvalues[i][-3:]])]
    #     logging.info(f"{i+1}-th H PD, Eigenvalues (top 3 and bottom 3): {top_3_and_bottom_3}")
    diag = torch.arange(H.shape[1], device=device)
    for i in range(H.shape[0]):
        avg_diag = torch.mean(torch.diag(H[i]))
        damp, prev_damp = 1e-7, 0.
        while True:
            try:
                torch.linalg.cholesky(H[i])
                logging.info(f"{i+1}-th H is PD, dampening factor={prev_damp:.2e}")
                break
            except Exception as e:
                print(e)
                logging.info(f"{i+1}-th H is not PD, try dampening with factor={damp:.2e}")
                H[i, diag, diag] += (damp - prev_damp) * avg_diag
                prev_damp = damp
                damp *= 10
                if damp > 1e0:
                    exit()

    best_obj_value = objective_function(W, H, P, C).item()
    best_P, best_C = P.detach().cpu().clone(), C.detach().cpu().clone()
    logging.info(f"Initial objective: {best_obj_value:.6f}")

    log_dict = {"objective": [], "iteration": []}
    log_dict["objective"].append(best_obj_value)
    log_dict["iteration"].append(0)

    for iteration in range(num_iterations):
        start_time = time.time()

        ######### Update P #########
        if iteration > 0:
            P = update_P(W, H, P, C, iteration, cd_cycles=cd_cycles)

        # Compute objective value for logging
        obj_value = objective_function(W, H, P, C).item()
        logging.info(f"Iteration {iteration + 1} (P update): Objective: {obj_value:.4f}")
        log_dict["objective"].append(obj_value)
        log_dict["iteration"].append(iteration + 1)


        ######### Update C #########
        C = update_C(W, H, P, C)

        # Check if the objective value improved
        current_obj_value = objective_function(W, H, P, C).item()
        log_dict["objective"].append(current_obj_value)
        log_dict["iteration"].append(iteration + 1)
        if current_obj_value < best_obj_value:
            best_obj_value = current_obj_value
            best_P, best_C = P.detach().cpu().clone(), C.detach().cpu().clone()
            logging.info(f"Iteration {iteration + 1} (C update): Objective: {current_obj_value:.4f} | Improved and using this one.")
        else:
            logging.info(f"Iteration {iteration + 1} (C update): Objective: {current_obj_value:.4f} | Not improved. Using previous best values.")
            P, C = best_P, best_C
            break  # Early stopping

        end_time = time.time()

        logging.info(f"Iteration {iteration + 1} / {num_iterations} completed. "
                     f"Update time: {end_time - start_time:.2f} sec")

    end_time = time.time()
    logging.info(f"Least squares training time: {end_time - start_time:.2f} seconds")

    P = P.detach().cpu()
    C = C.detach().cpu().to(torch.float32)

    return P, C, log_dict


def test():
    from lib.utils.kmeans import fit_kmeans
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    vec_sz = 4
    lut_size = (1 << (2 * vec_sz))
    W = torch.randn(4096, 4096)
    H = torch.randn(4096, 4096)

    rand_data = torch.randn(10000, vec_sz)
    C = fit_kmeans(rand_data, lut_size)[0]
    print("C", C.shape)

    W_vec = W.view(4096, 4096 // vec_sz, vec_sz)
    W_vec = W_vec.unsqueeze(0) # Shape: (1, 4096, 2048, 2)
    C_ = C.unsqueeze(1).unsqueeze(1) # Shape: (16, 1, 1, 2)
    diff = W_vec - C_ # Shape: (16, 4096, 2048, 2)
    dist_sq = diff.pow(2).sum(-1) # Shape: (16, 4096, 2048)
    idx = dist_sq.argmin(dim=0) # Shape: (4096, 2048)
    init_P = torch.zeros(4096, 4096 // vec_sz, lut_size)
    init_P.scatter_(2, idx.unsqueeze(-1), 1)

    H = H @ H.T
    H = H + 1e-6 * torch.eye(4096, 4096)
    H = H.unsqueeze(0)

    P, C, log_dict = train_least_squares(
        W=W.numpy(),
        init_P=init_P.numpy(),
        init_centroids=C.numpy(),
        H=H.numpy(),
        num_iterations=10,
        cd_cycles=4,
        eig_threshold=1e-3,
    )

    for i in range(len(log_dict["objective"])):
        logging.info(f"Iteration {log_dict['iteration'][i]}: Objective: {log_dict['objective'][i]:.4f}")
    print("P", P.shape)
    print("C", C.shape)

    # recons 
    W_hat = torch.einsum('ijc,ck->ijk', P, C)
    W_hat = W_hat.view(W_hat.shape[0], -1)
    err = (W - W_hat).pow(2).mean()
    print("err", err.item())

    dWHdW = (W - W_hat) @ H[0] @ (W - W_hat).T
    err_tr = torch.trace(dWHdW) / H.shape[1]
    print("err_tr", err_tr.item())

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    test()
    
