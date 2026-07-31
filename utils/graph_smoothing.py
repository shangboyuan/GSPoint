import torch
from torch.cuda.amp import autocast


def _topk_by_score(indices, scores, k):
    if indices.shape[-1] < k:
        pad = k - indices.shape[-1]
        indices = torch.cat([indices, indices[..., :1].expand(*indices.shape[:-1], pad)], dim=-1)
        scores = torch.cat([scores, scores.new_full((*scores.shape[:-1], pad), -1.0)], dim=-1)
    sorted_indices = indices.sort(dim=-1)[0]
    sorted_scores = scores.gather(-1, indices.argsort(dim=-1))
    starts = torch.ones_like(sorted_indices, dtype=torch.bool)
    starts[..., 1:] = sorted_indices[..., 1:] != sorted_indices[..., :-1]
    group_ids = starts.cumsum(dim=-1) - 1

    reduced = torch.zeros_like(sorted_scores)
    reduced.scatter_add_(-1, group_ids, sorted_scores)
    reduced = reduced.gather(-1, group_ids).masked_fill(~starts, -1.0)
    _, topk_idx = reduced.topk(k=k, dim=-1, largest=True, sorted=False)
    return sorted_indices.gather(-1, topk_idx), reduced.gather(-1, topk_idx).clamp_min(0)


def _compute_A_D(knn):
    B, N, K = knn.shape
    A_list = []
    D_list = []
    W_list = []
    rows = torch.arange(N, device=knn.device).repeat_interleave(K)
    row_grid = rows.view(N, K)
    for b in range(B):
        cols = knn[b].reshape(-1).long()
        key = rows * N + cols
        rev_key = cols * N + rows
        sorted_key = key.sort()[0]
        pos = torch.searchsorted(sorted_key, rev_key)
        valid = pos < sorted_key.numel()
        pos = pos.clamp(max=sorted_key.numel() - 1)
        mutual = valid & (sorted_key[pos] == rev_key)
        A = knn[b].long()
        D = mutual.view(N, K).sum(dim=-1).float().clamp_min(1)
        W = torch.zeros((N, K), dtype=torch.float32, device=knn.device)
        src_degree = D[row_grid]
        dst_degree = D[A]
        W[mutual.view(N, K)] = torch.rsqrt(src_degree[mutual.view(N, K)] * dst_degree[mutual.view(N, K)])
        A_list.append(A)
        D_list.append(D)
        W_list.append(W)
    return torch.stack(A_list), torch.stack(D_list), torch.stack(W_list)


def graph_smoothing_indices(knn, steps=3, k=None, max_candidates=None, alpha=0.5):
    if steps <= 0:
        return knn if k is None else knn[..., :k]

    squeeze = knn.dim() == 2
    if squeeze:
        knn = knn.unsqueeze(0)

    knn = knn.long()
    B, N, K = knn.shape
    k = K if k is None else k
    max_candidates = max(k, max_candidates or k * 4)
    A, D, W = _compute_A_D(knn)

    self_idx = torch.arange(N, device=knn.device).view(1, N, 1).expand(B, -1, -1)
    frontier_idx = self_idx
    frontier_score = torch.ones((B, N, 1), dtype=torch.float32, device=knn.device)
    candidates = [frontier_idx]
    candidate_scores = [frontier_score]
    for _ in range(steps):
        M = frontier_idx.shape[-1]
        gather_idx = frontier_idx.reshape(B, N * M, 1).expand(-1, -1, K)
        next_idx = torch.gather(A, 1, gather_idx).reshape(B, N, M * K)
        next_weight = torch.gather(W, 1, gather_idx).reshape(B, N, M * K)
        next_score = frontier_score.unsqueeze(-1).mul(alpha).mul(next_weight.view(B, N, M, K)).reshape(B, N, M * K)
        keep = min(max_candidates, next_idx.shape[-1])
        frontier_idx, frontier_score = _topk_by_score(next_idx, next_score, keep)
        candidates.append(frontier_idx)
        candidate_scores.append(frontier_score)

    candidates.append(knn[..., :k])
    candidate_scores.append(torch.zeros((B, N, k), dtype=torch.float32, device=knn.device))
    smoothed, _ = _topk_by_score(torch.cat(candidates, dim=-1), torch.cat(candidate_scores, dim=-1), k)
    return smoothed.squeeze(0) if squeeze else smoothed


def graph_smoothing_knn(pwd, k, steps=3, max_candidates=None, alpha=0.5):
    _, knn = pwd.topk(k=k, dim=-1, largest=False, sorted=False)
    return graph_smoothing_indices(
        knn, steps=steps, k=k, max_candidates=max_candidates, alpha=alpha
    )


@autocast(False)
def compute_eigen_geometry(xyz, knn, num_iters=3):
    squeeze = xyz.dim() == 2
    if squeeze:
        xyz = xyz.unsqueeze(0)
        knn = knn.unsqueeze(0)

    orig_dtype = xyz.dtype
    xyz = xyz.float()
    knn = knn.long()
    B, N, K = knn.shape
    eps = 1e-5

    nbr_xyz = torch.gather(
        xyz, 1, knn.reshape(B, N * K, 1).expand(-1, -1, 3)
    ).reshape(B, N, K, 3)
    delta_p = nbr_xyz - xyz.unsqueeze(2)
    cov = torch.matmul(delta_p.transpose(2, 3), delta_p) / K

    eye = torch.eye(3, dtype=torch.float32, device=xyz.device).view(1, 1, 3, 3)
    cov_shifted_1 = cov + eye

    v1 = torch.ones((B, N, 3, 1), dtype=torch.float32, device=xyz.device)
    for _ in range(num_iters):
        v1 = torch.matmul(cov_shifted_1, v1)
        v1 = v1 / torch.sqrt(torch.clamp(torch.sum(v1 * v1, dim=-2, keepdim=True), min=eps))

    max_idx_v1 = torch.argmax(v1.abs(), dim=-2, keepdim=True)
    sign_v1 = torch.gather(v1, -2, max_idx_v1).sign()
    sign_v1[sign_v1 == 0] = 1.0
    v1 = v1 * sign_v1

    l1 = torch.matmul(v1.transpose(-2, -1), torch.matmul(cov, v1)).squeeze(-1)
    cov_deflated = cov - l1.unsqueeze(-1) * torch.matmul(v1, v1.transpose(-2, -1))
    cov_shifted_2 = cov_deflated + eye

    axis_idx = torch.argmin(v1.squeeze(-1).abs(), dim=-1)
    v2 = torch.nn.functional.one_hot(axis_idx, num_classes=3).float().unsqueeze(-1)
    for _ in range(num_iters):
        v2 = torch.matmul(cov_shifted_2, v2)
        v2 = v2 - torch.matmul(v1.transpose(-2, -1), v2) * v1
        v2 = v2 / torch.sqrt(torch.clamp(torch.sum(v2 * v2, dim=-2, keepdim=True), min=eps))

    max_idx_v2 = torch.argmax(v2.abs(), dim=-2, keepdim=True)
    sign_v2 = torch.gather(v2, -2, max_idx_v2).sign()
    sign_v2[sign_v2 == 0] = 1.0
    v2 = v2 * sign_v2

    l2 = torch.matmul(v2.transpose(-2, -1), torch.matmul(cov, v2)).squeeze(-1)
    v3_vec = torch.cross(v1.squeeze(-1), v2.squeeze(-1), dim=-1)
    v3_vec = v3_vec / torch.sqrt(torch.clamp(torch.sum(v3_vec * v3_vec, dim=-1, keepdim=True), min=eps))
    v3 = v3_vec.unsqueeze(-1)
    l3 = torch.matmul(v3.transpose(-2, -1), torch.matmul(cov, v3)).squeeze(-1)

    eigenvalues = torch.cat([l3, l2, l1], dim=-1).to(orig_dtype)
    eigenvectors = torch.cat([v3, v2, v1], dim=-1).to(orig_dtype)
    eigenvalues_sorted = torch.flip(eigenvalues, dims=[-1])

    proj = torch.matmul(delta_p.to(orig_dtype), eigenvectors[..., [1, 0, 2]])
    x_prime = proj[..., 0]
    y_prime = proj[..., 1]
    z_prime = proj[..., 2]

    radius = torch.sqrt(torch.clamp(x_prime * x_prime + y_prime * y_prime, min=eps))
    cos_theta = x_prime / radius
    z_max = torch.clamp(z_prime.abs().max(dim=-1, keepdim=True)[0], min=1e-3)
    r_max = torch.clamp(radius.max(dim=-1, keepdim=True)[0], min=1e-3)
    p_prime = torch.stack([z_prime / z_max, radius / r_max, cos_theta], dim=-1)

    if squeeze:
        return eigenvalues_sorted.squeeze(0), delta_p.squeeze(0), p_prime.squeeze(0)
    return eigenvalues_sorted, delta_p, p_prime
