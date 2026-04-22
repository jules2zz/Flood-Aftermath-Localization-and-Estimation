import torch
import torch.nn.functional as F

def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist

def cosine_dist(x, y):
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    dist = 2 - 2 * torch.mm(x, y.t())
    return dist

def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6
    W = torch.exp(diff) * mask / Z
    return W

def hard_example_mining(dist_mat, is_pos, is_neg):
    assert len(dist_mat.size()) == 2
    dist_ap, _ = torch.max(dist_mat * is_pos, dim=1)
    dist_an, _ = torch.min(dist_mat * is_neg + is_pos * 1e9, dim=1)
    return dist_ap, dist_an

def weighted_example_mining(dist_mat, is_pos, is_neg):
    assert len(dist_mat.size()) == 2
    dist_ap = dist_mat * is_pos
    dist_an = dist_mat * is_neg
    weights_ap = softmax_weights(dist_ap, is_pos)
    weights_an = softmax_weights(-dist_an, is_neg)
    dist_ap = torch.sum(dist_ap * weights_ap, dim=1)
    dist_an = torch.sum(dist_an * weights_an, dim=1)
    return dist_ap, dist_an

def triplet_loss(embedding, targets, margin=0.2, norm_feat=True, hard_mining=True):
    """
    若 norm_feat=True -> 余弦距离（三元组与 LMCL 的角度空间一致）
    hard_mining=True -> 选择 hardest negative；A2(K=2)场景下正样只有1个，hard/weighted 对正样等价
    """
    dist_mat = cosine_dist(embedding, embedding) if norm_feat else euclidean_dist(embedding, embedding)

    N = dist_mat.size(0)
    is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
    is_neg = 1.0 - is_pos

    if hard_mining:
        dist_ap, dist_an = hard_example_mining(dist_mat, is_pos, is_neg)
    else:
        dist_ap, dist_an = weighted_example_mining(dist_mat, is_pos, is_neg)

    y = dist_an.new_full(dist_an.size(), 1)
    if margin > 0:
        loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=margin)
    else:
        loss = F.soft_margin_loss(dist_an - dist_ap, y)
        if loss == float('Inf'):
            loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)
    return loss
