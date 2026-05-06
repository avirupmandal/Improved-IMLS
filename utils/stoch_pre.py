import torch

def sample_uniform_ball(shape, alpha, device):
    direction = torch.randn(shape, device=device)
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    r = torch.rand(*shape[:-1], 1, device=device).pow(1.0 / 3.0) * alpha
    return direction * r

def stochastic_precondition(coords, alpha, minb, maxb, radii=None, m=None):
    if radii is not None and m is not None:
        m_clamped = m.clamp(min=1.0, max=3.0)
        support_radius = (m_clamped * radii).min().item()
        alpha = min(alpha, 0.25 * support_radius)

    noise = sample_uniform_ball(coords.shape, alpha, coords.device)
    coords = coords + noise

    coords = (coords - minb) % (2 * (maxb - minb)) + minb
    coords = torch.where(coords > maxb, 2 * maxb - coords, coords)

    return coords

