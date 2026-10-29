"""Fixed physical training coordinates with RNG independent of model draws."""

import torch


def training_points(config, seed):
    """Sample PDE interior, initial line, and both Dirichlet boundaries on CPU."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    dtype = getattr(torch, config.dtype)

    def uniform(count):
        """Draw values strictly inside the unit interval."""
        epsilon = torch.finfo(dtype).eps
        return torch.rand(count, 1, generator=generator, dtype=dtype).clamp(epsilon, 1 - epsilon)

    def space(count):
        """Map uniform samples to the physical spatial interval."""
        return config.x_min + (config.x_max - config.x_min) * uniform(count)

    f = torch.cat((space(config.n_f), config.t_max * uniform(config.n_f)), dim=1)
    ic = torch.cat((space(config.n_ic), torch.zeros(config.n_ic, 1, dtype=dtype)), dim=1)
    times = config.t_max * uniform(config.n_bc)
    bc = torch.cat((torch.cat((torch.full_like(times, config.x_min), times), dim=1),
                    torch.cat((torch.full_like(times, config.x_max), times), dim=1)), dim=0)
    return {"f": f, "ic": ic, "bc": bc}
