"""Physics-only objectives; analytic interior values never enter training."""

import torch

from .physics import breather, initial_velocity, pde_residual


def initial_losses(model, points, omega):
    """Compute displacement and physical time-derivative initial penalties."""
    coordinates = points.detach().clone().requires_grad_(True)
    phi = model(coordinates)
    phi_t = torch.autograd.grad(phi, coordinates, torch.ones_like(phi), create_graph=True)[0][:, 1:2]
    return phi.square().mean(), (phi_t - initial_velocity(coordinates[:, :1], omega)).square().mean()


def combine_losses(parts, config):
    """Form the specified IC sum and weighted total from raw mean losses."""
    parts["ic_loss"] = parts["ic_value_loss"] + parts["ic_velocity_loss"]
    parts["total_loss"] = (config.lambda_pde * parts["pde_loss"] +
                           config.lambda_ic * parts["ic_loss"] +
                           config.lambda_bc * parts["bc_loss"])
    return parts


def loss_terms(model, samples, config):
    """Return differentiable losses for one update, including both IC terms."""
    ic_value, ic_velocity = initial_losses(model, samples["ic"], config.omega)
    # This is the ONLY use of the analytic field within the training objective.
    boundary_target = breather(samples["bc"], config.omega).detach()
    parts = {"pde_loss": pde_residual(model, samples["f"]).square().mean(),
             "ic_value_loss": ic_value, "ic_velocity_loss": ic_velocity,
             "bc_loss": (model(samples["bc"]) - boundary_target).square().mean()}
    return combine_losses(parts, config)


def full_losses(model, samples, config):
    """Re-evaluate final-checkpoint means over all training points in chunks."""
    parts = dict.fromkeys(("pde_loss", "ic_value_loss", "ic_velocity_loss", "bc_loss"), 0.0)
    for group, points in samples.items():
        for chunk in points.split(config.eval_batch_size):
            weight = len(chunk) / len(points)
            if group == "f":
                parts["pde_loss"] += pde_residual(model, chunk).square().mean().item() * weight
            elif group == "ic":
                value, velocity = initial_losses(model, chunk, config.omega)
                parts["ic_value_loss"] += value.item() * weight
                parts["ic_velocity_loss"] += velocity.item() * weight
            else:
                with torch.no_grad():
                    loss = (model(chunk) - breather(chunk, config.omega)).square().mean()
                parts["bc_loss"] += loss.item() * weight
    return combine_losses(parts, config)
