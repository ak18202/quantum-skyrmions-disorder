import netket as nk
import optax
import json
import numpy as np

def setup_optimizer():
    """Define separate learning rate schedules for rho and phi nets."""
    rho_schedule = optax.join_schedules(
        [
            optax.linear_schedule(init_value=0.0, end_value=0.001, transition_steps=5000),
            optax.constant_schedule(0.001),
            optax.constant_schedule(0.0001),
        ],
        boundaries=[5000, 22000],
    )

    phi_schedule = optax.join_schedules(
        [
            optax.constant_schedule(0.001),
            optax.constant_schedule(0.0001),
        ],
        boundaries=[22000],
    )

    rho_optimizer = optax.adam(learning_rate=rho_schedule, b1=0.9, b2=0.999)
    phi_optimizer = optax.adam(learning_rate=phi_schedule, b1=0.9, b2=0.999)

    def param_label_fn(path):
        path_elems = tuple(path)
        if any(elem == "rho_net" for elem in path_elems):
            return "rho"
        elif any(elem == "phi_net" for elem in path_elems):
            return "phi"
        else:
            raise ValueError(f"param_label_fn: unknown path {path_elems!r}")

    optimizer = optax.multi_transform(
        {"rho": rho_optimizer, "phi": phi_optimizer}, param_label_fn
    )
    return optimizer


def run_training(ham, sampler, model, n_samples=1000, n_iter=30, L=5, out_prefix="ground_state"):
    """Run VMC optimization and return energy statistics."""
    vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples)
    optimizer = setup_optimizer()
    gs = nk.VMC(ham, optimizer, variational_state=vstate)

    gs.run(n_iter=n_iter, out=out_prefix, step_size=30)

    # Load log file
    data = json.load(open(f"{out_prefix}.log"))
    energy_iter = data["Energy"]["iters"]
    energy_means = np.array(data["Energy"]["Mean"]["real"])
    energy_means /= (L * L)  # normalize per site
    energy_sem = np.array([x if x is not None else np.nan for x in data["Energy"]["Sigma"]])
    variance = np.array([x if x is not None else np.nan for x in data["Energy"]["Variance"]])

    return vstate, energy_iter, energy_means, energy_sem, variance
