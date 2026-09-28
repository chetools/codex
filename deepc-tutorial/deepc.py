# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#     "marimo==0.25.0",
#     "numpy==2.5.3",
#     "cvxpy==1.9.3",
#     "matplotlib==3.11.2",
#     "osqp==1.1.3",
#     "clarabel>=0.11,<0.12",
# ]
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium", app_title="DeePC: control from experimental data")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import cvxpy as cp
    import matplotlib.pyplot as plt
    return cp, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # DeePC: optimal control directly from experimental data

    **A small experiment you can read, run, and change.** We will steer a
    two-state system using only its recorded inputs and outputs. No function
    in the DeePC controller receives or estimates $A,B,C,D$.

    This original teaching example follows the deterministic formulation in
    Coulson, Lygeros, and Dörfler,
    [*Data-Enabled Predictive Control: In the Shallows of the DeePC*](https://arxiv.org/abs/1811.05890),
    especially Equations (4)-(6) and Algorithm 2. It is not a reproduction of
    their quadcopter experiment.

    **Prerequisites:** Python arrays, matrix multiplication, and the idea of
    minimizing a function subject to constraints. Allow about 30-45 minutes.
    Expand code cells to see every implementation step. Everything needed is
    in this file, including dependency versions and generated experimental data.

    **Route:** experiment → trajectory windows → prediction → optimization →
    feedback → independent verification → measurement noise.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. What are we optimizing?

    At time $t$, choose $N$ future inputs to minimize
    $$J=\sum_{k=0}^{N-1}\left[q(y_{t+k}-r_{t+k})^2+\rho u_{t+k}^2\right].$$
    Here $r$ is the desired output, $q$ penalizes tracking error, and $\rho$
    penalizes effort. We also require $|u|\leq1$ and $|y|\leq1.5$.
    We solve again after each measurement, applying only the first input.
    This is *receding-horizon* control.

    The optimization needs a rule connecting future inputs and outputs.
    DeePC obtains that rule from overlapping windows of experimental data.
    The data forms an implicit prediction model, even though we never fit
    a state-space realization.

    **Timing convention throughout:** record $y_t=Cx_t$ alongside $u_t$,
    then advance to $x_{t+1}=Ax_t+Bu_t$. Thus $u_t$ first affects $y_{t+1}$.
    Mixing pre-update and post-update outputs would silently shift the model.
    """)
    return


@app.cell
def _(mo):
    effort = mo.ui.slider(0.01, 1.0, step=0.01, value=0.1, label="Input penalty ρ")
    mo.vstack([mo.md("Change the effort penalty and watch the controller below recompute."), effort])
    return (effort,)


@app.cell
def _(effort):
    T = 160
    T_ini = 4
    N = 12
    n_bound = 2
    q = 10.0
    rho = float(effort.value)
    u_limit = 1.0
    y_limit = 1.5
    return N, T, T_ini, n_bound, q, rho, u_limit, y_limit


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. A virtual experiment, with hidden states

    To make the tutorial reproducible, a simulator plays the role of the
    physical equipment. Its two states are coupled, stable, and observable
    from one output. Only the simulator and the later MPC benchmark use these
    matrices. On actual equipment, replace `simulate` with acquisition hardware.

    We start at rest and apply a seeded, bounded random input. Each returned
    output is paired with the input at the same sample. The returned final state is useful
    only for verification; it is never supplied to DeePC.
    """)
    return


@app.cell
def _(np):
    A = np.array([[0.8, 0.2], [0.0, 0.7]])
    B = np.array([0.1, 0.3])
    C = np.array([1.0, 0.0])

    def simulate(inputs, x0=None):
        """SISO plant; output is sampled BEFORE each state update (D=0)."""
        x = np.zeros(2) if x0 is None else np.array(x0, dtype=float).copy()
        outputs = []
        for uk in np.asarray(inputs).reshape(-1):
            outputs.append(float(C @ x))
            x = A @ x + B * uk
        return np.array(outputs), x

    return A, B, C, simulate


@app.cell
def _(T, np, simulate):
    u_data = np.random.default_rng(7).uniform(-1.0, 1.0, T)
    y_data, _ = simulate(u_data)
    return u_data, y_data


@app.cell
def _(plt, u_data, y_data):
    data_figure, _axes = plt.subplots(2, 1, figsize=(9, 4), sharex=True)
    _axes[0].step(range(len(u_data)), u_data, where="post", color="#2563eb")
    _axes[0].set_ylabel("Experimental u")
    _axes[1].plot(y_data, color="#0f766e")
    _axes[1].set_ylabel("Measured y")
    _axes[1].set_xlabel("Sample")
    data_figure.tight_layout()
    data_figure
    return (data_figure,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Turn a recording into a library of trajectories

    For a scalar recording $s=[1,2,3,4,5]$ and window length $L=3$,
    $$H_3(s)=\begin{bmatrix}1&2&3\\2&3&4\\3&4&5\end{bmatrix}.$$
    Each **column** is a consecutive three-sample experiment. Columns overlap;
    rows are time offsets within a window. There are $T-L+1$ columns.

    Set $L=T_{\mathrm{ini}}+N$. The first $T_{\mathrm{ini}}$ rows describe a
    past history and the last $N$ rows describe its continuation. The same
    partition on inputs and outputs produces $U_p,U_f,Y_p,Y_f$.

    The helper below also supports arrays shaped `(samples, channels)`.
    Each block row holds all channels at one time, so flattening a history
    must use time-major order. Our actual optimization stays scalar for clarity.
    """)
    return


@app.cell
def _(np):
    def hankel(samples, length):
        values = np.asarray(samples, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or not 1 <= length <= len(values):
            raise ValueError("Expected (samples, channels) and 1 <= length <= samples")
        if not np.isfinite(values).all():
            raise ValueError("Data must be finite")
        return np.column_stack([
            values[j:j + length].reshape(-1)
            for j in range(len(values) - length + 1)
        ])

    return (hankel,)


@app.cell
def _(N, T_ini, hankel, mo, u_data, y_data):
    L = T_ini + N
    H_u, H_y = hankel(u_data, L), hankel(y_data, L)
    Up, Uf = H_u[:T_ini], H_u[T_ini:]
    Yp, Yf = H_y[:T_ini], H_y[T_ini:]
    mo.md(f"""
    With these settings, **L = {L}** and there are **{H_u.shape[1]} columns**.

    | Matrix | Shape | Meaning |
    |---|---|---|
    | Up | {Up.shape} | past inputs |
    | Yp | {Yp.shape} | past outputs |
    | Uf | {Uf.shape} | future inputs |
    | Yf | {Yf.shape} | future outputs |
    """)
    return L, Up, Uf, Yp, Yf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. When does the trajectory library suffice?

    For a controllable, noiseless LTI system of order $n$, a sufficient condition
    is that $H_{L+n}(u^d)$ has full row rank. This is persistent excitation of
    order $L+n$. For $m$ inputs, the necessary sample count for that rank is
    $$T\geq(m+1)(L+n)-1.$$
    Enough samples alone do not guarantee rank: a constant input still fails.

    Under the fundamental lemma, every length-$L$ input/output trajectory is
    a linear combination of columns of the data matrix. Its coefficients are
    $g\in\mathbb R^{T-L+1}$:
    $$
    \begin{bmatrix}U_p\\Y_p\\U_f\\Y_f\end{bmatrix}g
    =\begin{bmatrix}u_{\mathrm{ini}}\\y_{\mathrm{ini}}\\u\\y\end{bmatrix}.
    $$
    The coefficients may be negative and need not sum to one. They are not
    probabilities or fitted state-space matrices.

    The recent history fixes the hidden initial condition when
    $T_{\mathrm{ini}}$ is at least the system lag (the history length needed
    to distinguish states). An order upper bound is sufficient here: our
    system has order 2 and we use 4 past samples. Real experiments require
    a justified bound or further validation; avoiding matrix identification
    does not remove this information requirement.

    Notice that we test **the input Hankel matrix**, not full row rank of the
    stacked input/output matrix, whose rows have dynamical dependencies.
    """)
    return


@app.cell
def _(L, T, hankel, mo, n_bound, np, u_data):
    excitation = hankel(u_data, L + n_bound)
    singular_values = np.linalg.svd(excitation, compute_uv=False)
    excitation_rank = int(np.linalg.matrix_rank(excitation))
    assert excitation_rank == excitation.shape[0], "Collect richer input data"
    mo.md(f"""
    **Excitation check passed:** rank = {excitation_rank}/{excitation.shape[0]},
    smallest singular value = {singular_values[-1]:.3f}.
    The scalar-input length bound is {2 * (L + n_bound) - 1}; we collected {T}.
    A tiny smallest singular value would signal numerical sensitivity even
    when a rank routine reports full rank.
    """)
    return (excitation_rank,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. The minimal controller

    `g` chooses a trajectory from the library. `u` and `y` name its future
    input and output. The four equality constraints enforce that it starts
    with the measured history and continues consistently with the experiment.
    CVXPY turns the quadratic cost and linear constraints into a convex program.

    Read the constructor in this order:

    1. Declare optimization variables and parameters for changing measurements.
    2. Write the tracking/effort objective.
    3. Impose history, prediction, and magnitude constraints.
    4. In `plan`, supply this moment's history and reference, then solve.

    Parameters let us reuse one compiled problem at every time step. We use
    OSQP, a quadratic-program solver. We check solver status before
    accessing results. No regularization is added in this exact-data case:
    adding a penalty on $g$ would change the optimization used in the MPC comparison.
    $g$ can be nonunique even when the optimal input trajectory is unique.
    """)
    return


@app.cell
def _(cp, np):
    class DeePC:
        """Scalar input/output controller receiving data matrices only."""

        def __init__(self, Up, Yp, Uf, Yf, q, rho, u_limit, y_limit,
                     lambda_g=0.0, lambda_y=0.0):
            past, columns = Up.shape
            horizon = Uf.shape[0]
            self.g = cp.Variable(columns)
            self.u = cp.Variable(horizon)
            self.y = cp.Variable(horizon)
            self.past_u = cp.Parameter(past)
            self.past_y = cp.Parameter(past)
            self.reference = cp.Parameter(horizon)
            self.sigma = cp.Variable(past) if lambda_y > 0 else None
            history_y = self.past_y if self.sigma is None else self.past_y + self.sigma
            objective = q * cp.sum_squares(self.y - self.reference)
            objective += rho * cp.sum_squares(self.u)
            if lambda_g > 0:
                objective += lambda_g * cp.norm1(self.g)
            if self.sigma is not None:
                objective += lambda_y * cp.norm1(self.sigma)
            constraints = [
                Up @ self.g == self.past_u,
                Yp @ self.g == history_y,
                Uf @ self.g == self.u,
                Yf @ self.g == self.y,
                self.u <= u_limit, self.u >= -u_limit,
                self.y <= y_limit, self.y >= -y_limit,
            ]
            self.problem = cp.Problem(cp.Minimize(objective), constraints)
            self.data = (Up, Yp, Uf, Yf)

        def plan(self, past_u, past_y, reference):
            self.past_u.value = np.asarray(past_u)
            self.past_y.value = np.asarray(past_y)
            self.reference.value = np.asarray(reference)
            if self.sigma is None:
                self.problem.solve(solver="OSQP", warm_start=True,
                                   eps_abs=1e-6, eps_rel=1e-6, max_iter=100000,
                                   polishing=True)
            else:
                self.problem.solve(solver="CLARABEL", warm_start=True)
            if self.problem.status != cp.OPTIMAL:
                raise RuntimeError(f"DeePC solve failed: {self.problem.status}")
            self.residual = max(float(np.max(c.violation())) for c in self.problem.constraints)
            if self.residual > 1e-6:
                raise RuntimeError(f"Constraint residual too large: {self.residual:.2e}")
            return self.u.value.copy(), self.y.value.copy(), float(self.problem.value)

    return (DeePC,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Predict once before closing the loop

    We initialize a **new** experiment at a different state and apply four
    zero inputs to collect its recent history. This checks that the library
    works beyond the initial state used during data collection.

    Solve for a constant reference of 0.8. Then, for verification only, replay
    the entire planned input sequence through the simulator from the actual
    current state. Predicted and replayed outputs should agree to solver precision.
    This replay does not alter the state of the later closed-loop experiment.
    """)
    return


@app.cell
def _(DeePC, N, T_ini, Up, Uf, Yp, Yf, np, q, rho, simulate, u_limit, y_limit):
    initial_u = np.zeros(T_ini)
    initial_y, current_x = simulate(initial_u, x0=[0.2, -0.1])
    controller = DeePC(Up, Yp, Uf, Yf, q, rho, u_limit, y_limit)
    planned_u, planned_y, planned_cost = controller.plan(initial_u, initial_y, np.full(N, 0.8))
    replayed_y, _ = simulate(planned_u, current_x)
    prediction_error = float(np.max(np.abs(planned_y - replayed_y)))
    assert prediction_error < 1e-6, prediction_error
    return current_x, initial_u, initial_y, planned_cost, planned_u, planned_y, prediction_error


@app.cell(hide_code=True)
def _(mo, prediction_error):
    mo.md(f"**Prediction replay passed.** Maximum absolute error: {prediction_error:.2e}.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Apply one input, measure, and repeat

    The loop below has two distinct participants. `controller.plan` sees only
    histories and a reference window. `simulate` stands in for hardware.
    At each iteration we log the pre-update output, advance the plant once,
    and append that input/output pair to the history, discarding the oldest pair.

    The reference changes from 0.8 to -0.4 after 35 samples. The whole future
    reference is known in this example, so the controller can anticipate the
    change when it enters the horizon. If future references are unknown, fill
    the horizon with the current setpoint instead.
    """)
    return


@app.cell
def _(N, T_ini, np, simulate):
    reference = np.r_[np.full(35, 0.8), np.full(35 + N, -0.4)]

    def run_loop(controller, initial_u, initial_y, current_x, reference, noise_std=0.0):
        past_u, past_y = initial_u.copy(), initial_y.copy()
        x = current_x.copy()
        rng = np.random.default_rng(19)
        inputs, outputs, measured, predictions, costs = [], [], [], [], []
        for t in range(len(reference) - N):
            future_u, future_y, cost = controller.plan(past_u, past_y, reference[t:t + N])
            applied = float(future_u[0])
            plant_y, x = simulate([applied], x)
            observed = float(plant_y[0] + rng.normal(0.0, noise_std))
            inputs.append(applied)
            outputs.append(plant_y[0])
            measured.append(observed)
            predictions.append(future_y[0])
            costs.append(cost)
            past_u = np.r_[past_u, applied][-T_ini:]
            past_y = np.r_[past_y, observed][-T_ini:]
        return {"u": np.array(inputs), "y": np.array(outputs),
                "measured": np.array(measured), "predicted": np.array(predictions),
                "cost": np.array(costs)}

    return reference, run_loop


@app.cell
def _(DeePC, Up, Uf, Yp, Yf, current_x, initial_u, initial_y, q, reference, rho, run_loop, u_limit, y_limit):
    exact_run = run_loop(DeePC(Up, Yp, Uf, Yf, q, rho, u_limit, y_limit),
                         initial_u, initial_y, current_x, reference)
    return (exact_run,)


@app.cell
def _(exact_run, np, plt, reference, u_limit, y_limit):
    tracking_figure, _axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    _t = np.arange(len(exact_run["u"]))
    _axes[0].step(_t, reference[:len(_t)], where="post", label="Reference", color="#64748b", linestyle="--")
    _axes[0].plot(_t, exact_run["y"], label="DeePC output", color="#0f766e", linewidth=2)
    _axes[0].axhline(y_limit, color="#dc2626", linestyle=":", label="Output bounds")
    _axes[0].axhline(-y_limit, color="#dc2626", linestyle=":")
    _axes[0].set_ylabel("Output")
    _axes[0].legend(loc="lower left", ncol=3)
    _axes[1].step(_t, exact_run["u"], where="post", color="#2563eb", label="Applied input")
    _axes[1].axhline(u_limit, color="#dc2626", linestyle=":")
    _axes[1].axhline(-u_limit, color="#dc2626", linestyle=":")
    _axes[1].set(xlabel="Control sample", ylabel="Input", ylim=(-1.2, 1.2))
    tracking_figure.tight_layout()
    tracking_figure
    return (tracking_figure,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Interpret the plot.** Input saturation limits how quickly the output can
    move. Increasing $\rho$ discourages sustained input, usually slowing response
    and increasing tracking offset. Penalizing $u^2$ can trade persistent tracking
    error for reduced effort: exact setpoint tracking is not promised by this cost.
    Integral action or a suitable steady-state target formulation would be a
    separate extension.

    ## 8. Does it solve the same problem as model-based MPC?

    An independent benchmark uses the true matrices and the same state,
    horizon, cost, reference, and bounds. It is deliberately isolated from
    DeePC. We compare the full planned input/output vectors and the objective,
    rather than comparing $g$, which has no counterpart in MPC.

    This is a numerical check of this example, supporting the deterministic
    equivalence in the paper's Theorem 5.1. It is not a general proof. The
    closed-loop checks also verify output prediction, history alignment, and
    actual constraint satisfaction over this run.
    """)
    return


@app.cell
def _(A, B, C, N, cp, current_x, exact_run, np, planned_cost, planned_u, planned_y, prediction_error, q, rho, u_limit, y_limit):
    _xm = cp.Variable((2, N + 1))
    _um = cp.Variable(N)
    _ym = C @ _xm[:, :N]
    _constraints = [_xm[:, 0] == current_x,
                    _um <= u_limit, _um >= -u_limit,
                    _ym <= y_limit, _ym >= -y_limit]
    for _k in range(N):
        _constraints.append(_xm[:, _k + 1] == A @ _xm[:, _k] + B * _um[_k])
    _mpc = cp.Problem(cp.Minimize(q * cp.sum_squares(_ym - 0.8) + rho * cp.sum_squares(_um)), _constraints)
    _mpc.solve(solver="OSQP", eps_abs=1e-6, eps_rel=1e-6, max_iter=100000,
               polishing=True)
    assert _mpc.status == cp.OPTIMAL
    verification = {
        "prediction_replay_max_error": prediction_error,
        "mpc_input_max_difference": float(np.max(np.abs(_um.value - planned_u))),
        "mpc_output_max_difference": float(np.max(np.abs(_ym.value - planned_y))),
        "mpc_objective_difference": float(abs(_mpc.value - planned_cost)),
        "closed_loop_prediction_max_error": float(np.max(np.abs(exact_run["y"] - exact_run["predicted"]))),
        "input_bound_violation": float(max(0.0, np.max(np.abs(exact_run["u"])) - u_limit)),
        "output_bound_violation": float(max(0.0, np.max(np.abs(exact_run["y"])) - y_limit)),
    }
    assert verification["mpc_input_max_difference"] < 1e-4, verification
    assert verification["mpc_output_max_difference"] < 1e-4, verification
    assert verification["mpc_objective_difference"] < 1e-5, verification
    assert verification["closed_loop_prediction_max_error"] < 1e-6, verification
    assert verification["input_bound_violation"] < 1e-6, verification
    assert verification["output_bound_violation"] < 1e-6, verification
    return (verification,)


@app.cell(hide_code=True)
def _(mo, verification):
    mo.md("**Verification passed.**\n\n| Check | Absolute error / violation |\n|---|---|\n" +
          "\n".join(f"| {key.replace('_', ' ')} | {value:.2e} |" for key, value in verification.items()))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. A small measurement-noise extension

    Add noise of standard deviation 0.01 to both the offline outputs and the
    online sensor readings. Exact equalities on noisy data may become
    inconsistent or allow the optimizer to exploit spurious trajectories.

    We relax the output history to $Y_p g=y_{\mathrm{ini}}+\sigma_y$ and add
    $$\lambda_g\|g\|_1+\lambda_y\|\sigma_y\|_1$$
    to the cost. The existing constructor activates these terms when their
    weights are positive. The slack lets history matching absorb measurement
    error, while penalizing $g$ discourages large coefficient combinations.
    Both weights depend on signal units and data quality; the values here are
    illustrative, not universally optimal.

    The regularized problem uses CLARABEL: it handles this example's one-norm
    penalties more reliably than OSQP. The exact problem uses OSQP with
    polishing; its redundant trajectory equalities can be numerically delicate.
    Tightening a stopping tolerance is not a substitute for checking the actual
    prediction and constraint residuals.

    These are two ingredients of the paper's Equation (8). **We omit its
    low-rank approximation** to keep the example small. Therefore this is a
    simplified variant, not the complete regularized algorithm in that equation.
    In particular, arbitrary SVD truncation need not preserve Hankel structure.
    No exact MPC-equivalence claim is made for this noisy extension.
    """)
    return


@app.cell
def _(DeePC, L, T_ini, Up, Uf, current_x, hankel, initial_u, initial_y, np, q, reference, rho, run_loop, u_limit, y_data, y_limit):
    _rng = np.random.default_rng(23)
    _noisy_data = y_data + _rng.normal(0, 0.01, len(y_data))
    _noisy_hankel = hankel(_noisy_data, L)
    _noisy_initial = initial_y + _rng.normal(0, 0.01, T_ini)
    noisy_controller = DeePC(Up, _noisy_hankel[:T_ini], Uf, _noisy_hankel[T_ini:],
                            q, rho, u_limit, y_limit, lambda_g=1.0, lambda_y=100.0)
    noisy_run = run_loop(noisy_controller, initial_u, _noisy_initial, current_x, reference, noise_std=0.01)
    return (noisy_run,)


@app.cell
def _(exact_run, noisy_run, np, plt, reference):
    noise_figure, _axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    _t = np.arange(len(noisy_run["u"]))
    _axes[0].step(_t, reference[:len(_t)], where="post", linestyle="--", color="#64748b", label="Reference")
    _axes[0].plot(_t, exact_run["y"], color="#0f766e", label="Exact-data baseline")
    _axes[0].plot(_t, noisy_run["y"], color="#d97706", label="Noisy-data controller: true output")
    _axes[0].scatter(_t, noisy_run["measured"], s=8, alpha=0.45, color="#d97706", label="Sensor readings")
    _axes[0].set_ylabel("Output")
    _axes[0].legend(fontsize=8, loc="lower left")
    _axes[1].step(_t, noisy_run["u"], where="post", color="#d97706")
    _axes[1].axhline(1.0, linestyle=":", color="#dc2626")
    _axes[1].axhline(-1.0, linestyle=":", color="#dc2626")
    _axes[1].set(xlabel="Control sample", ylabel="Noisy-run input", ylim=(-1.2, 1.2))
    noise_figure.tight_layout()
    noise_figure
    return (noise_figure,)


@app.cell(hide_code=True)
def _(mo, noisy_run, np, reference):
    _rmse = np.sqrt(np.mean((noisy_run["y"] - reference[:len(noisy_run["y"])]) ** 2))
    mo.md(f"""
    The noisy run's true-output tracking RMSE is **{_rmse:.3f}**, including
    both transitions. True output is available here only because we simulated
    the experiment. Predicted output constraints do not guarantee that a noisy
    real plant will obey those constraints. Slack in the history also does not
    guarantee feasibility of every other constraint.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Replace the simulator with your measurements

    1. Collect synchronized input/output pairs at a fixed sample interval.
       For this scalar example, supply one-dimensional `u_data` and `y_data`
       arrays of equal length. Do not concatenate separate experiments and
       construct windows crossing their reset boundaries.
    2. Use deviation variables around a consistent operating point if necessary.
       An affine plant with a nonzero offset is not the homogeneous LTI system
       assumed by the basic equations. Apply the same transformation to data,
       online histories, references, and limits.
    3. Choose an order upper bound and history length, collect enough excitation,
       and inspect singular values. For noisy outputs, matrix rank alone cannot
       reliably reveal the true system order. Use independent validation data.
    4. Scale signals using fixed, documented units before selecting weights.
       Choose $N$ relative to the plant response time and sample interval.
    5. Fill the initial history with recent real measurements, call `plan`,
       apply its first input, then acquire the correctly aligned output.
    6. Treat solver failure as a failure, not a valid control action. This tutorial
       raises an exception. Real equipment needs a separately designed fallback
       and an experiment appropriate for its operating limits.

    **What this tutorial establishes:** exact prediction and matching finite-horizon
    MPC solutions for this noiseless example. It does not establish general
    infinite-horizon optimality, recursive feasibility, or stability. Those need
    additional assumptions/design, such as suitable terminal ingredients.

    **Try next:** replace random excitation with a constant and watch the rank
    check fail; shorten the history; increase the effort penalty; increase sensor
    noise and vary the two regularization weights. Change one thing at a time.

    ### Sources and scope

    - [Coulson, Lygeros, Dörfler (2019)](https://arxiv.org/pdf/1811.05890):
      Sections IV-V for trajectory representation and deterministic DeePC;
      Section VI-A for the regularization motivation.
    - [marimo sandbox documentation](https://docs.marimo.io/guides/package_management/sandboxes/):
      inline dependency metadata and reproducible execution.
    - [Open notebooks from GitHub in molab](https://molab.marimo.io/github).

    The simulator, parameters, instructional text, and verification experiments
    in this notebook are an original minimal example. No external dataset or
    local helper module is required.
    """)
    return


if __name__ == "__main__":
    app.run()
