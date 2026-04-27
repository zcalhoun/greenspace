from datetime import datetime

import torch


class SimpleLogger:
    """
    A simple logger class that provides methods to log messages with a prefix and timestamp.
    """

    def __init__(self, prefix="LOG"):
        # Initializes the logger with a prefix and sets the initial time.
        self.prefix = prefix
        self.time = datetime.now()

    def log(self, message):
        """
        Logs a message with the current timestamp and the specified prefix.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{self.prefix} {timestamp}] {message}", flush=True)

    def info(self, message):
        """
        Logs an informational message.
        """
        self.log(f"INFO: {message}")

    def warn(self, message):
        # Logs a warning message.
        self.log(f"WARNING: {message}")

    def error(self, message):
        # Logs an error message.
        self.log(f"ERROR: {message}")

    def start_timer(self, timer_name):
        # Starts a timer and logs the start time with the given timer name.
        self.time = datetime.now()
        self.log(f"{timer_name}: Timer started.")

    def stop_timer(self, timer_name):
        # Stops the timer, calculates the elapsed time in minutes, and logs it with the timer name.
        elapsed_time = datetime.now() - self.time
        elapsed_time = elapsed_time.total_seconds()
        elapsed_time /= 60
        elapsed_time = round(elapsed_time, 2)
        self.log(f"{timer_name}: Timer stopped. Elapsed time: {elapsed_time} minutes.")


def lstsq_init(model, train_loader, device, l2_penalty, init_l2_min=1e-2):
    """
    Initialize model.beta, model.elevation_weight, and model.intercept via
    ridge regression using the fixed initial exp_weight.

    Assembles the full feature matrix [e | point | linear_terms | 1] over the
    training set and solves the L2-regularized normal equations:

        (X^T X + λ_init * I) w = X^T y

    λ_init = max(l2_penalty, init_l2_min) decouples initialization stability
    from the training penalty: small training lambdas produce near-singular
    systems whose solutions have large coefficients and a large initial loss,
    even though the collinearity means those coefficients cancel in practice.
    A stronger initialization lambda gives a stable, moderate-norm starting
    point that pre-training can then refine.

    The intercept column is excluded from regularization.
    """
    init_lambda = max(l2_penalty, init_l2_min)
    model.eval()
    all_X = []
    all_y = []

    with torch.no_grad():
        for c, e, s, y_batch in train_loader:
            e, s, y_batch = e.to(device), s.to(device), y_batch.to(device)
            point = s[:, :, model.size // 2, model.size // 2]
            linear_terms = model.exp_weight(s.flatten(start_dim=2))
            ones = torch.ones(y_batch.size(0), 1, device=device)
            X = torch.cat([e, point, linear_terms, ones], dim=1)
            all_X.append(X.cpu().float())
            all_y.append(y_batch.cpu().float())

    X = torch.cat(all_X, dim=0)  # (N, 1 + num_dims*2 + 1)
    y = torch.cat(all_y, dim=0)  # (N,)

    # Ridge normal equations: (X^T X + λ_init * I) w = X^T y
    # Don't regularize the intercept (last column)
    reg = init_lambda * torch.eye(X.shape[1])
    reg[-1, -1] = 0.0
    A = X.T @ X + reg
    b = X.T @ y
    solution = torch.linalg.solve(A, b)

    # Unpack: elevation_weight (1) | beta (num_dims*2) | intercept (1)
    num_beta = model.beta.shape[0]
    model.elevation_weight.data.copy_(solution[:1])
    model.beta.data.copy_(solution[1 : 1 + num_beta])
    model.intercept.data.copy_(solution[-1])


def farthest_point_sample(coords, residuals, k):
    """
    Select k points from coords using farthest-point sampling seeded by the
    highest residual. Each subsequent point maximises the minimum distance to
    the already-selected set, so the result is both spatially spread and
    biased toward high-error regions.

    Args:
        coords:    (N, 2) float tensor of standardized coordinates
        residuals: (N,)   float tensor of absolute residuals
        k:         number of points to select

    Returns:
        (k, 2) float tensor of selected coordinates
    """
    n = coords.size(0)
    selected = [residuals.argmax().item()]
    min_dists = torch.full((n,), float("inf"))

    while len(selected) < k:
        last = coords[selected[-1]]
        dists = ((coords - last) ** 2).sum(dim=1)
        min_dists = torch.minimum(min_dists, dists)
        selected.append(min_dists.argmax().item())

    return coords[torch.tensor(selected)]
