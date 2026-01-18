
import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import clone
from joblib import Parallel, delayed
from typing import Tuple, Optional


# ------------------------- utilities -------------------------
def _ridge(x: float, eps: float = 1e-12) -> float:
    "Stabilize denominator to avoid division by ~0."
    if abs(x) < eps:
        return np.sign(x) * eps if x != 0 else eps
    return x


def _crossfit_once_predict(
    x: np.ndarray,
    y: np.ndarray,
    estimator,
    kf: KFold,
    n_jobs: int,
) -> np.ndarray:
    """
    Cross-fit a single estimator once and return stacked out-of-fold predictions.
    """
    n = x.shape[0]
    y = y.reshape(-1)
    preds = np.full(n, np.nan, dtype=float)

    def _fit_predict_fold(train_idx, test_idx):
        est = clone(estimator)
        est.fit(x[train_idx], y[train_idx])
        return test_idx, est.predict(x[test_idx])

    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_predict_fold)(train_idx, test_idx) for train_idx, test_idx in kf.split(x)
    )
    for test_idx, p in results:
        preds[test_idx] = p
    return preds


def _crossfit_each_iter_predict_g(
    x: np.ndarray,
    y_minus_theta_d: np.ndarray,
    estimator,
    kf: KFold,
    n_jobs: int,
) -> np.ndarray:
    """
    Cross-fit g(x) each iteration with current target (y - theta*d).
    """
    return _crossfit_once_predict(x, y_minus_theta_d, estimator, kf, n_jobs)


# ======================== Unweighted IV-type ========================
def fit_plr_ivtype_iterative(
    x: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    ml_m,
    ml_g,
    n_folds: int = 2,
    max_iter: int = 50,
    tol: float = 1e-7,
    theta0: float = 0.0,
    random_state: Optional[int] = None,
    n_jobs: int = 1,
) -> Tuple[float, float, dict]:
    """
    Iterative IV-type PLR (unweighted).
    Improvements:
      - m(x) is cross-fitted ONCE (independent of theta) and fixed across iterations.
      - g(x) is cross-fitted at EACH iteration since it depends on theta.
      - Cross-fitting folds can be parallelized via joblib (n_jobs).
    """
    x = np.asarray(x); y = np.asarray(y).ravel(); d = np.asarray(d).ravel()
    n = x.shape[0]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # ---- Step 1: cross-fit m(x) once ----
    m_hat = _crossfit_once_predict(x, d, ml_m, kf, n_jobs)
    v_hat = d - m_hat

    # ---- Step 2: iterate theta while re-fitting g(x) each iteration ----
    theta = float(theta0)
    g_hat = np.full(n, np.nan)
    it = 0
    for it in range(max_iter):
        target = y - theta * d
        g_hat = _crossfit_each_iter_predict_g(x, target, ml_g, kf, n_jobs)

        psi_a = - v_hat * d
        psi_b = v_hat * (y - g_hat)
        denom = _ridge(np.mean(psi_a))
        num = np.mean(psi_b)
        theta_new = - num / denom

        if np.isfinite(theta_new) and abs(theta_new - theta) < tol:
            theta = theta_new
            break
        theta = theta_new

    # ---- variance & se ----
    psi_a = - v_hat * d
    psi_b = v_hat * (y - g_hat)
    psi = psi_a * theta + psi_b
    var_psi = np.mean((psi - np.mean(psi))**2)
    se = np.sqrt(var_psi / n) / abs(_ridge(np.mean(psi_a)))

    extras = {
        "iterations": it + 1,
        "converged": np.isfinite(theta),
        "n_folds": n_folds,
        "n_jobs": n_jobs,
        "v_hat": v_hat,
        "m_hat": m_hat,
        "g_hat": g_hat,
    }
    return theta, se, extras


# ================= Weighted (expectile-style) IV-type =================
def fit_plr_ivtype_iterative_weighted(
    x: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    ml_m,
    ml_g,
    tau: float,
    n_folds: int = 2,
    max_iter: int = 200,
    tol: float = 1e-7,
    theta0: float = 0.0,
    random_state: Optional[int] = None,
    n_jobs: int = 1,
) -> Tuple[float, float, dict]:
    """
    Iterative IV-type PLR with asymmetric least-squares (expectile-style) weights.
    At each iteration t:
      1) m(x) = E[D|X] is cross-fitted ONCE outside the loop and kept fixed.
      2) g(x) = E[Y - D*theta_t | X] is cross-fitted within the iteration.
      3) Residual r_i(theta_t) = Y_i - D_i*theta_t - g_hat(X_i).
      4) Weights w_i = |tau - 1{ r_i(theta_t) < 0 }| = tau for r>=0, 1-tau otherwise.
      5) Weighted IV-type update:
             theta_{t+1} = - mean( w * v_hat * (Y - g_hat) ) / mean( - w * v_hat * D ),
         with v_hat = D - m_hat.
    """
    x = np.asarray(x); y = np.asarray(y).ravel(); d = np.asarray(d).ravel()
    n = x.shape[0]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # ---- Step 1: cross-fit m(x) once ----
    m_hat = _crossfit_once_predict(x, d, ml_m, kf, n_jobs)
    v_hat = d - m_hat

    # ---- Step 2: iterate theta while re-fitting g(x) each iteration ----
    theta = float(theta0)
    g_hat = np.full(n, np.nan)
    it = 0
    for it in range(max_iter):
        target = y - theta * d
        g_hat = _crossfit_each_iter_predict_g(x, target, ml_g, kf, n_jobs)

        resid = y - d * theta - g_hat
        w = np.where(resid < 0, 1.0 - tau, tau)

        num = np.mean(w * v_hat * (y - g_hat))
        den = _ridge(np.mean(- w * v_hat * d))
        theta_new = - num / den

        if np.isfinite(theta_new) and abs(theta_new - theta) < tol:
            theta = theta_new
            break
        theta = theta_new

    # ---- variance & se (weighted) ----
    resid = y - d * theta - g_hat
    w = np.where(resid < 0, 1.0 - tau, tau)
    psi_a = - w * v_hat * d
    psi_b = w * v_hat * (y - g_hat)
    psi = psi_a * theta + psi_b
    var_psi = np.mean((psi - np.mean(psi))**2)
    se = np.sqrt(var_psi / n) / abs(_ridge(np.mean(psi_a)))

    extras = {
        "iterations": it + 1,
        "converged": np.isfinite(theta),
        "n_folds": n_folds,
        "n_jobs": n_jobs,
        "tau": tau,
        "v_hat": v_hat,
        "m_hat": m_hat,
        "g_hat": g_hat,
    }
    return theta, se, extras


# ================= Weighted (expectile-style) nonorthogonal  =================
import numpy as np
from sklearn.base import clone
from typing import Tuple

def fit_plr_nonorthogonal_weighted(
    x: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    ml,
    tau: float,
    n_folds: int = 2,
    n_jobs: int = 1,
    max_iter: int = 50,
    tol: float = 1e-7,
    theta0: float = 0.0,
    random_state: Optional[int] = None
) -> Tuple[float, float, dict]:
    """
    Non-orthogonal PLR weighted estimator (expectile-style) using KFold cross-fitting for g(x).
    """
    x = np.asarray(x); y = np.asarray(y).ravel(); d = np.asarray(d).ravel()
    n = x.shape[0]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    g_hat = _crossfit_once_predict(x, y, ml, kf, n_jobs)

    theta = float(theta0)
    for it in range(max_iter):
        resid = y - d * theta - g_hat
        w = np.abs(tau - (resid < 0).astype(float))

        num = np.mean(w * d * (y - g_hat))
        den = np.mean(w * d**2)
        theta_new = num / _ridge(den)

        if np.isfinite(theta_new) and abs(theta_new - theta) < tol:
            theta = theta_new
            break
        theta = theta_new

    # ---- variance & se ----
    resid = y - d * theta - g_hat
    w = np.abs(tau - (resid < 0).astype(float))
    psi_a = w * d**2
    psi_b = w * d * (y - g_hat)
    psi = psi_b / np.mean(psi_a)
    var_psi = np.mean((psi - np.mean(psi))**2)
    se = np.sqrt(var_psi / n)

    extras = {
        "iterations": it + 1,
        "converged": np.isfinite(theta),
        "tau": tau,
        "g_hat": g_hat,
        "n_folds": n_folds,
        "n_jobs": n_jobs
    }
    return theta, se, extras


from sklearn.base import clone
from typing import Tuple
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import root_scalar
from typing import Union
import warnings

def normal_expectile(tau, tol=1e-8):
    if not (0 < tau < 1):
        raise ValueError("tau 必须在 (0, 1) 之间")

    def f(mu):
        left = quad(lambda y: (mu - y) * norm.pdf(y), -np.inf, mu, epsabs=tol)[0]
        right = quad(lambda y: (y - mu) * norm.pdf(y), mu, np.inf, epsabs=tol)[0]
        return tau / (1 - tau) - left / right

    initial_guess = norm.ppf(tau)

    try:
        sol = root_scalar(f, x0=initial_guess, bracket=[-10, 10], method='brentq')
        if sol.converged:
            return sol.root
    except ValueError:
        pass

    mu = initial_guess
    for _ in range(1000):
        left = quad(lambda y: (mu - y) * norm.pdf(y), -np.inf, mu, epsabs=tol)[0]
        right = quad(lambda y: (y - mu) * norm.pdf(y), mu, np.inf, epsabs=tol)[0]
        ratio = left / right
        diff = tau / (1 - tau) - ratio
        if abs(diff) < tol:
            break
        mu += diff * 0.1
    return mu


class ExpectileRegression:

    def __init__(self, tau: float = 0.5, fit_intercept: bool = True,
                 max_iter: int = 100, tol: float = 1e-8, reg_lambda: float = 1e-8):
        if not (0 < tau < 1):
            raise ValueError("tau must be in the range (0, 1)")

        self.tau = tau
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.reg_lambda = reg_lambda

        self.coef_ = None
        self.intercept_ = None
        self.n_features_in_ = None
        self.n_iter_ = None
        self.is_fitted_ = False

        self.loss_history_ = []
        self.convergence_info_ = {}

    def _asymmetric_squared_loss(self, residuals: np.ndarray) -> np.ndarray:
        indicator = (residuals < 0).astype(float)

        weights = np.abs(self.tau - indicator)

        return weights * (residuals ** 2)

    def _asymmetric_weights(self, residuals: np.ndarray) -> np.ndarray:
        indicator = (residuals < 0).astype(float)
        weights = np.abs(self.tau - indicator)

        weights = np.maximum(weights, 1e-10)

        return weights

    def _compute_loss(self, residuals: np.ndarray) -> float:
        losses = self._asymmetric_squared_loss(residuals)
        return np.mean(losses)

    def fit(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list]) -> 'ExpectileRegression':

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim != 1:
            raise ValueError("y should be a 1d array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        if self.fit_intercept:
            X_with_intercept = np.column_stack([np.ones(n_samples), X])
            n_params = n_features + 1
        else:
            X_with_intercept = X.copy()
            n_params = n_features

        try:
            params = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            params = np.zeros(n_params)
            warnings.warn("使用零初始化参数，可能影响收敛速度")

        self.loss_history_ = []
        prev_loss = np.inf

        for iteration in range(self.max_iter):
            y_pred = X_with_intercept @ params
            residuals = y - y_pred

            current_loss = self._compute_loss(residuals)
            self.loss_history_.append(current_loss)

            weights = self._asymmetric_weights(residuals)

            W = np.diag(weights)

            XTW = X_with_intercept.T @ W
            XTWX = XTW @ X_with_intercept
            XTWy = XTW @ y

            reg_matrix = self.reg_lambda * np.eye(n_params)

            try:
                new_params = np.linalg.solve(XTWX + reg_matrix, XTWy)
            except np.linalg.LinAlgError:
                new_params = np.linalg.pinv(XTWX + reg_matrix) @ XTWy
                warnings.warn("使用伪逆求解，可能数值不稳定")

            loss_change = abs(prev_loss - current_loss)

            param_change = np.linalg.norm(new_params - params)

            if loss_change < self.tol or param_change < self.tol:
                params = new_params
                break

            params = new_params
            prev_loss = current_loss

        else:
            warnings.warn(f"IRLS未在 {self.max_iter} 次迭代内收敛")
            print(f"达到最大迭代次数 {self.max_iter}")

        self.n_iter_ = iteration + 1

        if self.fit_intercept:
            self.intercept_ = params[0]
            self.coef_ = params[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = params

        self.convergence_info_ = {
            'converged': iteration < self.max_iter - 1,
            'final_loss': current_loss,
            'loss_change': loss_change if 'loss_change' in locals() else None,
            'param_change': param_change if 'param_change' in locals() else None
        }

        self.is_fitted_ = True
        return self

    def predict(self, X: Union[np.ndarray, list]) -> np.ndarray:
        if not self.is_fitted_:
            raise ValueError("模型尚未训练，请先调用 fit() 方法")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but model was trained on {self.n_features_in_} features")

        y_pred = X @ self.coef_
        if self.fit_intercept:
            y_pred += self.intercept_

        return y_pred

    def score(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list]) -> float:
        if not self.is_fitted_:
            raise ValueError("模型尚未训练，请先调用 fit() 方法")

        y = np.asarray(y, dtype=float)
        y_pred = self.predict(X)

        # 计算 R²
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        return r2

    def asymmetric_score(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list]) -> float:
        if not self.is_fitted_:
            raise ValueError("模型尚未训练，请先调用 fit() 方法")

        y = np.asarray(y, dtype=float)
        y_pred = self.predict(X)
        residuals = y - y_pred

        return self._compute_loss(residuals)

    def get_params(self, deep: bool = True) -> dict:
        return {
            'tau': self.tau,
            'fit_intercept': self.fit_intercept,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'reg_lambda': self.reg_lambda
        }

    def set_params(self, **params) -> 'ExpectileRegression':
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key}")
        return self



from typing import Union, Callable, List, Optional


class NonlinearExpectileRegression:

    def __init__(self,
                 tau: float = 0.5,
                 basis_functions: Optional[List[Callable]] = None,
                 fit_intercept: bool = True,
                 max_iter: int = 100,
                 tol: float = 1e-8,
                 reg_lambda: float = 1e-8):
        if not (0 < tau < 1):
            raise ValueError("tau must be in the range (0, 1)")

        self.tau = tau
        self.basis_functions = basis_functions
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.reg_lambda = reg_lambda

        self.coef_ = None
        self.intercept_ = None
        self.n_features_in_ = None
        self.n_basis_ = None
        self.n_iter_ = None
        self.is_fitted_ = False

        self.loss_history_ = []
        self.convergence_info_ = {}

    def _transform_features(self, X: np.ndarray) -> np.ndarray:

        if self.basis_functions is None:
            return X

        n_samples = X.shape[0]
        n_basis = len(self.basis_functions)

        transformed = np.zeros((n_samples, n_basis))
        for i, func in enumerate(self.basis_functions):
            try:
                result = func(X)
                # 确保结果是1维的
                if result.ndim > 1:
                    result = result.flatten()
                transformed[:, i] = result
            except Exception as e:
                raise ValueError(f"基函数 {i} 执行失败: {str(e)}")

        return transformed

    def _asymmetric_squared_loss(self, residuals: np.ndarray) -> np.ndarray:
        indicator = (residuals < 0).astype(float)
        weights = np.abs(self.tau - indicator)
        return weights * (residuals ** 2)

    def _asymmetric_weights(self, residuals: np.ndarray) -> np.ndarray:
        indicator = (residuals < 0).astype(float)
        weights = np.abs(self.tau - indicator)
        weights = np.maximum(weights, 1e-10)
        return weights

    def _compute_loss(self, residuals: np.ndarray) -> float:
        losses = self._asymmetric_squared_loss(residuals)
        return np.mean(losses)

    def fit(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list]) -> 'NonlinearExpectileRegression':

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim != 1:
            raise ValueError("y应该是1维数组")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X和y必须有相同的样本数")

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        X_transformed = self._transform_features(X)
        self.n_basis_ = X_transformed.shape[1]

        if self.fit_intercept:
            X_augmented = np.column_stack([np.ones(n_samples), X_transformed])
            n_params = self.n_basis_ + 1
        else:
            X_augmented = X_transformed.copy()
            n_params = self.n_basis_

        try:
            params = np.linalg.lstsq(X_augmented, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            params = np.zeros(n_params)
            warnings.warn("使用零初始化参数，可能影响收敛速度")

        self.loss_history_ = []
        prev_loss = np.inf

        for iteration in range(self.max_iter):
            y_pred = X_augmented @ params
            residuals = y - y_pred

            current_loss = self._compute_loss(residuals)
            self.loss_history_.append(current_loss)

            weights = self._asymmetric_weights(residuals)

            W = np.diag(weights)
            XTW = X_augmented.T @ W
            XTWX = XTW @ X_augmented
            XTWy = XTW @ y

            reg_matrix = self.reg_lambda * np.eye(n_params)

            try:
                new_params = np.linalg.solve(XTWX + reg_matrix, XTWy)
            except np.linalg.LinAlgError:
                new_params = np.linalg.pinv(XTWX + reg_matrix) @ XTWy
                warnings.warn("使用伪逆求解，可能数值不稳定")

            loss_change = abs(prev_loss - current_loss)
            param_change = np.linalg.norm(new_params - params)

            if loss_change < self.tol or param_change < self.tol:
                params = new_params
                break

            params = new_params
            prev_loss = current_loss

        else:
            warnings.warn(f"IRLS未在 {self.max_iter} 次迭代内收敛")

        self.n_iter_ = iteration + 1

        if self.fit_intercept:
            self.intercept_ = params[0]
            self.coef_ = params[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = params

        self.convergence_info_ = {
            'converged': iteration < self.max_iter - 1,
            'final_loss': current_loss,
            'loss_change': loss_change if 'loss_change' in locals() else None,
            'param_change': param_change if 'param_change' in locals() else None
        }

        self.is_fitted_ = True
        return self

    def predict(self, X: Union[np.ndarray, list]) -> np.ndarray:

        if not self.is_fitted_:
            raise ValueError("模型尚未训练，请先调用 fit() 方法")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X有 {X.shape[1]} 个特征，但模型是用 {self.n_features_in_} 个特征训练的")

        X_transformed = self._transform_features(X)

        y_pred = X_transformed @ self.coef_
        if self.fit_intercept:
            y_pred += self.intercept_

        return y_pred

    def score(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list]) -> float:

        if not self.is_fitted_:
            raise ValueError("模型尚未训练，请先调用 fit() 方法")

        y = np.asarray(y, dtype=float)
        y_pred = self.predict(X)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        return r2

    def asymmetric_score(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list]) -> float:

        if not self.is_fitted_:
            raise ValueError("模型尚未训练，请先调用 fit() 方法")

        y = np.asarray(y, dtype=float)
        y_pred = self.predict(X)
        residuals = y - y_pred

        return self._compute_loss(residuals)

    def get_params(self, deep: bool = True) -> dict:
        return {
            'tau': self.tau,
            'basis_functions': self.basis_functions,
            'fit_intercept': self.fit_intercept,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'reg_lambda': self.reg_lambda
        }

    def set_params(self, **params) -> 'NonlinearExpectileRegression':
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"无效参数 {key}")
        return self