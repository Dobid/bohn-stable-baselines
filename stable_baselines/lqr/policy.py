import numpy as np
import scipy.linalg
import casadi


class LQR:
    def __init__(self, A, B, Q, R, weights=None, input_shape=None, output_shape=(1,), eps=1e-8, **kwargs):
        self.E = None  # eigenvalues
        self.eps = eps

        self.A = casadi.SX.sym("A", *A.shape)
        self.A_num = A
        self.B = casadi.SX.sym("B", *B.shape)
        self.B_num = B
        self.cQ = casadi.SX.sym("Q", *A.shape)
        if len(Q.shape) > 1:
            try:
                self.cQ_num = np.linalg.cholesky(Q)
            except np.linalg.LinAlgError:
                self.cQ_num = np.linalg.cholesky(Q + self.eps * np.eye(Q.shape[0]))  # If still fails, notify user
        else:
            self.cQ_num = np.sqrt(Q)
        self.cR = casadi.SX.sym("R", B.shape[1], B.shape[1])
        if len(R.shape) > 1:
            try:
                self.cR_num = np.linalg.cholesky(R)
            except np.linalg.LinAlgError:
                self.cR_num = np.linalg.cholesky(R + self.eps * np.eye(R.shape[0]))
        else:
            self.cR_num = np.sqrt(R)
        self.K = casadi.SX.sym("K", B.shape[1], A.shape[1])
        self.K_num = None
        self.S = casadi.SX.sym("S", *A.shape)
        self.S_num = None
        self._compute_lqr()
        self.S_eq = self.A.T @ self.S @ self.A - self.S - self.A.T @ self.S @ self.B @ self.K + self.cQ @ self.cQ.T
        self.S_fun = casadi.Function("S_eq", [self.A, self.B, self.cQ, self.cR, self.S, self.K], [self.S_eq])
        self.K_eq = (self.B.T @ self.S @ self.B + self.cR @ self.cR.T) @ self.K - self.B.T @ self.S @ self.A
        self.K_fun = casadi.Function("K_eq", [self.A, self.B, self.cQ, self.cR, self.S, self.K], [self.K_eq])
        self.lqr_eq_vec = casadi.vertcat(casadi.reshape(self.S_eq, -1, 1), casadi.reshape(self.K_eq, -1, 1))

        w = []
        lqr_eq_parameters = []
        for sym, num, name in [(self.cQ, self.cQ_num, "Q"), (self.cR, self.cR_num, "R")]:
            if num.size == 1:
                w.append(num.ravel())
                lqr_eq_parameters.append(casadi.reshape(sym, -1, 1))
            else:
                #w.append(np.sqrt(num[np.triu_indices_from(num)]))
                #lqr_eq_parameters.append(casadi.reshape(sym, -1, 1))
                for c in range(sym.shape[1]):
                    w.append(num[c:, c].ravel())
                    lqr_eq_parameters.append(casadi.reshape(sym[c:, c], -1, 1))

        self.lqr_eq_parameters = lqr_eq_parameters = casadi.vertcat(*lqr_eq_parameters)

        self._w = np.concatenate(w)  # Q and R elements as vector

        # Implicit Function Theorem
        self.lqr_eq_unknowns = lqr_eq_unknowns = casadi.vertcat(casadi.reshape(self.S, -1, 1), casadi.reshape(self.K, -1, 1))
        self.df_dx = df_dx = casadi.jacobian(self.lqr_eq_vec, lqr_eq_unknowns)
        self.df_dy = df_dy = casadi.jacobian(self.lqr_eq_vec, lqr_eq_parameters)
        self.dx_dy = -casadi.inv(df_dx) @ df_dy
        self.df_dx_fun = casadi.Function("LQR_dx", [self.A, self.B, self.cQ, self.cR, self.S, self.K], [self.df_dx])
        #self.df_dx_det_fun = casadi.Function("LQR_dx_det", [self.A, self.B, self.Q, self.R, self.S, self.K], [casadi.det(self.df_dx)])
        self.dx_dy_fun = casadi.Function("LQR_grad", [self.A, self.B, self.cQ, self.cR, self.S, self.K], [self.dx_dy])
        #self._add_save_attr(_w='numpy')
        """
        #Non vec
        self.lqr_eq = casadi.vertcat(self.K_eq, self.S_eq)
        lqr_x = casadi.vertcat(self.K, self.S)
        self.m_df_dx = casadi.jacobian(self.lqr_eq, lqr_x)
        self.m_df_dQ = casadi.jacobian(self.lqr_eq, self.Q)
        self.m_df_dR = casadi.jacobian(self.lqr_eq, self.R)
        self.m_dx_dQ = -casadi.inv(self.m_df_dx) @ self.m_df_dQ
        self.m_dx_dR = casadi.inv(self.m_df_dx) @ self.m_df_dR
        self.m_dx_dQ_fun = casadi.Function("LQRm_grad", [self.A, self.B, self.Q, self.R, self.S, self.K], [self.m_dx_dQ])
        self.m_dx_dR_fun = casadi.Function("LQRm_grad", [self.A, self.B, self.Q, self.R, self.S, self.K], [self.m_dx_dR])
        """

    def set_numeric_value(self, components):
        assert isinstance(components, dict)
        for k, v in components.items():
            if k in ["Q", "R"]:
                k = "c" + k
                if len(v.shape) > 1:
                    try:
                        v = np.linalg.cholesky(v)
                    except np.linalg.LinAlgError:
                        v = np.linalg.cholesky(v + self.eps * np.eye(v.shape[0]))
                else:
                    v = np.sqrt(v)
            assert hasattr(self, k)
            setattr(self, "{}_num".format(k), v)
        self._compute_lqr()

    def get_numeric_value(self, component_name):
        assert isinstance(component_name, str)
        if component_name in ["Q", "R"]:
            component_name = "c" + component_name
        v = getattr(self, component_name + "_num")
        if component_name in ["cQ", "cR"]:
            if len(v.shape) > 1:
                v = v @ v.T
            else:
                v = v * v

        return v

    def get_policy_gradient(self, x):
        #assert self.df_dx_det_fun(self.A_num, self.B_num, self.Q_num, self.R_num, self.S_num, self.K_num) != 0, "dF/dx is singular, IFT not applicable"
        return -self._grad_K().T @ x

    def diff_log(self, state, action=None):
        """
        Compute the derivative of the output w.r.t. ``state``, and ``action``
        if provided.

        Args:
            state (np.ndarray): the state;
            action (np.ndarray, None): the action.

        Returns:
            The derivative of the output w.r.t. ``state``, and ``action``
            if provided.

        """
        raise NotImplementedError

    def _grad_K(self):
        dKS_dQR = self.dx_dy_fun(self.A_num, self.B_num, self.cQ_num, self.cR_num, self.S_num, self.K_num)
        dK_dQR = dKS_dQR[-np.product(self.K.shape):, :]
        return dK_dQR.toarray()
        """  # Matrix version
        dKS_dQ = self.m_dx_dQ_fun(self.A_num, self.B_num, self.Q_num, self.R_num, self.S_num, self.K_num)
        dKS_dR = self.m_dx_dR_fun(self.A_num, self.B_num, self.Q_num, self.R_num, self.S_num, self.K_num)
        K_inds = [i for i in range(dKS_dR.shape[0]) if i % (self.S.shape[0] + 1) == 0]
        dK_dR = dKS_dR[K_inds]
        dK_dQ = dKS_dQ[K_inds, :]
        """

    def predict(self, x, **predict_params):
        #prediction = np.ones((x.shape[0], self.K.shape[0]))
        #for i, x_i in enumerate(x):
        #    prediction[i] = -self.K_num @ x_i
        prediction = -self.K_num @ x
        return np.atleast_2d(prediction)

    def get_action(self, x):
        return -self.K_num @ x

    def diff(self, x):
        return self.get_policy_gradient(x)

    def _compute_lqr(self):  # TODO: supress pending deprecated warnings about matrix
        """Solve the discrete time lqr controller.

        dx/dt = A x + B u

        cost = integral x.T*Q*x + u.T*R*u
        """

        # ref Bertsekas, p.151
        try:
            # first, try to solve the ricatti equation
            Q_num = self.cQ_num @ self.cQ_num.T
            R_num = self.cR_num @ self.cR_num.T
            S = np.matrix(scipy.linalg.solve_discrete_are(self.A_num, self.B_num, Q_num, R_num))
            self.S_num = S.A

            # compute the LQR gain
            #self.K = np.matrix(scipy.linalg.inv(np.atleast_2d(self.R)) * (self.B.T * self.S))
            self.K_num = np.matrix(scipy.linalg.inv(np.atleast_2d(self.B_num.T @ S @ self.B_num + R_num)) @ (self.B_num.T @ S @ self.A_num)).A

            self.E, eigVecs = scipy.linalg.eig(self.A_num - self.B_num @ self.K_num)
        except np.linalg.LinAlgError as e:
            print("LinalgError for system:")
            for comp_name in ["A", "B", "cQ", "cR"]:
                print(comp_name)
                print(getattr(self, "{}_num".format(comp_name)))
            raise e

    @property
    def weights_size(self):
        """
        Returns:
            The size of the array of weights.

        """
        return self._w.size

    def get_weights(self):
        """
        Getter.

        Returns:
            The set of weights of the approximator.

        """
        return self._w.flatten()

    def set_weights(self, w):
        """
        Setter.

        Args:
            w (np.ndarray): the set of weights to set.

        """
        def set_matrix_weights(matrix, weights):
            if matrix.size == 1:
                matrix[0] = weights
            else:
                #num[np.triu_indices_from(num)] = np.square(new_w)
                #num[np.tril_indices_from(num)] = np.square(new_w)
                start_idx = 0
                for c in range(matrix.shape[1]):
                    c_elements = matrix.shape[1] - c
                    matrix[c:, c] = weights[start_idx:start_idx + c_elements]
                    start_idx += c_elements
                #matrix = np.tril(matrix) + np.triu(matrix.T, 1)
            return matrix
        self._w = w.reshape(self._w.shape)
        w_q, w_r = np.split(self._w, [int(self.cQ_num.shape[0] * (self.cQ_num.shape[0] + 1) / 2)])
        self.cQ_num = set_matrix_weights(self.cQ_num, w_q)
        self.cR_num = set_matrix_weights(self.cR_num, w_r)

        self._compute_lqr()

    def get_weight_names(self):
        all_params = str(self.lqr_eq_parameters).strip("[]").split(",")
        weight_names = []
        for w_n in all_params:
            w_n = w_n.strip(" ")
            if "_" in w_n:
                idx = int(w_n.split("_")[1])
                matrix_size = self.cQ_num.shape[0] if w_n[0] == "Q" else self.cR_num.shape[0]
                w_n = "{}_{}{}".format(w_n[0], idx % matrix_size + 1, idx // matrix_size + 1)
                if int(w_n[-1]) > int(w_n[-2]):
                    continue
            weight_names.append(w_n)
        return weight_names


def gradient_check(system, print_mode, n=5, eps=1e-4, tolerance=1e-6, seed=None):
    import copy
    def perturb_weight(_A, _B, _Q, _R, idx, _eps=1e-4):
        def print_results(K_p, K_n, _grad, name):
            num_grad = (K_p - K_n) / (2 * _eps)
            _grad = _grad.reshape(*num_grad.shape, order="F")
            is_close = np.isclose(_grad, num_grad, atol=tolerance)
            if print_mode == "all" or (print_mode == "failure" and not np.all(is_close)):
                print(name)
                print("np.isclose: \n{}".format(is_close))
                print("(K_p - K_n) / 2eps = \n{}".format(num_grad))
                print("Grad = \n{}".format(_grad))
                print("Diff: \n{}".format(num_grad - _grad))
                print("-" * 30)
        pi = LQR(_A, _B, _Q, _R)
        grad = pi._grad_K()
        ws = np.copy(pi.get_weights())
        ws[idx] += eps
        pi.set_weights(ws)
        Kp = np.copy(pi.K_num)
        ws[idx] -= 2*eps
        pi.set_weights(ws)
        Kn = np.copy(pi.K_num)
        w_names = pi.get_weight_names()
        print_results(Kp, Kn, grad[:, idx], w_names[idx])

    assert print_mode in ["all", "failure"]
    pi = LQR(**system)

    if seed is not None:
        np.random.seed(seed)

    for n_i in range(n):
        success = False
        while not success:
            try:
                _system = copy.deepcopy(system)
                if n_i > 0:
                    rand_comp = np.random.choice(["A", "B", "Q", "R"])
                    _system[rand_comp] = np.random.uniform(-3, 3, size=_system[rand_comp].shape)
                    if rand_comp in ["Q", "R"]:
                        _system[rand_comp] = (_system[rand_comp] + _system[rand_comp].T) / 2
                        if np.any(np.linalg.eigvals(np.atleast_2d(_system[rand_comp])) < 0):
                            raise np.linalg.LinAlgError
                pi.set_numeric_value(_system)
                for w_i in range(pi.weights_size):
                    perturb_weight(*_system.values(), w_i, _eps=eps)
                success = True
            except np.linalg.LinAlgError as e:
                pass

if __name__ == "__main__":  # TODO: fix bug where R bigger than dim 1 doesnt work.
    test_cases = {
        "2x1": {
            "A": np.array([[1, 1], [1, 1]], dtype=np.float32),
            "B": np.array([[1], [0.5]], dtype=np.float32),
            "Q": np.array([[0.5, 0], [0, 0.75]], dtype=np.float32),
            "R": np.array([1.05], dtype=np.float32),
        },
        "4x1": {
            "A": np.array([[1., 0., 7.89585366, 0.], [0., 1., 0., 1.], [1., 0., 1., 0.], [0., 0., -0.71780488, 1.]]),
            "B": np.array([[-0.73170732], [0.], [0.], [0.97560976]]),
            "Q": np.array([[0.01666667, 0, 0, 0.05], [0, 0, 0, 0], [0, 0, 0.4905, 0], [0.05, 0, 0, 0.8]]),
            "R": np.array([0.1], dtype=np.float32),
        },
        "2x2": {
            "A": np.array([[1, 1], [1, 1, ]], dtype=np.float32),
            "B": np.array([[1, 0.3], [0, 0.5]], dtype=np.float32),
            "Q": np.array([[0.5, 0.3], [0.3, 1.3]], dtype=np.float32),
            "R": np.array([[0.5, 0.25], [0.25, 1.5]], dtype=np.float32)
        },
        "3x2": {
            "A": np.array([[1., 0.3, 2.3], [0.3, 1., 1.], [0., 2., 1.]]),
            "B": np.array([[-0.5, 0.3], [0., 0.1], [0.97560976, 0.3]]),
            "Q": np.array([[0.8, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 3]]),
            "R": np.array([[1.0, 1.2], [1.2, 2.5]], dtype=np.float32)
        }
    }
    lqr = LQR(**test_cases["2x1"])
    lqr.set_weights(np.arange(lqr.weights_size))
    #lqr.set_weights(np.arange(4))
    #print(lqr.get_weight_names())
    #pi = LQR(A, B, Q, R)
    gradient_check(test_cases["4x1"], "all", n=1, tolerance=1e-3)
    print("hei")
    lqr._grad_K()

