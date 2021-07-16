import numpy as np
import scipy.linalg
import casadi


class LQR:
    def __init__(self, A, B, Q, R, time_varying, weights=None, input_shape=None, output_shape=(1,), eps=1e-8, **kwargs):
        self.E = None  # eigenvalues
        self.eps = eps
        self.time_varying = time_varying
        self._horizons = None

        A = self._numpyify(A)
        B = self._numpyify(B)
        Q = self._numpyify(Q)
        R = self._numpyify(R)

        self.A = casadi.SX.sym("A", *A.shape[-2:])
        self.A_num = A
        self.B = casadi.SX.sym("B", *B.shape[-2:])
        self.B_num = B

        if self.time_varying:
            if len(A.shape) == 4:
                self._horizons = tuple(A[i].shape[-3] for i in range(A.shape[0]))
            else:
                self._horizons = (A.shape[-3],)

        self.cQ = casadi.SX.sym("Q", *A.shape[-2:])
        try:
            self.cQ_num = np.linalg.cholesky(np.atleast_2d(Q).astype(np.float64))
        except np.linalg.LinAlgError:
            self.cQ_num = np.linalg.cholesky(np.atleast_2d(Q).astype(np.float64) + self.eps * np.eye(Q.shape[0]))  # If still fails, notify user

        self.cR = casadi.SX.sym("R", B.shape[-1], B.shape[-1])
        try:
            self.cR_num = np.linalg.cholesky(np.atleast_2d(R).astype(np.float64))
        except np.linalg.LinAlgError:
            self.cR_num = np.linalg.cholesky(np.atleast_2d(R).astype(np.float64) + self.eps * np.eye(R.shape[0]))

        self.K = casadi.SX.sym("K", B.shape[-1], A.shape[-1])
        self.K_num = None
        self.S = casadi.SX.sym("S", *A.shape[-2:])
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
        self.dx_dy_fun = casadi.Function("LQR_grad", [self.A, self.B, self.cQ, self.cR, self.S, self.K], [self.dx_dy])

        # Time-Varying Derivatives
        self.dQ = casadi.SX.sym("dQ", *self.cQ.shape)
        self.dR = casadi.SX.sym("dR", *self.cR.shape)

        self.dQ_fun = casadi.Function("dQ", [self.cQ], [casadi.jacobian(self.cQ @ self.cQ.T, self.lqr_eq_parameters)])
        self.dR_fun = casadi.Function("dR", [self.cR], [casadi.jacobian(self.cR @ self.cR.T, self.lqr_eq_parameters)])

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

    def _numpyify(self, m):
        if isinstance(m, list) or isinstance(m, np.ndarray) and m.dtype == np.object:
            if self.time_varying:
                if any([len(m[0]) != len(m[i]) for i in range(1, len(m))]):
                    horizons = [len(m[i]) for i in range(len(m))]
                    max_horizon = max(horizons)
                    res = np.full(shape=(len(m), max_horizon, *m[0][0].shape), fill_value=np.nan, dtype=np.float64)
                    for i in range(len(m)):
                        res[i, :len(m[i]), :, :] = m[i]
                    return res
                else:
                    res = np.array(m)
                    if len(res.shape) == 4 and res.shape[0] == 1:
                        res = np.squeeze(res, axis=0)
                    return res
            else:
                res = np.array(m)
                if len(res.shape) == 3 and res.shape[0] == 1:
                    res = np.squeeze(res, axis=0)
                return res
        else:
            return m

    def _set_weights_from_numeric(self, cQ, cR):
        w = []
        for num in [cQ, cR]:
            if num.size == 1:
                w.append(num.ravel())
            else:
                for c in range(num.shape[1]):
                    w.append(num[c:, c].ravel())

        self._w = np.concatenate(w)

    def set_numeric_value(self, components, indices=None):
        def get_horizons(a):
            horizons = []
            for i in range(len(a)):
                if isinstance(a[i], np.ndarray) and a[i].dtype != np.object and len(a[i].shape) == 2:
                    horizons.append(len(a))
                    break
                else:
                    horizons.extend(get_horizons(a[i]))
            return horizons
        assert isinstance(components, dict)
        assert all([k in ["A", "B", "Q", "R"] for k in components])
        new_horizons = None
        if isinstance(indices, int):
            indices = [indices]
        if self.time_varying and ("A" in components or "B" in components):
            assert "A" in components and "B" in components  # TODO: add support for one at a time
            if indices is not None:
                new_horizons = list(self._horizons)
                for i, ind in enumerate(indices):
                    if len(indices) == 1:
                        new_horizons[ind] = len(components["A"])
                    else:
                        new_horizons[ind] = len(components["A"][i])
            else:
                new_horizons = get_horizons(components["A"])
        for k, v in components.items():
            v = self._numpyify(v)
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
            num = getattr(self, "{}_num".format(k))
            if indices is None or indices == [0] and (self.time_varying and len(num.shape) == 3 or not self.time_varying and len(num.shape) == 2):
                setattr(self, "{}_num".format(k), v)
            else:
                if self.time_varying:
                    assert len(num.shape) == 4 and num.shape[0] > max(indices)
                    num = self.get_numeric_value(k)
                    if isinstance(num, np.ndarray):
                        num = list(num)
                    for i, ind in enumerate(indices):
                        if len(indices) > 1:
                            num[ind] = v[i]
                        else:
                            num[ind] = v
                    setattr(self, "{}_num".format(k), self._numpyify(num))
                else:
                    assert len(num.shape) == 3 and num.shape[0] >= indices
                    getattr(self, "{}_num".format(k))[indices] = v
        if new_horizons is not None:
            self._horizons = tuple(new_horizons)
        if "Q" or "R" in components:
            self._set_weights_from_numeric(self.cQ_num, self.cR_num)
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
        elif self.time_varying and len(self._horizons) > 1:
            if any([self._horizons[0] != h for h in self._horizons]):
                v = [v[i, :self._horizons[i]] for i in range(v.shape[0])]

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
        if self.time_varying:
            dKs = []
            dSs = []

            dQ = self.dQ_fun(self.cQ_num).toarray().T.reshape(self.weights_size, *self.cQ.shape)
            dR = self.dR_fun(self.cR_num).toarray().T.reshape(self.weights_size, *self.cR.shape)
            for i in reversed(range(self.K_num.shape[-3])):
                if i == self.K_num.shape[-3] - 1:
                    if len(self.K_num.shape) == 4:
                        for s_i in range(self.K_num.shape[0]):
                            if self._horizons[s_i] == i + 1:
                                dKS_dQR = self.dx_dy_fun(self.A_num[s_i, -1], self.B_num[s_i, -1], self.cQ_num, self.cR_num, self.S_num[s_i, -1], self.K_num[s_i, -1])
                                dKs.append(dKS_dQR[-np.product(self.K.shape):, :].toarray().T.reshape(self.weights_size, *self.K.shape))
                                dSs.append(dKS_dQR[:-np.product(self.K.shape), :].toarray().T.reshape(self.weights_size, *self.S.shape))
                            else:
                                dKs.append(np.full(shape=(self.weights_size, *self.K.shape), fill_value=np.nan, dtype=np.float64))
                                dSs.append(np.full(shape=(self.weights_size, *self.S.shape), fill_value=np.nan, dtype=np.float64))
                        dKs = [np.array(dKs).swapaxes(0, 1)]
                        dSs = [np.array(dSs).swapaxes(0, 1)]
                        dQ = np.repeat(dQ[:, np.newaxis, ...], self.A_num.shape[0], axis=1)
                        dR = np.repeat(dR[:, np.newaxis, ...], self.A_num.shape[0], axis=1)
                    else:
                        dKS_dQR = self.dx_dy_fun(self.A_num[i], self.B_num[i], self.cQ_num, self.cR_num, self.S_num[i], self.K_num[i])
                        dKs.append(dKS_dQR[-np.product(self.K.shape):, :].toarray().T.reshape(self.weights_size, *self.K.shape))
                        dSs.append(dKS_dQR[:-np.product(self.K.shape), :].toarray().T.reshape(self.weights_size, *self.S.shape))
                else:
                    if len(self.A_num.shape) == 4:
                        Ai = self.A_num[:, i, :, :]
                        Bi = self.B_num[:, i, :, :]
                        AiT = np.transpose(Ai, (0, 2, 1))
                        BiT = np.transpose(Bi, (0, 2, 1))
                        S_ip1 = self.S_num[:, i+1, :, :]
                        S_i = self.S_num[:, i, :, :]
                    else:
                        Ai = self.A_num[i]
                        Bi = self.B_num[i]
                        AiT = Ai.T
                        BiT = Bi.T
                        S_ip1 = self.S_num[i+1]
                        S_i = self.S_num[i]
                    dS = dQ + AiT @ dSs[-1] @ Ai - (AiT @ dSs[-1] @ Bi @ np.linalg.inv(self.cR_num @ self.cR_num.T + BiT @ S_ip1 @ Bi) @ BiT @ S_ip1 @ Ai + AiT @ S_ip1 @ Bi @ (-np.linalg.inv(self.cR_num @ self.cR_num.T + BiT @ S_ip1 @ Bi) @ (dR + BiT @ dSs[-1] @ Bi) @ np.linalg.inv(self.cR_num @ self.cR_num.T + BiT @ S_ip1 @ Bi) @ BiT @ S_ip1 @ Ai) + AiT @ S_ip1 @ Bi @ np.linalg.inv(self.cR_num @ self.cR_num.T + BiT @ S_ip1 @ Bi) @ BiT @ dSs[-1] @ Ai)
                    dK = -np.linalg.inv(self.cR_num @ self.cR_num.T + BiT @ S_i @ Bi) @ (dR + BiT @ dS @ Bi) @ np.linalg.inv(self.cR_num @ self.cR_num.T + BiT @ S_i @ Bi) @ BiT @ S_i @ Ai + np.linalg.inv(self.cR_num @ self.cR_num.T + BiT @ S_i @ Bi) @ BiT @ dS @ Ai
                    if i + 1 in self._horizons:
                        for s_i in range(self.K_num.shape[0]):
                            if self._horizons[s_i] == i + 1:
                                dKS_dQR = self.dx_dy_fun(self.A_num[s_i, i], self.B_num[s_i, i], self.cQ_num, self.cR_num, self.S_num[s_i, i], self.K_num[s_i, i])
                                dK[:, s_i] = dKS_dQR[-np.product(self.K.shape):, :].toarray().T.reshape(self.weights_size, *self.K.shape)
                                dS[:, s_i] = dKS_dQR[:-np.product(self.K.shape), :].toarray().T.reshape(self.weights_size, *self.S.shape)

                    dSs.append(dS)
                    dKs.append(dK)
            if len(self.K_num.shape) == 4:
                dKs = np.flip(np.transpose(dKs, (2, 0, 3, 4, 1)), axis=1)
                if any([self._horizons[0] != h for h in self._horizons]):
                    dKs = [dKs[i, :self._horizons[i]] for i in range(dKs.shape[0])]
                return dKs
            else:
                return np.flip(np.transpose(dKs, (0, 2, 3, 1)), axis=1)
        else:
            dKS_dQR = self.dx_dy_fun(self.A_num, self.B_num, self.cQ_num, np.expand_dims(self.cR_num, axis=-1), self.S_num, self.K_num)  # TODO: is it correct with expand dims here??
            dK_dQR = dKS_dQR[-np.product(self.K.shape):, :]
            return dK_dQR.toarray().reshape(*self.K.shape, self.weights_size)

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

    def get_action(self, x, t=None):
        if t is not None:
            assert self.time_varying
            K = self.get_numeric_value("K")
            if isinstance(K, list):
                t = np.minimum(t, [len(K_i) - 1 for K_i in K]).astype(np.int32)
            else:
                t = np.minimum(t, K.shape[1] - 1).astype(np.int32)
            K = np.array([K_i[t[i]] for i, K_i in enumerate(K)]).reshape(x.shape[0], *self.K.shape)
            return np.squeeze(-K @ np.atleast_3d(x), -1)
        else:
            assert not self.time_varying
            return -self.K_num @ x

    def diff(self, x):
        return self.get_policy_gradient(x)

    def _compute_lqr(self):  # TODO: can have indices argument to only recalculate certain systems
        """Solve the discrete time lqr controller.

        dx/dt = A x + B u

        cost = integral x.T*Q*x + u.T*R*u
        """

        # ref Bertsekas, p.151
        try:
            Q_num = self.cQ_num @ self.cQ_num.T
            R_num = self.cR_num @ self.cR_num.T

            if self.time_varying:
                S_num, K_num = [], []
                for i in reversed(range(self.A_num.shape[-3])):
                    if len(self.A_num.shape) == 4:
                        Ai = self.A_num[:, i, :, :]
                        Bi = self.B_num[:, i, :, :]
                        AiT = np.transpose(Ai, (0, 2, 1))
                        BiT = np.transpose(Bi, (0, 2, 1))
                    else:
                        Ai = self.A_num[i]
                        Bi = self.B_num[i]
                        AiT = Ai.T
                        BiT = Bi.T
                    if i == self.A_num.shape[-3] - 1:
                        if len(self.A_num.shape) == 4:
                            S_num.append([])
                            for s_i in range(self.A_num.shape[0]):
                                if self._horizons[s_i] == i + 1:
                                    S_num[-1].append(scipy.linalg.solve_discrete_are(Ai[s_i], Bi[s_i], Q_num, R_num))
                                else:
                                    S_num[-1].append(np.full(shape=self.S.shape, fill_value=np.nan, dtype=np.float64))
                        else:
                            S_num.append(scipy.linalg.solve_discrete_are(Ai, Bi, Q_num, R_num))
                    else:
                        S_num.append(Q_num + AiT @ S_num[-1] @ Ai - AiT @ S_num[-1] @ Bi @ np.linalg.inv(R_num + BiT @ S_num[-1] @ Bi) @ BiT @ S_num[-1] @ Ai)
                        if i + 1 in self._horizons:
                            for s_i in range(self.A_num.shape[0]):
                                if self._horizons[s_i] == i + 1:
                                    S_num[-1][s_i] = scipy.linalg.solve_discrete_are(Ai[s_i], Bi[s_i], Q_num, R_num)
                    K_num.append(self._calculate_gain_matrix(Ai, Bi, R_num, S_num[-1]))

                self.S_num = np.array(list(reversed(S_num)))
                self.K_num = np.array(list(reversed(K_num)))
                if len(self.S_num.shape) == 4:
                    self.S_num = self.S_num.swapaxes(0, 1)
                    self.K_num = self.K_num.swapaxes(0, 1)

            else:
                # first, try to solve the ricatti equation
                S = np.matrix(scipy.linalg.solve_discrete_are(self.A_num, self.B_num, Q_num, R_num))
                self.S_num = S.A

                # compute the LQR gain
                #self.K = np.matrix(np.linalg.inv(np.atleast_2d(self.R)) * (self.B.T * self.S))
                self.K_num = np.matrix(np.linalg.inv(np.atleast_2d(self.B_num.T @ S @ self.B_num + R_num)) @ (self.B_num.T @ S @ self.A_num)).A

                self.E, eigVecs = scipy.linalg.eig(self.A_num - self.B_num @ self.K_num)
        except np.linalg.LinAlgError as e:
            print("LinalgError for system:")
            for comp_name in ["A", "B", "cQ", "cR"]:
                print(comp_name)
                print(getattr(self, "{}_num".format(comp_name)))
            raise e

    def _calculate_gain_matrix(self, A, B, R, S):
        if len(B.shape) == 3:
            BT = np.transpose(B, (0, 2, 1))
        else:
            BT = B.T
        return np.linalg.inv(np.atleast_2d(BT @ S @ B + R)) @ BT @ S @ A

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

    def set_weights(self, w, compute_lqr=True):
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

        if compute_lqr:
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


def gradient_check_tv(system, print_mode, n=5, h=5, eps=1e-4, tolerance=1e-6, seed=None):
    import copy
    def perturb_weight(_A, _B, _Q, _R, idx, _eps=1e-4):
        def print_results(K_p, K_n, _grad, name, t):
            num_grad = (K_p - K_n) / (2 * _eps)
            _grad = _grad.reshape(*num_grad.shape, order="F")
            is_close = np.isclose(_grad, num_grad, atol=tolerance)
            if print_mode == "all" or (print_mode == "failure" and not np.all(is_close)):
                print("{} (t = {})".format(name, t))
                print("np.isclose: \n{}".format(is_close))
                print("(K_p - K_n) / 2eps = \n{}".format(num_grad))
                print("Grad = \n{}".format(_grad))
                print("Diff: \n{}".format(num_grad - _grad))
                print("-" * 30)

        pi = LQR(_A, _B, _Q, _R, time_varying=True)
        pi.set_numeric_value({"A": np.array(_A), "B": _B})
        grad = pi._grad_K()
        ws = np.copy(pi.get_weights())
        ws[idx] += eps
        pi.set_weights(ws)
        Kp = np.copy(pi.get_numeric_value("K"))
        ws[idx] -= 2*eps
        pi.set_weights(ws)
        Kn = np.copy(pi.get_numeric_value("K"))
        w_names = pi.get_weight_names()
        for t in reversed(range(h)):
            if n > 1:
                if isinstance(grad, list):
                    for s_i in range(len(Kp)):
                        if t < grad[s_i].shape[0]:
                            print_results(Kp[s_i][t], Kn[s_i][t], grad[s_i][t, :, :, idx], w_names[idx], t)
                else:
                    print_results(Kp[:, t, ...], Kn[:, t, ...], grad[:, t, :, :, idx], w_names[idx], t)
            else:
                print_results(Kp[t], Kn[t], grad[t, :, :, idx], w_names[idx], t)

    assert print_mode in ["all", "failure", "none"]
    pi = None

    if seed is not None:
        np.random.seed(seed)

    if not isinstance(system["A"], list):
        system["A"] = [system["A"] + np.random.uniform(-0.5, 0.5, size=system["A"].shape) for _ in range(h)]
        system["B"] = [system["B"] + np.random.uniform(-0.5, 0.5, size=system["B"].shape) for _ in range(h)]

    systems = []
    for n_i in range(n):
        success = False
        while not success:
            try:
                _system = copy.deepcopy(system)
                rand_comp = np.random.choice(["A", "B"])#, "Q", "R"])
                _system[rand_comp] = np.random.uniform(-3, 3, size=_system[rand_comp].shape if not isinstance(_system[rand_comp], list) else _system[rand_comp][0].shape)
                if rand_comp in ["Q", "R"]:
                    _system[rand_comp] = (_system[rand_comp] + _system[rand_comp].T) / 2
                    if np.any(np.linalg.eigvals(np.atleast_2d(_system[rand_comp])) < 0):
                        raise np.linalg.LinAlgError
                elif rand_comp in ["A", "B"]:
                    _system[rand_comp] = [_system[rand_comp] + np.random.uniform(-0.5, 0.5,  size=_system[rand_comp][0].shape) for _ in range(h)]
                if pi is None:
                    pi = LQR(**_system, time_varying=True)
                pi.set_numeric_value(_system)
                systems.append(_system)
                success = True
            except np.linalg.LinAlgError as e:
                pass

    for w_i in range(pi.weights_size):
        perturb_weight([s["A"] for s in systems],
                       [s["B"] for s in systems],
                       system["Q"],
                       system["R"], w_i, _eps=eps)


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
        pi = LQR(_A, _B, _Q, _R, True)
        grad = pi._grad_K()
        ws = np.copy(pi.get_weights())
        ws[idx] += eps
        pi.set_weights(ws)
        Kp = np.copy(pi.K_num)
        ws[idx] -= 2*eps
        pi.set_weights(ws)
        Kn = np.copy(pi.K_num)
        w_names = pi.get_weight_names()
        print_results(Kp, Kn, grad[:, :, idx], w_names[idx])

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
    invp = {"A": np.array([[-0., 0., 36.33333333, 0.],
                            [ 0., 0., 0., 1.],
                            [ 1., 0., 0., 0.        ],
                            [ 0., 0., -2.27083333, 0.]]),
            "B": np.array([[-4.62962963], [ 0.], [ 0.], [ 1.53935185]]),
            "Q": np.eye(4),
            "R": np.array([[1]])}
    #lqr = LQR(**test_cases["2x1"])
    #lqr.set_weights(np.arange(lqr.weights_size))
    #lqr.set_weights(np.arange(4))
    #print(lqr.get_weight_names())
    #pi = LQR(A, B, Q, R)
    import time
    t_b = time.process_time()
    #gradient_check(invp, "all", n=2, seed=1)
    gradient_check_tv(test_cases["4x1"], "failure", n=2, h=5, tolerance=1e-3, seed=0)
    #print("elapsed_time {}".format(time.process_time() - t_b))
    #print("hei")
    #lqr._grad_K()

