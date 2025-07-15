import numpy as np
import gpflow
from scipy.optimize import minimize

class Generalised_PoE:

    def __init__(self):
        pass

    def neg_log_likelihood(self, hyperparams=None):
        """
        Evaluate the negative log likelihood (NLL) for each GP model in self.models.
        
        Parameters:
        - hyperparams: Dictionary of hyperparameters to set for each model's kernel
                       (e.g., {'lengthscales': 1.0, 'variance': 1.0}).
        
        Returns:
        - nll_values: List of NLL values for each expert.
        """

        nll = 0
        for i, model in enumerate(self.models):

            for param, value in hyperparams.items():
                setattr(model.kernel, param, value)
            setattr(model.likelihood, 'variance', hyperparams['likelihood_variance'])

            # Compute the NLL for the model
            nll += model.training_loss().numpy()

        return nll

    def train(self, X, Y, no_experts=1, allow_seperate_hyperparms=False):
        self.no_experts = no_experts

        # Dimension of input
        self.D = X.shape[1]

        # We use the squared exponential kernel with 'height' equal to 1 by default
        self.kernel = gpflow.kernels.SquaredExponential(lengthscales=np.repeat(10, self.D))
        self.kernel.variance.assign(1.0)
        gpflow.set_trainable(self.kernel.variance, False)

        # Divide the data into no_experts parts
        X_parts = np.array_split(X, no_experts)
        Y_parts = np.array_split(Y, no_experts)

        self.models = []
        for i in range(no_experts):
            self.models.append(gpflow.models.GPR(data=(X_parts[i], Y_parts[i]), kernel=self.kernel))

        if allow_seperate_hyperparms:
            # Optimise hyperparameters for each expert
            for i in range(no_experts):
                opt = gpflow.optimizers.Scipy()
                opt.minimize(self.models[i].training_loss, self.models[i].trainable_variables)
        else:

            def objective(hyperparams):
                # Convert hyperparams array to dictionary
                hyperparams_dict = {
                    'lengthscales': hyperparams[0:self.D],
                    'likelihood_variance': hyperparams[-1]
                }
                return self.neg_log_likelihood(hyperparams=hyperparams_dict)

            # Initial guess for hyperparameters
            initial_hyperparams = [10] * self.D + [1]  # lengthscales and likelihood_variance

            # Bounds for hyperparameters (optional)
            bounds = [(1e-3, 10.0)] * self.D +  [(1e-6, 1.0)]  # Bounds for lengthscales and likelihood variance

            # Minimise the objective function
            result = minimize(objective, initial_hyperparams, bounds=bounds, method='L-BFGS-B')

            # Print and store optimised hyperparameters
            print("Optimised hyperparameters:", result.x)
            self.joint_hyperparameters = result.x

    def predict(self, X_new):

        N_new = X_new.shape[0]

        # Run all experts, collection their predictive means and
        # standard deviation
        mu_all = np.zeros([N_new, self.no_experts])
        sigma_all = np.zeros([N_new, self.no_experts])
        for i in range(self.no_experts):
            mu, var = self.models[i].predict_y(X_new)
            mu_all[:, i] = mu.numpy()[:, 0]
            sigma_all[:, i] = np.sqrt(var.numpy()[:, 0])

        # Calculate the normalised predictive power of the predictions made
        # by each GP. Note that here we are assuming that k(x_star, x_star)=1
        # to simplify the calculation. Note that we also add a small 'jitter'
        # term to the prior, to ensure that beta never goes below zero.
        beta = np.zeros([N_new, self.no_experts])
        for i in range(self.no_experts):
            noise_std = np.sqrt(self.models[i].likelihood.variance)
            beta[:, i] = (0.5 * np.log(1 + 1e-9 + noise_std**2) - np.log(sigma_all[:, i]))

        # Normalise beta
        for i in range(N_new):
            beta[i, :] = beta[i, :] / np.sum(beta[i, :])

        # Find generalised PoE GP predictive precision
        prec_star = np.zeros(N_new)
        for i in range(self.no_experts):
            prec_star += beta[:, i] * sigma_all[:, i]**-2

        # Find generalised PoE GP predictive variance and standard
        # deviation
        var_star = prec_star**-1

        # Find generalised PoE GP predictive mean
        y_star_mean = np.zeros(N_new)
        for i in range(self.no_experts):
            y_star_mean += beta[:, i] * sigma_all[:, i]**-2 * mu_all[:, i]
        y_star_mean *= var_star

        return np.vstack(y_star_mean), np.vstack(var_star), np.vstack(beta)
