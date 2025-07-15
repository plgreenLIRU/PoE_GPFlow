from PoE_GPFlow import Generalised_PoE
import numpy as np

def test_1D():

    # Initialise
    m = Generalised_PoE()

    # Generate a simple dataset
    np.random.seed(42)
    X = np.linspace(0, 10, 500).reshape(-1, 1)
    noise_std = 0.05
    Y = np.sin(X) + noise_std * np.random.randn(500, 1)  # Noisy sine wave

    # Train model, all hyperparameters equal
    m.train(X, Y, no_experts=3, allow_seperate_hyperparms=False)

    # Check hyperparameters are equal
    for i in range(3):
        assert(np.allclose(m.models[i].kernel.lengthscales, m.joint_hyperparameters[0]))
        assert(np.allclose(m.models[i].likelihood.variance, m.joint_hyperparameters[-1]))

    # Check noise variance
    assert(np.allclose(m.joint_hyperparameters[-1], noise_std**2, atol=0.01))

    # Predictions
    X_new = np.linspace(0, 10, 100).reshape(-1, 1)
    mean, var, beta = m.predict(X_new)
    assert(np.allclose(mean, np.sin(X_new), atol=0.05))

def test_2D():

    # Define the true function
    def f(x, y):
        return np.sin(x) * np.cos(y)

    # Generate training data
    np.random.seed(2)
    n_train = 200
    x_train = np.random.uniform(0, 5, size=(n_train,))
    y_train = np.random.uniform(0, 5, size=(n_train,))
    X_train = np.vstack((x_train, y_train)).T
    sigma = 0.1
    z_train = f(x_train, y_train) + sigma * np.random.randn(n_train)

    # Generate test grid
    x_lin = np.linspace(0, 5, 50)
    y_lin = np.linspace(0, 5, 50)
    X1, X2 = np.meshgrid(x_lin, y_lin)
    X_test = np.vstack([X1.ravel(), X2.ravel()]).T

    Z_true = f(X_test[:, 0], X_test[:, 1])

    m = Generalised_PoE()
    m.train(X=X_train, Y=np.vstack(z_train), no_experts=3, allow_seperate_hyperparms=False)

    for i in range(3):
        assert(np.allclose(m.models[i].kernel.lengthscales, m.joint_hyperparameters[0:2]))
        assert(np.allclose(m.models[i].likelihood.variance, m.joint_hyperparameters[-1]))

    # Check inferred noise variance
    assert(np.allclose(m.joint_hyperparameters[-1], sigma**2, atol=0.01))

    # Check mean predictions
    mean, var, beta = m.predict(X_test)
    assert(np.allclose(mean[:, 0], Z_true, atol=0.2))
