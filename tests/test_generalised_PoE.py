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
