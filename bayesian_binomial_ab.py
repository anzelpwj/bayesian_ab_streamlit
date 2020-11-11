import streamlit as st
import numpy as np
from scipy.stats import beta as betadist
import matplotlib.pyplot as plt
import ast


# Base option
def base_option(option_name):
    st.subheader("Please select a calculator from the left sidebar")


# Prior calculator
def compute_alpha_beta(mu, sigma):
    """Method of moments calculation of Beta distribution parameters"""
    try:
        mu = float(mu)
        sigma = float(sigma)
    except Exception:
        raise ValueError("Need to give valid floating point numbers for μ and σ")
    if not 0 < mu < 1:
        raise ValueError("Need to give μ in a the range (0, 1)")
    if not 0 < sigma < 1:
        raise ValueError("Need to give σ in a the range (0, 1)")
    alpha = -mu + ((1 - mu)*mu**2)/(sigma**2)
    beta = -alpha + (alpha/mu)
    return alpha, beta


def alpha_beta_prior_calculator(option_name):
    st.subheader(option_name)

    mu = st.text_input("μ (fraction between 0 and 1)", value=".2")
    sigma = st.text_input("σ (fraction between 0 and 1)", value=".03")
    alpha, beta = compute_alpha_beta(mu, sigma)
    st.write(f"α = {alpha:.2f}, β = {beta:.2f}")

    x = np.linspace(0, 1, 1000)
    y = betadist.pdf(x, alpha, beta)
    fig, ax = plt.subplots()
    ax.plot(x, y, color='k', linewidth=2)
    ax.set_title("Prior distribution", fontsize=20)
    st.pyplot(fig)


# Prior and posterior


# A > B chance
def chance_option_A_true(s_a, n_a, s_b, n_b, alpha, beta, x_steps=5000):
    """
    Inputs:
    - s_a: Number of succeses for option A
    - n_a: Number of total samples for option A
    - s_b: Number of successes for option B
    - n_b: Number of total samples for option B
    - alpha: Alpha
    - beta: Beta
    - x_steps: Number of steps for numerical integral
    """
    s_a = float(s_a)
    s_b = float(s_b)
    n_a = float(n_a)
    n_b = float(n_b)
    a = float(alpha)
    b = float(beta)

    f_a = n_a - s_a
    f_b = n_b - s_b
    x = np.linspace(0, 1, num=x_steps, endpoint=False)
    dx = x[1] - x[0]
    beta_a = betadist.pdf(x, a + s_a, b + f_a)
    beta_b = betadist.pdf(x, a + s_b, b + f_b)

    array = np.outer(beta_a, beta_b)
    array = np.tril(array)  # Mask out upper-triangular bit
    total = array.sum().sum()*dx**2

    return total


def ab_chance_of_being_correct(option_name):
    st.subheader(option_name)

    s_a = st.text_input("Number of successes with option A", value="45")
    n_a = st.text_input("Number of samples with option A", value="200")
    s_b = st.text_input("Number of successes with option B", value="40")
    n_b = st.text_input("Number of samples with option B", value="200")
    alpha = st.text_input("α", value="35")
    beta = st.text_input("β", value="141")
    total = chance_option_A_true(s_a, n_a, s_b, n_b, alpha, beta)
    if total*100 >= 0.01:
        st.write(f"Chance A is better than B: {100*total:.2f}%")
    else:
        st.write(f"Chance A is better than B: {100*total:.2e}%")


# A > B loss
def loss_option_A(s_a, n_a, s_b, n_b, alpha, beta, x_steps=5000):
    """
    Inputs:
    - s_a: Number of succeses for option A
    - n_a: Number of total samples for option A
    - s_b: Number of successes for option B
    - n_b: Number of total samples for option B
    - alpha: Alpha
    - beta: Beta
    - x_steps: Number of steps for numerical integral
    """
    s_a = float(s_a)
    s_b = float(s_b)
    n_a = float(n_a)
    n_b = float(n_b)
    a = float(alpha)
    b = float(beta)

    f_a = n_a - s_a
    f_b = n_b - s_b
    x = np.linspace(0, 1, num=x_steps, endpoint=False)
    dx = x[1] - x[0]
    beta_a = betadist.pdf(x, a + s_a, b + f_a)
    beta_b = betadist.pdf(x, a + s_b, b + f_b)

    joint_posterior = np.outer(beta_a, beta_b)
    loss_error = np.fromfunction(lambda i, j: (j - i)*dx, (x_steps, x_steps))

    loss_error = np.triu(loss_error)  # Mask out lower-triangular bit
    final = joint_posterior * loss_error

    total = final.sum().sum()*dx**2
    return total


def ab_loss(option_name):
    st.subheader(option_name)

    s_a = st.text_input("Number of successes with option A", value="45")
    n_a = st.text_input("Number of samples with option A", value="200")
    s_b = st.text_input("Number of successes with option B", value="40")
    n_b = st.text_input("Number of samples with option B", value="200")
    alpha = st.text_input("α", value="35")
    beta = st.text_input("β", value="141")
    total = loss_option_A(s_a, n_a, s_b, n_b, alpha, beta)
    if total*100 >= 0.01:
        st.write(f"Expected loss with option A: {100*total:.2f}%")
    else:
        st.write(f"Expected loss with option A: {100*total:.2e}%")


# A > B, C, ... Monte Carlo
def abc_mc_chance_of_being_correct(successes, sample_sizes, a, b, N=5000000, seed=None):
    """
    Monte Carlo simulation to determine chance option A (out of many options) is
    the correct choice.

    Inputs:
    - successes: Array of successes for samples A, B, C, ...
    - sample_sizes: Array of sample sizes for samples A, B, C, ...
    - a: Prior alpha
    - b: Prior beta
    - N: Number of samples to pick
    - seed: Random seed
    """

    if len(successes) != len(sample_sizes):
        raise ValueError("successes and sample_sizes need to be the same length")

    if seed:
        np.random.seed(seed)

    successes = np.array(successes)
    failures = np.array(sample_sizes) - successes

    dim = len(successes)

    random_vals = np.random.rand(dim, N)

    # Joint probability
    beta_vals = np.zeros((dim, N))
    for ii in range(dim):
        beta_vals[ii, :] = beta.pdf(random_vals[ii, :], a + successes[ii], b + failures[ii])
    prod_beta = np.prod(beta_vals, axis=0)

    # Find where lambda_A >= all the others (indicator function)
    indicator = np.ones(N, dtype=bool)
    for ii in range(1, dim):
        indicator = indicator * (random_vals[0, :] >= random_vals[ii, :])

    value = (prod_beta * indicator).sum()/N
    return value


def abc_mc_correct(option_name):
    st.subheader(option_name)

    successes = st.text_input("Number of successes with option A, B, ...", value="[45, 40]")
    sample_sizes = st.text_input("Number of samples with option A, B, ...", value="[200, 200]")
    successes = ast.literal_eval(successes)
    sample_sizes = ast.literal_eval(sample_sizes)
    alpha = st.text_input("α", value="35")
    beta = st.text_input("β", value="141")
    mc_samples = st.text_input("Number of Monte Carlo samples (generally 1 million to 100 million)", value="5E6")
    seed = st.text_input("Random seed", value="0")
    mc_samples = int(mc_samples)
    alpha = float(alpha)
    beta = float(beta)
    seed = int(seed)
    ls = len(successes)
    ln = len(sample_sizes)

    if ls != ln:
        st.write(f"Need the same number of success values and total values. Imbalance {ls} != {ln}")
    else:
        total = abc_mc_chance_of_being_correct(successes, sample_sizes, alpha, beta, N=mc_samples, seed=seed)
        if total*100 >= 0.01:
            st.write(f"Chance option A is correct: {100*total:.2f}%")
        else:
            st.write(f"Chance option A is correct: {100*total:.2e}%")


# Loss with A versus B, C, ... Monte Carlo
def abc_mc_expected_loss(successes, sample_sizes, a, b, N=1000000, seed=None):
    """
    Monte Carlo simulation to determine expected loss if you choose option A.

    Inputs:
    - successes: Array of successes for samples A, B, C, ...
    - sample_sizes: Array of sample sizes for samples A, B, C, ...
    - a: Prior alpha
    - b: Prior beta
    - N: Number of samples to pick
    - seed: Random seed
    """

    if len(successes) != len(sample_sizes):
        raise ValueError("successes and sample_sizes need to be the same length")

    if seed:
        np.random.seed(seed)

    successes = np.array(successes)
    failures = np.array(sample_sizes) - successes
    dim = len(successes)
    random_vals = np.random.rand(dim, N)

    # Joint probability
    beta_vals = np.zeros((dim, N))
    for ii in range(dim):
        beta_vals[ii, :] = beta.pdf(random_vals[ii, :], a + successes[ii], b + failures[ii])
    prod_beta = np.prod(beta_vals, axis=0)

    # Find max lambda_notA > lambda_A (loss)
    minus_lambda_A = random_vals - random_vals[0, :]
    loss = np.max(minus_lambda_A, axis=0)
    value = (prod_beta * loss).sum()/N

    return value


# TODO
def need_to_implement(option_name):
    st.subheader(f"{option_name} has not been implemented yet")


BASE_OPTION = "Please select an option..."
ALPHA_BETA_PRIOR = "α-β prior calculator"
AB_CHANCE_CORRECT = "A/B chance of being correct"
AB_LOSS = "A/B predicted loss"
ABC_MC_CHANCE_CORRECT = "A/B/C... chance of being correct"
ABC_MC_LOSS = "A/B/C... predicted loss"


SELECT_OPTIONS = {
    BASE_OPTION: base_option,
    ALPHA_BETA_PRIOR: alpha_beta_prior_calculator,
    AB_CHANCE_CORRECT: ab_chance_of_being_correct,
    AB_LOSS: ab_loss,
    ABC_MC_CHANCE_CORRECT: abc_mc_correct,
    ABC_MC_LOSS: need_to_implement,
}


def main():
    st.title("Bayesian Binomial A/B Calculator")

    select_options = list(SELECT_OPTIONS.keys())
    function = st.sidebar.selectbox("Select desired calculator", select_options)

    SELECT_OPTIONS[function](function)


if __name__ == "__main__":
    main()
