# Caelan Osman
# March 7, 2022

import sys
from scipy.stats.distributions import norm
from scipy.stats import multivariate_normal
from scipy.optimize import fmin
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pydataset import data as pydata
from statsmodels.tsa.api import VARMAX
import statsmodels.api as sm
from statsmodels.tsa.base.datetools import dates_from_str
from statsmodels.tsa.stattools import arma_order_select_ic as order_select
import pandas as pd

def arma_forecast_naive(file='weather.npy', p=2, q=1, n=20):
    """
    Perform ARMA(1,1) on data. Let error terms be drawn from
    a standard normal and let all constants be 1.
    Predict n values and plot original data with predictions.

    Parameters:
        file (str): data file
        p (int): order of autoregressive model
        q (int): order of moving average model
        n (int): number of future predictions
    """
    # load file
    Z = np.load(file)
    # get differencing
    Y = np.diff(Z, 1)

    # set values of Phi and Theta
    Phi = 1/2.
    Theta = 1/10.

    # get predictions

    Predictions = np.empty(n)
    Predictions = np.concatenate((Y, Predictions))
    '''
    Errors = np.random.normal(loc=0, scale=1, size=Predictions.size)
    '''
    Errors = np.random.normal(loc=0, scale=1, size=q)
    for i in range(Y.size, Predictions.size):
        Curr_Error = np.random.normal(loc=0, scale=1)
        Predictions[i] = Phi*np.sum(Predictions[i-p:i]) + Curr_Error + Theta*np.sum(Errors)
        Errors[0] = Curr_Error

    Predictions=Predictions[Y.size:]

    # linspace for plotting
    T1 = np.linspace(13+19/24, 16+18/24, Y.size)
    T2 = np.linspace(16+19/24, 17+15/24, n)

    # plot
    fig = plt.figure()
    fig.set_dpi(150)
    ax = fig.add_subplot(111)
    ax.set_xticks([14, 15, 16, 17, 18])
    ax.plot(T1, Y, label='Old Data')
    ax.plot(T2, Predictions, label='New Data')
    ax.set_xlabel('Day of Month')
    ax.set_ylabel(r'Change in Temperature $(C) - \mu = 0$')
    ax.set_title(r'$\operatorname{ARMA}(2, 1)$ Naive Forecast')
    ax.legend(loc='best')
    plt.show()

    return

def arma_likelihood(file='weather.npy', phis=np.array([0.9]), thetas=np.array([0]), mu=17., std=0.4):
    """
    Transfer the ARMA model into state space.
    Return the log-likelihood of the ARMA model.

    Parameters:
        file (str): data file
        phis (ndarray): coefficients of autoregressive model
        thetas (ndarray): coefficients of moving average model
        mu (float): mean of error
        std (float): standard deviation of error

    Return:
        log_likelihood (float)
    """
    # load data
    Z = np.load(file)
    # take first difference
    Y = np.diff(Z)

    F, Q, H, dim_states, dim_time_series = state_space_rep(phis, thetas, mu, std)
    mus, covs = kalman(F, Q, H, Y-mu)

    # calculate log likelihood
    log_like = 0
    if dim_time_series == 1:
        log_like = np.sum(np.log([norm.pdf(y, loc=float(H@mu_i+mu), scale=np.sqrt(float(H@cov_i@(H.T)))) for y, mu_i, cov_i in zip(Y, mus, covs)]))


    elif dim_time_series > 1:
        log_like = np.sum(np.log([multivariate_normal.pdf(y, mean=H@mu_i + mu, cov=H@cov_i@H.T) for y, mu_i, cov_i in zip(Y, mus, covs)]))

    else:
        raise ValueError("Incorrect dimensionality")

    return log_like

def model_identification(file='weather.npy', p=4, q=4):
    """
    Identify parameters to minimize AIC of ARMA(p,q) model

    Parameters:
        file (str): data file
        p (int): maximum order of autoregressive model
        q (int): maximum order of moving average model

    Returns:
        phis (ndarray (p,)): coefficients for AR(p)
        thetas (ndarray (q,)): coefficients for MA(q)
        mu (float): mean of error
        std (float): std of error
    """

    # load
    Z = np.load(file)
    # first difference
    Y = np.diff(Z, 1)
    # initialize variables
    n = len(Y)
    Y_mean = Y.mean()
    Y_std = Y.std()
    Best_AIC = np.inf
    # iterate through and find the best model
    for i in range(1, p+1):
        for j in range(1, q+1):

            # negative log likelihood function
            def f(x):
                return -1*arma_likelihood(file=file, phis=x[:i], thetas=x[i:i+j], mu=x[-2], std=x[-1])

            # initial guess for minimize function
            x0 = np.zeros(i+j+2)
            x0[-2] = Y_mean
            x0[-1] = Y_std

            # minimize negative log likelihood
            sol = minimize(f, x0, method="SLSQP")
            xopt = sol["x"]
            fopt = sol["fun"]

            # calculate the AIC
            k = i+j+2
            AIC = 2*k*(1 + (k+1) / (n-k)) - 2*fopt

            # check for best model
            if AIC < Best_AIC:
                Best_AIC = AIC
                phis = xopt[:i]
                thetas = xopt[i:i+j]
                mu = xopt[-2]
                std = xopt[-1]


    return phis, thetas, mu, std

def arma_forecast(file='weather.npy', phis=np.array([0]), thetas=np.array([0]), mu=0., std=0., n=30):
    """
    Forecast future observations of data.

    Parameters:
        file (str): data file
        phis (ndarray (p,)): coefficients of AR(p)
        thetas (ndarray (q,)): coefficients of MA(q)
        mu (float): mean of ARMA model
        std (float): standard deviation of ARMA model
        n (int): number of forecast observations

    Returns:
        new_mus (ndarray (n,)): future means
        new_covs (ndarray (n,)): future standard deviations
    """

    # load and get first difference
    Z = np.load(file)
    Y = np.diff(Z, 1)

    # get time series values
    F, Q, H, dim_states, dim_time_series = state_space_rep(phis, thetas, mu, std)

    # mus and covs
    mus, covs = kalman(F, Q, H, Y-mu)

    # run update step
    x_predict = mus[-1]
    P_predict = covs[-1]

    zk = np.array([Y[-1]])

    y_tilde = zk - H @ x_predict

    S = H@P_predict @ H.T

    K = P_predict @ H.T @ np.linalg.inv(S)
    est = x_predict + K @ y_tilde

    P = (np.eye(K.shape[0]) - K@H)@P_predict

    # now run predict step n times
    new_mus = np.empty(n)
    new_covs = np.empty(n)
    for i in range(n):
        P = F@P@F.T + Q
        est = F@est
        new_covs[i] = np.squeeze(H@P@H.T)
        new_mus[i] = np.squeeze(H@est + mu)

    # time linspaces
    T1 = np.linspace(13+19/24, 16+18/24, Y.size)
    T2 = np.linspace(16+19/24, 16+19/24 + 30/24, n)

    # plot
    fig = plt.figure()
    fig.set_dpi(150)
    ax = fig.add_subplot(111)
    ax.set_xticks([14, 15, 16, 17, 18])
    ax.plot(T1, Y, label='Old Data')
    ax.plot(T2, new_mus, '--', label = 'forecast')
    ax.plot(T2, new_mus + 2*np.sqrt(new_covs), 'g-', label='95% confidence interval')
    ax.plot(T2, new_mus -2*np.sqrt(new_covs))
    ax.set_xlabel('Day of the Month')
    ax.set_ylabel(r'Change in Temperature $(C) - \mu = 0$')
    ax.set_title(r'$\operatorname{ARMA}(1, 1)$')
    ax.legend(loc='best')
    plt.show()

    return new_mus, new_covs

def sm_arma(file = 'weather.npy', p=3, q=3, n=30):
    """
    Build an ARMA model with statsmodel and
    predict future n values.

    Parameters:
        file (str): data file
        p (int): maximum order of autoregressive model
        q (int): maximum order of moving average model
        n (int): number of values to predict

    Return:
        aic (float): aic of optimal model
    """

    # load and get diff
    Z = np.load(file)
    Y = np.diff(Z, 1)

    # initialize model
    best_model = None
    best_AIC = np.inf
    for i in range(1, p+1):
        for j in range(1, q+1):
            model = ARIMA(Y, order=(i, 0, j), trend='c').fit(method="innovations_mle")
            AIC = model.aic

            if AIC < best_AIC:
                best_AIC = AIC
                best_model = model

    # get model prediction
    prediction = best_model.predict(start=0, end=Y.size+n)

    # plot
    T1 = np.linspace(13+19/24, 16+18/24, Y.size)
    T2 = np.linspace(13+19/24, 16+18/24 + n/24, prediction.size)
    fig = plt.figure()
    fig.set_dpi(150)
    ax = fig.add_subplot(111)
    ax.plot(T1, Y, label='Old Data')
    ax.plot(T2, prediction, label='ARMA Model')
    ax.set_xlabel('Day of the Month')
    ax.set_ylabel(r'Change in Temperature $(C) - \mu = 0$')
    ax.set_title(r'Statsmodel $\operatorname{ARMA}(1, 1)$')
    ax.legend(loc='best')
    ax.set_xticks([14, 15, 16, 17, 18])
    plt.show()

    return best_AIC

def sm_varma(start ='1959-09-30', end = '2012-09-30'):
    """
    Build an ARMA model with statsmodel and
    predict future n values.
​a
    Parameters:
        start (str): the data at which to begin forecasting
        end (str): the date at which to stop forecasting
​
    Return:
        aic (float): aic of optimal model
    """
    # Load in data
    df = sm.datasets.macrodata.load_pandas().data
    # Create DateTimeIndex
    dates = df[['year', 'quarter']].astype(int).astype(str)
    dates = dates["year"] + "Q" + dates["quarter"]

    dates = dates_from_str(dates)
    df.index = pd.DatetimeIndex(dates)
    # Select columns used in prediction
    df = df[['realgdp','realcons','realinv']]

    # initialize and fit model
    mod = VARMAX(df)
    mod = mod.fit(maxiter=1000, disp=False, ic='aic')
    # Predict from start to end
    pred = mod.predict(start, end)
    # Get confidence intervals
    forecast_obj = mod.get_forecast(end)
    all_CI = forecast_obj.conf_int(alpha=0.05)

    fig, axs = plt.subplots(3, 1, constrained_layout=True, figsize=(10, 10))
    fig.set_dpi(150)

    ax = axs[0]

    ax.plot(df['realgdp'], label='realgdp')
    ax.plot(all_CI['lower realgdp'], 'k--', label='95% confidence interval')
    ax.plot(all_CI['upper realgdp'], 'k--')
    ax.plot(pred['realgdp'], label='forecast')
    ax.set_xlabel('year')
    ax.set_ylabel('realgdp')
    ax.set_title('realgdbp prediction')
    ax.legend(loc='best')


    ax = axs[1]
    ax.plot(df['realcons'], label='realcons')
    ax.plot(all_CI['lower realcons'], 'k--', label='95% confidence interval')
    ax.plot(all_CI['upper realcons'], 'k--')
    ax.plot(pred['realcons'], label='forecast')
    ax.set_xlabel('year')
    ax.set_ylabel('realcons')
    ax.set_title('realcons prediction')
    ax.legend(loc='best')


    ax = axs[2]
    ax.plot(df['realinv'], label='realinv')
    ax.plot(all_CI['lower realinv'], 'k--', label='95% confidence interval')
    ax.plot(all_CI['upper realinv'], 'k--')
    ax.plot(pred['realinv'], label='forecast')
    ax.set_xlabel('year')
    ax.set_ylabel('realinv')
    ax.set_title('realinv prediction')
    ax.legend(loc='best')
    plt.show()

    return mod.aic

def manaus(start='1983-01-31', end='1995-01-31', p=4, q=4):
    """
    Plot the ARMA(p,q) model of the River Negro height
    data using statsmodels built-in ARMA class.

    Parameters:
        start (str): the data at which to begin forecasting
        end (str): the date at which to stop forecasting
        p (int): max_ar parameter
        q (int): max_ma parameter
    Return:
        aic_min_order (tuple): optimal order based on AIC
        bic_min_order (tuple): optimal order based on BIC
    """
    # Get dataset
    raw = pydata('manaus')
    # Make DateTimeIndex
    manaus = pd.DataFrame(raw.values, index=pd.date_range('1903-01', '1993-01', freq='M'))
    manaus = manaus.drop(0, axis=1)
    # Reset column names


###############################################################################
def kalman(F, Q, H, time_series):
    # Get dimensions
    dim_states = F.shape[0]

    # Initialize variables
    # covs[i] = P_{i | i-1}
    covs = np.zeros((len(time_series), dim_states, dim_states))
    mus = np.zeros((len(time_series), dim_states))

    # Solve of for first mu and cov
    covs[0] = np.linalg.solve(np.eye(dim_states**2) - np.kron(F, F), np.eye(dim_states**2)).dot(Q.flatten()).reshape(
            (dim_states, dim_states))
    mus[0] = np.zeros((dim_states,))

    # Update Kalman Filter
    for i in range(1, len(time_series)):
        t1 = np.linalg.solve(H.dot(covs[i-1]).dot(H.T), np.eye(H.shape[0]))
        t2 = covs[i-1].dot(H.T.dot(t1.dot(H.dot(covs[i-1]))))
        covs[i] = F.dot((covs[i-1] - t2).dot(F.T)) + Q
        mus[i] = F.dot(mus[i-1]) + F.dot(covs[i-1].dot(H.T.dot(t1))).dot(
                time_series[i-1] - H.dot(mus[i-1]))
    return mus, covs

def state_space_rep(phis, thetas, mu, sigma):
    # Initialize variables
    dim_states = max(len(phis), len(thetas) + 1)
    dim_time_series = 1 #hardcoded for 1d time_series

    F = np.zeros((dim_states, dim_states))
    Q = np.zeros((dim_states, dim_states))
    H = np.zeros((dim_time_series, dim_states))

    # Create F
    F[0][:len(phis)] = phis
    F[1:, :-1] = np.eye(dim_states - 1)
    # Create Q
    Q[0][0] = sigma**2
    # Create H
    H[0][0] = 1.
    H[0][1:len(thetas)+1] = thetas

    return F, Q, H, dim_states, dim_time_series

def main(key, verbose = False):
    if key == "1":
        if verbose:
            print("Testing ARMA naive.\n")
        arma_forecast_naive()

        if verbose:
            print("\nTest complete.")

    elif key =="2":
        if verbose:
            print("Testing arma likelihood.\n")

        AL = arma_likelihood()
        print(AL)

        if verbose:
            print("\nTest complete.")

    elif key == "3":
        if verbose:
            print("Testing model identification.\n")
        MI = model_identification()
        print(MI)
        if verbose:
            print("\nTest complete.")

    elif key == "4":
        if verbose:
            print("Testing arma forecast.\n")

        phis = np.array([ 0.72135856])
        thetas = np.array([-0.26246788])
        mu =  0.35980339870105321
        std = 1.5568331253098422

        arma_forecast(file="weather.npy", phis=phis, thetas=thetas, mu=mu, std=std, n=30)

        if verbose:
            print("\nTest complete.")

    elif key == "5":
        if verbose:
            print("Testing statsmodels.\n")

        print(sm_arma())

        if verbose:
            print("\nTest complete.")

    elif key == "6":
        if verbose:
            print("Testing VARMAX.\n")

        print(sm_varma())

        if verbose:
            print("\nTest complete.")
    return

if __name__ == "__main__":

    if len(sys.argv) == 1:
        pass

    elif len(sys.argv) == 2:
        main(sys.argv[1])

    elif len(sys.argv) == 3:
        if sys.argv[-1] == "--verbose":
            main(sys.argv[1], verbose=True)
        else:
            raise ValueError("Incorrect problem specification.")

    else:
        raise ValueError("Incorrect problem specification.")

