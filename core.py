"""
Core implementation of time series modelling project.

This module contains both the primary implementation and sanity check 
validation code used during development. The structure is
intentionally consolidated at this stage and will be refactored into
separate modules and formal tests once the design stabilizes.

See notes.pdf for the mathematical background.
"""

# configuration flags
REDOWNLOAD = False # this controls whether yfinance is used to redownload data. otherwise, data is pulled from pkl files
VERBOSE = True # this toggles lots of console output and plotting of figures, useful for troubleshooting
RECOMPUTE_FACTORS = True # this controls whether to recompute X, f, F, D (since this is kinda slow and independent of backtesting)


# date range for data
START = "2006-01-01"
END = "2025-03-07"

# market index to use
INDEX = "^GSPC" # market index is the sp500 index


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cvxpy as cp

def beta(r_i, r_m, w): # this loses w days from the start
    var = r_m.rolling(w, min_periods=w).var() # compute rolling variance of the index
    beta = r_i.rolling(w, min_periods=w).cov(r_m.iloc[:, 0]) # compute covariances with each security
    return beta.div(var.iloc[:, 0], axis=0) # return the quotient

def volatility(r_i, w): # this loses w days from the start
    return r_i.rolling(w, min_periods=w).std() # sample std dev over w

def idio_volatility(r_i, r_m, beta, w): # this loses 2w days from the start
    residuals = r_i - beta.mul(r_m.iloc[:, 0], axis=0) # compute residuals
    return residuals.rolling(w, min_periods=w).std() # sample std dev of residuals over w

def momentum(p_i): # this loses 252 days from the start
    return (p_i.shift(21) / p_i.shift(252)) - 1 # daily momentum score

def reversal(p_i): # this loses 21 days from the start
    return  (-p_i / p_i.shift(21)) + 1 # daily reversal score

def illiquidity(r_i, p_i, v_i): # this loses 252 days from the start
    ratio = r_i.abs() / (v_i * p_i) # compute the illiquidity ratio
    return ratio.rolling(252, min_periods=252).sum() * (1/252) # average over past year

def exposure_matrix(descriptors): # this loses 0 days from the start
    frames = []
    for i in range(len(descriptors)):
        d = descriptors[i][0]
        factor_name =  descriptors[i][1] # get the factor name
        normalized = d.sub(d.mean(axis=1), axis=0).div(d.std(axis=1), axis=0) # subtract mean and divide by std dev
        normalized.columns = pd.MultiIndex.from_product([[factor_name], normalized.columns]) # create dataframe for the factor
        frames.append(normalized)
    X = pd.concat(frames, axis=1)
    return X

def factor_returns(X, r_i, w): # this loses w days from the start
    ones = pd.DataFrame(1, index=r_i.index, columns=r_i.columns) # frame of ones, to act as gls weights to do ols
    f_ols = gls(X, r_i, ones) # compute ols factor returns
    residuals = residual_returns(X, r_i, f_ols) # compute residuals based on ols factor returns
    var_i = residuals.rolling(w, min_periods=w).var() # compute new gls weights
    f_gls = gls(X, r_i, var_i) # compute gls factor returns
    return (f_ols,f_gls)

def residual_returns(X, r_i, f): # this loses 0 days from the start
    common_dates = X.index.intersection(r_i.index).intersection(f.index) # get common dates
    residuals = []
    for date in common_dates:
        Xt = X.loc[date].unstack().transpose() # reconstruct X matrix
        rt = r_i.loc[date]
        ft = f.loc[date]
        residual = rt - Xt.dot(ft) # compute residual return r - Xf for each date
        residuals.append(residual)
    residuals = pd.DataFrame(residuals, index=common_dates)
    return residuals

def gls(X, r_i, var_i): # this loses 0 days from the start
    common_dates = X.index.intersection(r_i.index).intersection(var_i.index) # get common dates
    f_gls = []
    for date in common_dates: 
        Xt = X.loc[date].unstack().transpose() # reconstruct X matrix
        factors = Xt.columns # get labels
        rt = r_i.loc[date]
        variance = var_i.loc[date]
        common_tickers = Xt.index.intersection(rt.index).intersection(variance.index) # get common tickers
        rt = rt.loc[common_tickers].values 
        D = np.diag(1/(variance.loc[common_tickers].values)) # construct D matrix from variances
        D = pd.DataFrame(D, index=Xt.index, columns=Xt.index)
        XtX = Xt.T @ D @ Xt # setup the gls system
        Xtr = Xt.T @ D @ rt
        if not XtX.isna().values.any(): # print warning when XtX matrix poorly conditioned
            condition = np.linalg.cond(XtX)
            if condition > 100000:
                print("caution! condition numbers of XtX, X are " + str(condition) + ", " + str(np.linalg.cond(Xt)))
        ft = np.linalg.solve(XtX, Xtr) # do gls regression
        ft = pd.Series(ft, index=factors, name=date) # put result in series
        f_gls.append(ft)
    f_gls = pd.DataFrame(f_gls)
    return f_gls

def ewma_covariance(f, w, lamb): # this loses w days from the start
    f_mean = ewma(f, w, lamb) # compute ewma mean on lookback window of length w for each date
    factors = f.columns # get labels
    covs = []
    for date in f.index: # iterate over dates and consider windows of length w ending on each date
        mean = f_mean.loc[date] # this is the ewma mean over the present window
        f_window = f.loc[:date].tail(w) # this is the dataframe of factor returns during the present window
        f_res = f_window - mean # compute demeaned factor returns during the present window
        frames = [] 
        for i in range(len(factors)): # iterate through columns of demeaned factor returns to compute outer product
            col = f_res.iloc[:,i].to_numpy
            product = f_res.mul(f_res.iloc[:,i].values, axis=0)
            multi_index = pd.MultiIndex.from_product([[factors[i]], f_res.columns])
            product.columns = multi_index
            frames.append(product)
        products = pd.concat(frames, axis=1) # this should result in a series of symmetric matrices spanning the present window
        ewma_cov = ewma(products, w, lamb).loc[date] # take the ewma over the present window

        if not np.isnan(np.sum(ewma_cov.unstack().values)) and not np.all(np.linalg.eigvals(ewma_cov.unstack().values) >= 0):
            print("caution! factor covariance matrix not positive semidefinite") # hopefully this can catch errors
          
        
        covs.append(ewma_cov)
    ewma_covs = pd.DataFrame(covs, index=f.index)
    return ewma_covs

def ewma_variance(f, w, lamb): # this loses w days from the start
    f_mean = ewma(f, w, lamb) # series of ewma means on windows of length w
    variances = []
    for date in f.index:
        mean = f_mean.loc[date] # the ewma mean on the window ending on the present date
        f_window = f.loc[:date].tail(w) # this is the dataframe of factor returns during the present window
        f_res = (f_window - mean)**2 # compute the variance during the present window
        variance = ewma(f_res, w, lamb).loc[date] # compute the ewma mean over the present window
        variances.append(variance)
    return pd.DataFrame(variances, index=f.index)

def shrink_id(F, intensity): # this shrinks a factor covariance matrix toward a scaled identity by a given intensity
    series = []
    for date in F.index:
        rows = F.loc[date].unstack().index # get labels
        Ft = F.loc[date].unstack() # reconstruct matrix
        m = Ft.values.trace() / len(rows) # average of the diagonal entries of the covariance matrix
        target = m*np.eye(len(rows)) # this is the scaled identity
        target = pd.DataFrame(target, index=rows, columns=rows)
        result = intensity * target + (1 - intensity) * Ft # linear combo
        series.append(result.stack())
    return pd.DataFrame(series, index=F.index)
  
def shrink_diag(F, intensity): # this shrinks a factor covariance matrix toward its diag by a given intensity
    series = []
    for date in F.index:
        rows = F.loc[date].unstack().index
        Ft = F.loc[date].unstack()
        target = Ft * np.eye(len(rows)) # same as shrink_id but the target is the diagonal of F
        target = pd.DataFrame(target, index=rows, columns=rows)
        result = intensity * target + (1 - intensity) * Ft
        series.append(result.stack())
    return pd.DataFrame(series, index=F.index)
   
def lw_covariance(f, w, lamb): # this loses w days from the start, implements ledoit-wolf shrinkage (currently probably broken)
    f_mean = ewma(f, w, lamb)
    factors = f.columns
    F_ewma = ewma_covariance(f, w, lamb)
    covs = []
    for date in F_ewma.index:
        mean = f_mean.loc[date]
        f_window = f.loc[:date].tail(w)
        f_res = f_window - mean
        frames = []
        for i in range(len(factors)):
            col = f_res.iloc[:,i].values
            product = f_res.mul(f_res.iloc[:,i].values, axis=0)
            multi_index = pd.MultiIndex.from_product([[factors[i]], f_res.columns])
            product.columns = multi_index
            frames.append(product)
        products = pd.concat(frames, axis=1)
        b = ewma(((products - F_ewma.loc[date]) ** 2).sum(axis=1), w, lamb*lamb)[date]
        #print(products)
        #print(products - F_ewma.loc[date])
        #print(((products - F_ewma.loc[date]) ** 2).sum(axis=1))
        #print(ewma(((products - F_ewma.loc[date]) ** 2).sum(axis=1), w_ewma, lamb*lamb))
        Ft = F_ewma.loc[date].unstack().transpose().values
        m = Ft.trace() / len(factors)
        target = m*np.eye(len(factors))
        d = np.linalg.norm(Ft - target, ord='fro') ** 2 # frobenius norm squared
        print(b/d)
        b = min(b,d)
        intensity = min(max(b/d,0),1)
        print(intensity)
    return

def ewma_specific(X, r_i, f, w, lamb): # this loses w days from the start, computes time series of ewma D matrix 
    residuals = residual_returns(X, r_i, f)
    return ewma(residuals**2, w, lamb)
 
def ewma(series, w, lamb): # this loses w days from the start, computes time series of ewma means of any series
    weights = lamb ** np.arange(w) # compute decaying weights
    weights = weights[::-1]         
    W = 1 / weights.sum() # normalize weights
    def ewma_window(x):
        return np.dot(x, weights) * W
    return series.rolling(w, min_periods=w).apply(ewma_window, raw=True)

def condition(X): # compute condition number of factor exposure matrix X
    conds = []
    for date in X.index: # iterate over dates
        Xt = X.loc[date].unstack().transpose().values # reconstruct numpy X matrix
        if not np.isnan(np.sum(Xt)): # only compute if no nans
            conds.append(np.linalg.cond(Xt))
        else: # if nans, return nan
            conds.append(np.nan)
    return pd.DataFrame(conds, index=X.index)














def run_pipeline(): 
    """
        Main pipeline for factor model computation and exploratory diagnostics.
    """
    # ------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------
    if REDOWNLOAD:
        sp500 = pd.read_csv("SP500.csv") # read in a list of sp500 tickers
        securities = sp500["Symbol"].tolist()
        market_data = yf.download(INDEX, start=START, end=END, auto_adjust=False) # download data for index
        market_dates = market_data.dropna(axis=0, how='any').index # get dates for which market index has data
        securities_used = []
        for sec in securities: # only include a ticker if it has full data
            securities_data = yf.download([sec], start=START, end=END, auto_adjust=False)
            if market_dates.equals(market_dates.intersection(securities_data.dropna(axis=0, how='any').index)):
                print(sec + " has full data")
                securities_used.append(sec)
        print(str(len(securities_used)) + " securities in use") # print number of tickers in the universe

        securities_data = yf.download(securities_used, start=START, end=END, auto_adjust=False) # download the data (this is kinda redundant but convenient)
        
        securities_data.to_pickle("securities_data.pkl") # save data to pkl files
        market_data.to_pickle("market_data.pkl")
    else:
        securities_data = pd.read_pickle("securities_data.pkl") # fetch data from pkl files to avoid redownloading every time
        market_data  = pd.read_pickle("market_data.pkl")

    # ------------------------------------------------------------
    # Descriptor Computation
    # ------------------------------------------------------------
    p_i = securities_data["Adj Close"] # adjusted close price for securities up to the present
    p_m = market_data["Adj Close"] # adjusted close price for market index up to the present
    r_i = p_i.pct_change() # compute daily returns on securities
    r_m = p_m.pct_change() # compute daily returns on market index
    v_i = securities_data["Volume"] # volume for securities up to the present
    v_i = v_i.replace(0, np.nan).ffill() # volume occasionally has 0 for a security on a particular day. in this case, use volume from prev day

    if VERBOSE:
        print(str(len(p_i.columns)) + " securities in use")

    w = 252  # lookback window length for descriptors
    w_var = 252 # window for computing sample variance for computing factor returns
    lamb = 0.99 # parameter for ewma 
    w_ewma = 252 # lookback window for ewma factor covariance, factor return, and specific risk

    # ------------------------------------------------------------
    # Factor Computation
    # ------------------------------------------------------------
    if RECOMPUTE_FACTORS:

        d_mom = momentum(p_i)
        d_beta = beta(r_i, r_m, w)
        d_ivol = idio_volatility(r_i, r_m, d_beta, w)
        d_vol = volatility(r_i, w)
        d_rev = reversal(p_i)
        d_illiq = illiquidity(r_i, p_i, v_i)

        if VERBOSE:
            # checking how many dates each descriptor is valid on
            print("MOM   has " + str(len(d_mom.index)) + " days, of which " + str(len(d_mom.dropna(axis=0, how='any').index)) + " are not nan")
            print("BETA  has " + str(len(d_beta.index)) + " days, of which " + str(len(d_beta.dropna(axis=0, how='any').index)) + " are not nan")
            print("IVOL  has " + str(len(d_ivol.index)) + " days, of which " + str(len(d_ivol.dropna(axis=0, how='any').index)) + " are not nan")
            print("VOL   has " + str(len(d_vol.index)) + " days, of which " + str(len(d_vol.dropna(axis=0, how='any').index)) + " are not nan")
            print("REV   has " + str(len(d_rev.index)) + " days, of which " + str(len(d_rev.dropna(axis=0, how='any').index)) + " are not nan")
            print("ILLIQ has " + str(len(d_illiq.index)) + " days, of which " + str(len(d_illiq.dropna(axis=0, how='any').index)) + " are not nan")
        # removing illiq descriptor reduces condition number from like 4500 to like 800
        descriptors = [(d_mom, "MOM"), (d_beta, "BETA"), (d_ivol, "IVOL"), (d_vol, "VOL"), (d_rev, "REV")] # , (d_illiq, "ILLIQ")] 
        X = exposure_matrix(descriptors)

        if VERBOSE:
            # checking and plotting mean and std of X matrix over the whole period
            print("X matrix: ")
            print(X)
            print("X matrix mean over whole period: ")
            print(X.mean().unstack().transpose())
            print("X std over whole period: ")
            print(X.std().unstack().transpose())

            # plotting condition number of X matrix:
            conds = condition(X)
            print("X has " + str(len(conds.dropna().index)) + " non nan days")
            conds.plot()
            plt.xlabel('Date')
            plt.ylabel('Condition number')
            plt.title('Condition of factor exposure matrix over time')
            plt.show()


        (f_ols,f_gls) = factor_returns(X, r_i, w_var) # compute factor returns


        if VERBOSE:
            # plotting ols and gls factor returns:
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)

            f_ols.plot(ax=axes[0], title='OLS factor returns')
            axes[0].set_ylabel('Factor return')
            axes[0].grid(True)

            f_gls.plot(ax=axes[1], title='GLS factor returns')
            axes[1].set_xlabel('Date')
            axes[1].set_ylabel('Factor return')
            axes[1].grid(True)

            plt.tight_layout()
            plt.show()

            # checking explanatory power of ols and gls factor returns (this is pretty unaffected by exclusion of illiq)
            date = pd.Timestamp("2025-01-07")
            ols_residuals = residual_returns(X, r_i, f_ols)
            gls_residuals = residual_returns(X, r_i, f_gls)
            ols_residual_var = ols_residuals.rolling(w, min_periods=w).var()
            gls_residual_var = gls_residuals.rolling(w, min_periods=w).var()
            return_var = r_i.rolling(w, min_periods=w).var()

            print("Mean and median residual return variance for OLS factor returns")
            print((ols_residual_var.loc[date]/return_var.loc[date]).mean()) # mean and median ratios
            print((ols_residual_var.loc[date]/return_var.loc[date]).median())
            print("Mean residual return over time for OLS factor returns")
            print(max(abs(ols_residuals.mean().values))) # check to see if average residuals over time are close to zero
            print("Mean and median residual return variance for GLS factor returns")
            print((gls_residual_var.loc[date]/return_var.loc[date]).mean())
            print((gls_residual_var.loc[date]/return_var.loc[date]).median())
            print("Mean residual return over time for GLS factor returns")
            print(max(abs(gls_residuals.mean().values))) # check to see if average residuals over time are close to zero


        D = ewma_specific(X, r_i, f_gls, w_ewma, lamb) # compute specific risks
        F = ewma_covariance(f_gls, w_ewma, lamb) # compute factor covariances


        if VERBOSE:
            # check factor covariance matrix diagonal entries against ewma factor variances
            print("diagonals of factor covariance matrix should match factor variances:")
            date = pd.Timestamp("2025-01-07")
            f_variances = ewma_variance(f_gls,w_ewma, lamb)
            print(F.loc[date].unstack())
            print(f_variances.loc[date])

            # plotting condition number of F matrix:
            conds = condition(F)
            conds.plot()
            plt.xlabel('Date')
            plt.ylabel('Condition number')
            plt.title('Condition of factor covariance matrix over time')
            plt.show()

            # plotting variances from the model against rolling variances of returns
            date = pd.Timestamp("2025-01-07")
            XFX = X.loc[date].unstack().transpose().dot(F.loc[date].unstack()).dot(X.loc[date].unstack())
            var_factors = pd.Series(np.diag(XFX), index=XFX.index)
            var_model = var_factors + D.loc[date]
            var_actual = ewma_variance(r_i, w_ewma, lamb).loc[date]
            var_error = var_actual - var_model
            var_combined = pd.concat([var_factors, var_model, var_actual, var_error], axis=1)
            var_combined.columns = ['factor variance', 'model variance', 'actual variance', 'actual minus model']
            var_every_10 = var_combined.iloc[::10]  
            ax = var_every_10.plot(kind='bar', figsize=(10,6))
            ax.set_xlabel('Stock')
            ax.set_ylabel('Variance')
            ax.set_title('Variances')
            plt.xticks(rotation=0)   
            plt.grid(axis='y')
            plt.show()

        #Fl = lw_covariance(f_gls, w_ewma, lamb)





        # setup for portfolio optimization:


        F = shrink_id(F, 0.5)

        f = ewma(f_gls, w_ewma, lamb)


        X.to_pickle("X.pkl") # save X, f, F, D time series to avoid recomputing, since this is a little slow
        f.to_pickle("f.pkl")
        F.to_pickle("F_mat.pkl")
        D.to_pickle("D.pkl")

    else:

        X = pd.read_pickle("X.pkl") # these are ready to use for any backtesting
        f  = pd.read_pickle("f.pkl")
        F = pd.read_pickle("F_mat.pkl")
        D = pd.read_pickle("D.pkl") 

    #print(F)
    #print(f)

    # ------------------------------------------------------------
    # Formulate Portfolio
    # ------------------------------------------------------------

    '''

    # rebuild X and D for the portfolio universe
    portfolio_universe = ['MMM', 'AOS', 'ABT', 'ABBV', 'ACN', 'AYI', 'ADBE', 'AAP', 'AMD', 'AES', 'AMG', 'AFL', 'A', 'APD', 'AKAM', 'ALK', 'ALB', 'ARE', 'ALGN', 'ALLE', 'LNT']
    # smaller portfolio universe can be used to reduce computation when solving QPs
    d_mom = d_mom[portfolio_universe]
    d_beta = d_beta[portfolio_universe]
    d_ivol = d_ivol[portfolio_universe]
    d_vol = d_vol[portfolio_universe]
    d_rev = d_rev[portfolio_universe]

    descriptors = [(d_mom, "MOM"), (d_beta, "BETA"), (d_ivol, "IVOL"), (d_vol, "VOL"), (d_rev, "REV") ] 
    X = exposure_matrix(descriptors)
    D = ewma_specific(X, r_i[portfolio_universe], f_gls, w_ewma, lamb)
    #print(X)
    #print(D)
    # new X, D are computed but f and F are from the big universe
    '''

    # ------------------------------------------------------------
    # Solve Portfolio
    # ------------------------------------------------------------

    '''
    # this section rebalances (solves QP to compute weights) on the first day of each month

    firsts = X.groupby([X.index.year, X.index.month]).head(1).index # get list of first days of each month
    weights = []
    print(firsts)

    for date in firsts:
        Xt = X.loc[date].unstack().transpose().values # reconstruct X matrix
        Ft = F.loc[date].unstack().values
        Dt = np.diag(D.loc[date].values)
        ft = f.loc[date].values

        n_securities, n_factors = Xt.shape # get number of securities and number of factors

        if not np.isnan(np.sum(Xt)) and not np.isnan(np.sum(Ft)) and not np.isnan(np.sum(Dt)) and not np.isnan(np.sum(ft)): # check for nans

            #print(Xt)
            #print(Ft)
            #print(Dt)
            #print(ft)
        
            w = cp.Variable(n_securities) # variable for QP solver

            Sigma = Xt @ Ft @ Xt.T + Dt  # n×n

            alpha = Xt @ ft            # n-vector

            # Risk aversion parameter
            lam = 1.0  # choose your λ

            # Objective: maximize alpha' w - (λ/2) w' Σ w
            objective = cp.Maximize(alpha @ w - 0.5 * lam * cp.quad_form(w, Sigma))

            # Constraints
            constraints = []

            # Example constraints:
            constraints += [cp.sum(w) == 0]                # dollar neutrality
            constraints += [cp.norm1(w) <= 2.0]    # max 200% gross exposure
            #constraints += [w <= 0.05]                    # upper bound (example)
            #constraints += [w >= -0.05]                   # lower bound (example)
            # constraints += [X.T @ w == 0]               # optional factor neutrality

            # Problem
            prob = cp.Problem(objective, constraints)
            prob.solve()
            print(str(date))
            weights.append(pd.Series(w.value, index=D.columns, name=date))

    weights = pd.DataFrame(weights)

    weights.plot()
    plt.xlabel('Date')
    plt.ylabel('weights')
    plt.title('weights over time')
    plt.show()

    '''

if __name__ == "__main__":
    # Enable exploratory output when run as a script
    VERBOSE = True
    run_pipeline()
