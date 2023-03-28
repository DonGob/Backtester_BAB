import pandas as pd
import numpy as np
import statsmodels.api as sm
from plotnine import *
from matplotlib import pyplot as plt
from statsmodels.regression.linear_model import OLS




class Trader:
    """
    Trader can look at past information and generate a portfolio with weights 
    for that month. This is the parent class for different types of traders.

    """
    def __init__(self, firm_info, leverage = 1, rebalancing_window = 1):
        """Initializes Trader object
        
        ...
        Attributes
        ----------
        monthly : pd.Dataframe
            Monthly return data for all companies
        yearly : pd.Dataframe
            yearly return data for all companies
        factors : pd.Dataframe
            Factor returns
        firm_info : pd.Dataframe
            Firm info
        """
        self.name = ""
        self.monthly = None
        self.yearly = None
        self.factors = None
        self.current_portfolio = None
        self.leverage = leverage
        self.firm_info = firm_info
        self.rebalancing_window = rebalancing_window
        self.portfolio_repeat_counter = 0


        
    def set_available_info(self, monthly_new, yearly_new, factors_new):
        """ Sets the ex ante information the trader has access to """
        self.monthly = monthly_new
        self.yearly = yearly_new
        self.factors = factors_new
        
    def set_name(self, name):
        self.name = name
        
    #This method will be overwritten by the different traders
    def get_portfolio(self):
        pass

class RandomTrader(Trader):
    """RandomTrader selects random portfolio every month"""
    
    def __init__(self, firm_info):
        Trader.__init__(self, firm_info)
        self.name = 'RandomTrader' 
    
    
    def get_portfolio(self):
        """
        Selects a random portfolio size and stocks with equal weights for this month
                 
        ...
        Parameters
        ----------
            
        Returns
        -------
        portfolio: np.array
            Array of the stock ISIN values in the portfolio of the trader
        weights: np.array
            Array of the corresponding weights for each stock
            
        """
        
        portfolio_size = np.random.randint(1, 10)
        weights = np.repeat(1/portfolio_size, portfolio_size)
        
        #Select random stocks from most recent month available at the time
        most_recent_date = np.max(self.monthly['mdate'])
        most_recent_returns = self.monthly[self.monthly['mdate'] == most_recent_date]
        
        #Select the random stocks
        portfolio = np.random.choice(most_recent_returns['ISIN'] , size=portfolio_size , replace=False)
        
        return portfolio, weights
        
class BaB(Trader):
    
    def __init__(self, firm_info, leverage, split_ratio = 0.5, rebalancing_window = 1):
        Trader.__init__(self, firm_info, leverage, rebalancing_window)
        self.name = 'BoB' 
        self.split_ratio = split_ratio
        self.riskfree_weight = 0 ## strategy uses 0 money so all available money goes into risk free asset
    
    def get_portfolio(self):
        if self.portfolio_repeat_counter == 0:
            self.portfolio_repeat_counter = self.rebalancing_window #how many repeats to go after this

            date = self.monthly['mdate'].max()
            current_data = self.monthly[self.monthly['mdate'] == date]
            current_data = current_data.sort_values('b')

            n_stocks = current_data.shape[0]
            stocks_per_side =int(np.round(n_stocks *self.split_ratio))

            low_beta = current_data[:stocks_per_side]['ISIN']
            high_beta = current_data[(n_stocks-stocks_per_side):]['ISIN']
            portfolio = pd.concat([low_beta, high_beta])

            stock_betas = current_data.loc[current_data['ISIN'].isin(portfolio)]['b']
            weights = np.flip(stock_betas / stock_betas.sum())

            leverage_adj_weights = weights *self.leverage

            
            current_beta_values = np.r_[current_data[:stocks_per_side]['b'], current_data[(n_stocks-stocks_per_side):]['b']]

            print(f'tests:weights sum: {weights.sum()}')
            print(f'weights: {weights @ current_beta_values}' )

            self.current_portfolio = [portfolio, leverage_adj_weights]
        self.portfolio_repeat_counter -= 1
        return self.current_portfolio[0], self.current_portfolio[1], self.riskfree_weight        
        

class BackTester:
    
    def __init__(self, Traders, monthly, yearly, factors, firm_info):
        """Initializes Backtesters object
        
        ...
        Attributes
        ----------
        Traders : List of Trader objects
            Traders to compare and backtest
        monthly : pd.Dataframe
            Monthly return data for all companies
        yearly : pd.Dataframe
            yearly return data for all companies
        factors : pd.Dataframe
            Factor returns
        firm_info : pd.Dataframe
            Firm info
        results : List of Dict
            List of dictionaries containing performance data of all trader
        """
        self.Traders = Traders
        self.monthly = monthly
        self.yearly = yearly
        self.factors = factors
        self.firm_info = firm_info
        self.results = []
        self.final_res = None
    
    def reset(self):
        self.results = []
        
    
    def add_result(self, result):
        self.results.append(result)
        
    def backtest(self, begindate, enddate):
        #Reset old data
        self.reset()
        
        #Get all dates
        dates = np.unique(self.monthly['mdate'])
        
        for date in dates:
            if date >= begindate and date <= enddate:
                #Get past info
                monthly_new = self.monthly[self.monthly['mdate'] < date]
                factors_new = self.factors[self.factors['mdate'] < date]
                #Date // 100 converts yearmonth to year
                yearly_new = self.yearly[self.yearly['fyear'] < date // 100]
                rf_rate = factors_new.iloc[-1]['RF']
                #Let each trader act
                for trader in self.Traders:
                    #Supply trader with past information 
                    trader.set_available_info(monthly_new, yearly_new, factors_new)
                    
                    #Get response from trader and calculate return
                    portfolio, weights, riskfree_weight = trader.get_portfolio()
                    ret = self.calc_return(portfolio, weights, riskfree_weight, rf_rate , date)
                    
                    #Save result in result dict and add to results
                    result = {
                        "trader": trader.name,
                        "RET": ret,
                        "mdate": date
                        }
                    
                    self.add_result(result)
        
                #Add benchmarks
                ret = self.factors[self.factors['mdate'] == date]
                for benchmark in ["MktRF", "SMB", "HML"]:
                    result = {
                        "trader": benchmark,
                        "RET": ret.iloc[0][benchmark],
                        "mdate": date
                        }
                    self.add_result(result)

    def get_results(self):
        df = pd.DataFrame(self.results)
        
        df['grossRET'] = 1 + df['RET']
        df['cumRET'] = df.groupby(['trader']).grossRET.cumprod()
        df['date'] = pd.to_datetime(
            df['mdate'],format='%Y%m') + pd.tseries.offsets.MonthEnd(1)
        self.final_res = df
        return df
    
    def get_discriptive_stats(self):
        
        res = self.get_results()
        
        return res.groupby('trader')['RET'].describe()
        
        

    
    def calc_return(self, portfolio, weights, riskfree_weight, rf_rate, mdate):
        #Find the returns of all stocks on mdate
        all_returns = self.monthly[self.monthly['mdate'] == mdate]
        
        returns = []
        for stock, weight in zip(portfolio, weights):
            try:
                r = all_returns[all_returns['ISIN'] == stock]['RET'].iloc[0]*weight
                returns.append(r)
            except:
                #Lots of companies do not exist or have missing data so for now we ignore the error
                pass
        returns.append(riskfree_weight*rf_rate)    
            
        return np.sum(returns)
    
    def get_sharpe(self): #still buggy
        # if self.final_res != pd.core.frame.DataFrame:
        #     print(f'final results not calculated yet! running get results for the first time')
        #     self.get_results()
        vol = (self.final_res.groupby('trader')['cumRET']).std() * np.sqrt(12)
        years_held = 20 # magic number. we held for 20 years in this strategy
        final_returns = self.final_res.groupby('trader').max('date')['RET']
        sharpe = (final_returns **(1/years_held))/vol
        return sharpe

def get_information(X, y):
    ols = OLS(y, X).fit()
    params = ols.params
    information = np.sqrt(12)*ols.params['const']/(ols.resid.std())
    print(ols.summary())
    return information
        
     
monthly = pd.read_csv('AU_NZ_SG_data_monthly.csv')
yearly  = pd.read_csv('NL_FR_BE_data_annual.csv')   
factors = pd.read_csv('Asia_Pacific_ex_Japan_FF_Factors.csv')
firm_info =  pd.read_csv('AU_NZ_SG_firms.csv')

trader3 = BaB(firm_info, leverage=1 ,split_ratio=0.1, rebalancing_window=1)
trader3.set_name('BAB 1')

Traders= [trader3]

#Initialize backtester
backtester = BackTester(Traders, monthly, yearly, factors, firm_info)

backtester.backtest(200001, 201912)

res = backtester.get_results()
stats = backtester.get_discriptive_stats()
print(stats)




y = (res[res['trader'] == 'BAB 1']['grossRET'] - 1)
X = factors[factors['mdate'] >= 200001][factors['mdate'] <=201912][["MktRF","SMB","HML"]].reset_index(drop=True)

print(y, X)
y = y.reset_index(drop=True)
X = sm.add_constant(X)

print(f'test Y: {y}')

IR = get_information(X, y)

print(f'information ratio is: {IR}')

print(
    ggplot(res)
    + aes(x = 'date', y = 'cumRET', color = 'trader')
    + geom_line()
    + xlab('Date')
    + ylab('Portfolio value in USD')
    + scale_y_log10()
    + scale_x_date(breaks = ('1 year'))
    + theme(axis_text_x = element_text(rotation = 90))
)