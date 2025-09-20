import pandas as pd
from pandas.tseries.offsets import DateOffset
import statsmodels.api as sm
import matplotlib.pyplot as plt

df=pd.read_csv('perrin-freres-monthly-champagne.csv')
df['Month']=pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
print(df.head())
df.plot()

model=sm.tsa.statespace.SARIMAX(df['Sales'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
results=model.fit()
print(results.summary())
future_dates=[df.index[-1]+ DateOffset(months=x)for x in range(0,60)]
future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df.columns)
future_df=pd.concat([df,future_datest_df])
future_df['forecast'] = results.predict(start = 104, end = 156, dynamic= True)
future_df[['Sales', 'forecast']].plot(figsize=(12, 8))

plt.show()