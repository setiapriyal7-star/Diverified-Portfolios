import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from statsmodels.api import OLS, add_constant

st.set_page_config(page_title='PortfolioIQ | Diversification Lab', page_icon='📊', layout='wide')

STOCKS = {
    'Reliance Industries': 'RELIANCE.NS',
    'TCS': 'TCS.NS',
    'HDFC Bank': 'HDFCBANK.NS',
    'ITC': 'ITC.NS',
    'Larsen & Toubro': 'LT.NS',
}

@st.cache_data(ttl=3600, show_spinner=False)
def get_market_data(period: str):
    tickers = list(STOCKS.values()) + ['^NSEI']
    raw = yf.download(tickers, period=period, interval='1d', auto_adjust=True, progress=False, threads=True)
    if raw is None or raw.empty:
        raise ValueError('No market data was returned. Check your internet connection and refresh.')
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close'].copy()
    else:
        close = raw[['Close']].copy()
        if len(tickers) == 1:
            close.columns = tickers
    close = close.rename(columns={'^NSEI': 'NIFTY 50'}).dropna(how='all').ffill()
    available = [ticker for ticker in STOCKS.values() if ticker in close.columns]
    if len(available) < 3:
        raise ValueError('Fewer than three stock histories were returned. Try refreshing later.')
    keep = available + (['NIFTY 50'] if 'NIFTY 50' in close.columns else [])
    return close[keep].dropna(subset=available, how='all')

st.markdown('''
<style>
.block-container {padding-top: 1.7rem; padding-bottom: 2.5rem; max-width: 1450px;}
.hero {padding: 1.35rem 1.55rem; border-radius: 18px; background: linear-gradient(115deg,#10243a,#203f59); color: white; margin-bottom: 1rem;}
.hero h1 {font-size: 2rem; margin: 0 0 .35rem 0; color: white;}
.hero p {margin: 0; color: #d8e5ef; font-size: .98rem;}
.rec-card {padding: 1.2rem 1.35rem; border: 1px solid rgba(128,145,160,.35); border-radius: 15px; background: rgba(128,145,160,.07);}
.small-muted {color: #82909c; font-size: .86rem;}
div[data-testid="stMetric"] {border: 1px solid rgba(128,145,160,.25); padding: 12px 14px; border-radius: 12px;}
</style>
''', unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>PortfolioIQ</h1><p>A practical stock diversification and portfolio decision-support dashboard</p></div>', unsafe_allow_html=True)
st.caption('Five NSE-listed companies · Actual historical adjusted closing prices · Educational analysis, not investment advice')

with st.sidebar:
    st.header('Investor profile')
    investor = st.text_input('Your name', 'Investor')
    risk = st.selectbox('Risk appetite', ['Low', 'Moderate', 'High'], index=1)
    horizon = st.selectbox('Investment horizon', ['Less than 1 year', '1–3 years', 'More than 3 years'], index=1)
    period_label = st.selectbox('Historical data period', ['1 year', '2 years', '5 years'], index=1)
    period = {'1 year': '1y', '2 years': '2y', '5 years': '5y'}[period_label]
    st.caption('Prices refresh automatically once per hour. Use the button to request a fresh download.')
    if st.button('↻ Refresh market data', use_container_width=True):
        get_market_data.clear()

try:
    with st.spinner('Loading actual NSE historical prices…'):
        prices = get_market_data(period)
except Exception as exc:
    st.error(f'Unable to load actual market data: {exc}')
    st.info('Check your internet connection. In the VS Code terminal, run: `py -3.11 -m pip install --upgrade yfinance` and then refresh.')
    st.stop()

name_by_ticker = {ticker: name for name, ticker in STOCKS.items()}
prices = prices.rename(columns=name_by_ticker)
stock_cols = [name for name in STOCKS if name in prices.columns]
if len(stock_cols) < 3:
    st.error('Not enough stock histories are available to calculate the analysis.')
    st.stop()

latest_date = prices.index.max().date()
summary_cols = st.columns(4)
summary_cols[0].metric('Stocks available', len(stock_cols))
summary_cols[1].metric('Latest price date', latest_date.strftime('%d %b %Y'))
summary_cols[2].metric('Trading days', f'{len(prices):,}')
summary_cols[3].metric('Investor', investor or 'Investor')

selected = st.multiselect('Select stocks for your analysis', stock_cols, default=stock_cols)
if len(selected) < 3:
    st.warning('Select at least three stocks to run the portfolio analysis.')
    st.stop()

p = prices[selected].dropna(how='all').ffill().dropna()
r = p.pct_change().dropna()
if len(r) < 30:
    st.error('There are not enough daily observations to calculate reliable statistics. Select a longer historical period.')
    st.stop()

# Portfolio definitions: A equal weight; B inverse volatility; C inverse individual variance proxy.
vol_daily = r.std().replace(0, np.nan)
var_daily = r.var().replace(0, np.nan)
w_a = pd.Series(1 / len(selected), index=selected)
w_b = (1 / vol_daily).replace([np.inf, -np.inf], np.nan).fillna(0)
w_b = w_b / w_b.sum() if w_b.sum() else w_a.copy()
w_c = (1 / var_daily).replace([np.inf, -np.inf], np.nan).fillna(0)
w_c = w_c / w_c.sum() if w_c.sum() else w_a.copy()
weights = pd.DataFrame({'Portfolio A': w_a, 'Portfolio B': w_b, 'Portfolio C': w_c})
port_returns = pd.DataFrame({name: r.dot(weights[name]) for name in weights.columns})

comparison = []
for name, series in port_returns.items():
    annual_return = ((1 + series.mean()) ** 252 - 1) * 100
    annual_vol = series.std() * np.sqrt(252) * 100
    wealth = (1 + series).cumprod()
    max_drawdown = (wealth / wealth.cummax() - 1).min() * 100
    sharpe = (series.mean() / series.std() * np.sqrt(252)) if series.std() else np.nan
    comparison.append({'Portfolio': name, 'Annualised historical return (%)': annual_return,
                       'Annualised volatility (%)': annual_vol, 'Maximum drawdown (%)': max_drawdown,
                       'Sharpe ratio (historical)': sharpe})
comparison_df = pd.DataFrame(comparison).set_index('Portfolio')

analysis_tab, portfolio_tab, recommendation_tab, methodology_tab = st.tabs([
    '📈 Stock Analysis', '⚖️ Portfolios A, B & C', '✦ Recommendations', 'ⓘ Methodology'
])

with analysis_tab:
    st.subheader('Actual stock prices')
    latest = p.iloc[-1]
    average = p.mean()
    price_summary = pd.DataFrame({
        'Latest available close (₹)': latest,
        'Average daily close (₹)': average,
        'First date': p.apply(lambda s: s.first_valid_index().date()),
        'Latest date': p.apply(lambda s: s.last_valid_index().date())
    })
    st.dataframe(price_summary.style.format({'Latest available close (₹)': '₹{:,.2f}', 'Average daily close (₹)': '₹{:,.2f}'}), use_container_width=True)
    st.plotly_chart(px.line(p, x=p.index, y=selected, title='Historical adjusted closing prices (₹)',
                            labels={'value': 'Price (₹)', 'index': 'Date', 'variable': 'Stock'}), use_container_width=True)

    st.subheader('Individual stock risk and return')
    metrics = pd.DataFrame(index=selected)
    metrics['Average daily return (%)'] = r.mean() * 100
    metrics['Annualised volatility (%)'] = r.std() * np.sqrt(252) * 100
    metrics['Daily variance'] = r.var()
    metrics['Annualised return (%)'] = ((1 + r.mean()) ** 252 - 1) * 100
    st.dataframe(metrics.style.format('{:,.4f}'), use_container_width=True)

    st.subheader('Correlation analysis')
    corr = r.corr()
    st.plotly_chart(px.imshow(corr, text_auto='.2f', zmin=-1, zmax=1, color_continuous_scale='RdBu_r',
                              title='Correlation matrix of daily stock returns'), use_container_width=True)
    st.caption('Correlation ranges from −1 to +1. Lower correlations may improve diversification, but do not eliminate risk.')

    st.subheader('SVD: common movement factors')
    scaled = StandardScaler().fit_transform(r)
    n_components = min(4, len(selected), scaled.shape[0], max(1, scaled.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(scaled)
    factor_names = [f'Factor {i + 1}' for i in range(n_components)]
    loadings = pd.DataFrame(svd.components_.T, index=selected, columns=factor_names)
    explained = pd.Series(svd.explained_variance_ratio_ * 100, index=factor_names, name='Variance explained (%)')
    left, right = st.columns(2)
    left.markdown('**Variance explained by each factor**')
    left.dataframe(explained.to_frame().style.format('{:.2f}%'), use_container_width=True)
    right.markdown('**Stock factor loadings**')
    right.dataframe(loadings.style.format('{:.3f}'), use_container_width=True)
    st.caption('SVD identifies mathematical patterns in the combined return data. A factor is not automatically a named economic cause.')

    st.subheader('Regression against NIFTY 50')
    if 'NIFTY 50' in prices.columns:
        benchmark = prices['NIFTY 50'].pct_change().reindex(r.index)
        reg_data = pd.concat([r.mean(axis=1).rename('Selected-stock average'), benchmark.rename('NIFTY 50')], axis=1).dropna()
        if len(reg_data) >= 20:
            regression = OLS(reg_data['Selected-stock average'], add_constant(reg_data['NIFTY 50'])).fit()
            m1, m2, m3 = st.columns(3)
            m1.metric('Market beta', f'{regression.params.iloc[1]:.3f}')
            m2.metric('R-squared', f'{regression.rsquared:.3f}')
            m3.metric('Observations', f'{len(reg_data):,}')
        else:
            st.warning('Not enough overlapping benchmark observations for regression.')
    else:
        st.warning('NIFTY 50 data was unavailable, so regression is skipped.')

with portfolio_tab:
    st.subheader('Compare three portfolio strategies')
    st.write('Each portfolio uses the same selected stocks and historical period, but applies a different weighting rule.')
    pta, ptb, ptc = st.tabs(['Portfolio A · Equal weight', 'Portfolio B · Lower volatility', 'Portfolio C · Lower variance'])
    portfolio_details = [
        (pta, 'Portfolio A — Equal-weighted', 'Every selected stock receives the same allocation.', w_a, 'A simple baseline portfolio; easy to understand and compare.'),
        (ptb, 'Portfolio B — Inverse-volatility weighted', 'Stocks with lower individual historical volatility receive higher weights.', w_b, 'Aims to reduce the influence of more volatile individual stocks.'),
        (ptc, 'Portfolio C — Inverse-variance weighted', 'Stocks with lower individual historical variance receive higher weights.', w_c, 'A low-individual-variance weighting proxy; it is not a full covariance-based minimum-variance optimisation.')
    ]
    for tab, title, description, weight_series, note in portfolio_details:
        with tab:
            st.markdown(f'### {title}')
            st.write(description)
            met = comparison_df.loc[title.split(' — ')[0]]
            k1, k2, k3 = st.columns(3)
            k1.metric('Historical annualised return', f"{met['Annualised historical return (%)']:.2f}%")
            k2.metric('Annualised volatility', f"{met['Annualised volatility (%)']:.2f}%")
            k3.metric('Maximum drawdown', f"{met['Maximum drawdown (%)']:.2f}%")
            st.markdown('**Suggested allocation within this portfolio**')
            allocation = (weight_series * 100).rename('Allocation (%)').to_frame()
            st.dataframe(allocation.style.format('{:.2f}%'), use_container_width=True)
            st.plotly_chart(px.pie(values=weight_series.values, names=weight_series.index, hole=.55,
                                   title='Portfolio allocation'), use_container_width=True)
            st.info(note)
    st.divider()
    st.markdown('**Side-by-side comparison**')
    st.dataframe(comparison_df.style.format('{:.2f}'), use_container_width=True)
    st.plotly_chart(px.bar(comparison_df.reset_index(), x='Portfolio', y='Annualised volatility (%)',
                           title='Historical annualised volatility by portfolio'), use_container_width=True)
    st.caption('All portfolio metrics are historical estimates. Portfolio B and C are transparent weighting rules, not guaranteed optimal portfolios.')

with recommendation_tab:
    st.subheader('Your portfolio insight')
    st.caption('A concise, profile-aware reading of the historical results — not a prediction of future performance.')
    risk_target = {
        'Low': 'Prioritise lower historical volatility and smaller historical drawdowns. Avoid treating higher returns as worth any level of risk.',
        'Moderate': 'Look for a balance between historical return, volatility and diversification rather than choosing on return alone.',
        'High': 'You may be more comfortable with larger fluctuations, but high historical returns can come with substantial losses.'
    }[risk]
    lowest_risk = comparison_df['Annualised volatility (%)'].idxmin()
    smallest_drawdown = comparison_df['Maximum drawdown (%)'].idxmax()
    highest_return = comparison_df['Annualised historical return (%)'].idxmax()
    if risk == 'Low':
        primary = lowest_risk
        reason = 'It recorded the lowest annualised volatility in the selected historical period.'
    elif risk == 'Moderate':
        # A transparent balanced score: rank return high and volatility/drawdown low.
        rank_return = comparison_df['Annualised historical return (%)'].rank(ascending=False)
        rank_vol = comparison_df['Annualised volatility (%)'].rank(ascending=True)
        rank_dd = comparison_df['Maximum drawdown (%)'].rank(ascending=False)
        score = rank_return + rank_vol + rank_dd
        primary = score.idxmin()
        reason = 'It had the strongest combined ranking across historical return, volatility and maximum drawdown.'
    else:
        primary = highest_return
        reason = 'It recorded the highest annualised historical return in the selected period; this also may involve higher risk.'

    st.markdown(f'''<div class="rec-card">
    <div class="small-muted">PERSONALISED ANALYSIS FOR</div>
    <h2 style="margin:.2rem 0;">{investor if investor else 'Investor'}</h2>
    <div>Risk profile: <b>{risk}</b> &nbsp; · &nbsp; Investment horizon: <b>{horizon}</b></div>
    <hr style="opacity:.25;">
    <div class="small-muted">HISTORICAL STRATEGY TO EXPLORE FIRST</div>
    <h2 style="margin:.2rem 0;">{primary}</h2>
    <p style="margin:.2rem 0;">{reason}</p>
    </div>''', unsafe_allow_html=True)
    st.write('')
    st.write(risk_target)
    q1, q2, q3 = st.columns(3)
    q1.metric('Lowest volatility', lowest_risk)
    q2.metric('Smallest drawdown', smallest_drawdown)
    q3.metric('Highest historical return', highest_return)

    st.markdown('#### What the analysis suggests')
    st.write(f'- **{lowest_risk}** had the lowest historical annualised volatility.')
    st.write(f'- **{smallest_drawdown}** had the least severe maximum drawdown in this sample.')
    st.write(f'- **{highest_return}** had the highest annualised historical return.')
    st.write(f'- Your selected risk profile is **{risk}**. {risk_target}')
    st.warning('This is an educational decision-support result based on past data, not a promise, forecast, or personal financial advice. Past performance does not guarantee future results.')

with methodology_tab:
    st.subheader('How PortfolioIQ calculates the results')
    st.markdown('''
- **Daily return:** (Today's adjusted close ÷ Previous trading day's adjusted close) − 1.
- **Average price:** arithmetic mean of the adjusted daily closing prices in the selected period.
- **Variance:** measures how widely daily returns vary around their average.
- **Correlation:** measures how two stocks' daily returns move together.
- **SVD:** decomposes the standardised return matrix into mathematical factors and stock loadings.
- **Regression beta:** slope of selected-stock average daily returns against NIFTY 50 daily returns.
- **Portfolio A:** equal weight across selected stocks.
- **Portfolio B:** inverse individual-volatility weighting.
- **Portfolio C:** inverse individual-variance weighting (a simple low-variance proxy, not a full optimiser).
- **Annualisation:** daily mean return compounded over 252 trading days; daily standard deviation multiplied by √252.
''')
    st.caption('Data source: Yahoo Finance via yfinance. Data availability and update timing are controlled by the provider. Prices are adjusted historical closes, not guaranteed live quotes.')
