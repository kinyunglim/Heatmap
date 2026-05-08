import pandas as pd
import yfinance as yf
import requests
from io import StringIO
from datetime import datetime
import warnings

# --- CONFIG: Suppress Warnings ---
# 忽略 Pandas 的版本建議警告，保持 GitHub Actions Console 乾淨
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message=".*SettingWithCopyWarning.*")

def generate_clean_sp500_heatmap(top_n=50):
    # --- STEP 1: Fetch S&P 500 List ---
    print("1. Fetching S&P 500 list from Wikipedia...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers)
        sp500_df = pd.read_html(StringIO(response.text))[0]
        # 處理 Ticker 符號，例如 BRK.B 改為 BRK-B 以配合 yfinance
        sp500_df['Symbol'] = sp500_df['Symbol'].str.replace('.', '-', regex=False)
        info_map = sp500_df.set_index('Symbol')[['Security', 'GICS Sector']]
    except Exception as e:
        print(f"Error fetching S&P 500 list: {e}")
        return

    # --- STEP 2: Download Data ---
    current_year = datetime.now().year
    start_date = f"{current_year}-01-01"
    tickers = sp500_df['Symbol'].tolist()

    print(f"2. Downloading data for {len(tickers)} companies...")
    # 使用 threads=True 提高 GitHub Actions 的下載速度
    data = yf.download(tickers, start=start_date, progress=False, auto_adjust=True, threads=True)

    # 處理 yfinance 可能出現的多層 Index (MultiIndex)
    if isinstance(data.columns, pd.MultiIndex):
        if 'Close' in data.columns.get_level_values(0):
            price_data = data.xs('Close', axis=1, level=0)
        else:
            price_data = data.iloc[:, data.columns.get_level_values(0) == 'Close']
    else:
        price_data = data['Close'] if 'Close' in data.columns else data

    price_data = price_data.dropna(axis=1, how='all')

    # --- STEP 3: Calculate YTD & Filter ---
    initial_prices = price_data.iloc[0]
    current_prices = price_data.iloc[-1]
    ytd_pct = ((current_prices - initial_prices) / initial_prices) * 100

    # 排序並取得表現最強的 Top N
    top_series = ytd_pct.sort_values(ascending=False).head(top_n)
    top_tickers = top_series.index.tolist()

    # --- STEP 4: Process Daily Returns ---
    subset_prices = price_data[top_tickers]
    daily_pct = subset_prices.pct_change() * 100
    daily_pct = daily_pct.iloc[1:][top_tickers]

    # --- STEP 5: Format Table Structure ---
    df_heatmap = daily_pct.T
    df_heatmap = df_heatmap.iloc[:, ::-1]  # 讓最新的日期排在最左邊

    # Reset Index 準備合併資料
    df_heatmap.index.name = 'Ticker'
    df_heatmap = df_heatmap.reset_index()

    # 建立日期 Header
    date_cols = df_heatmap.columns[1:]
    dates_iso = [d.strftime('%m-%d') if isinstance(d, datetime) else str(d) for d in date_cols]

    header_tuples = [('Info', 'Ticker')]
    for i, date_str in enumerate(dates_iso):
        label = "T (Latest)" if i == 0 else f"T-{i}"
        header_tuples.append((label, date_str))

    df_heatmap.columns = pd.MultiIndex.from_tuples(header_tuples)

    # 插入公司名稱、行業及 YTD 數據
    tickers_list = df_heatmap[('Info', 'Ticker')].tolist()
    sectors = info_map.loc[tickers_list, 'GICS Sector'].values
    names = info_map.loc[tickers_list, 'Security'].values
    ytd_values = top_series[tickers_list].values

    df_heatmap.insert(1, ('Info', 'Name'), names)
    df_heatmap.insert(2, ('Info', 'Sector'), sectors)
    df_heatmap.insert(3, ('Performance', 'YTD %'), ytd_values)

    # --- STEP 6: "Institutional" Dark Mode Styling ---
    print("3. Generating Styled HTML...")

    daily_cols_subset = df_heatmap.columns[4:]
    ytd_col_subset = [df_heatmap.columns[3]]
    info_cols_subset = df_heatmap.columns[:3]

    # 專業配色邏輯
    def pro_color_map(val):
        if pd.isna(val): return 'background-color: #020617; color: #4b5563'
        # POSITIVE (Green)
        if val > 3.0: return 'background-color: #047857; color: #ffffff; font-weight: bold'
        if val > 1.0: return 'background-color: #065f46; color: #d1fae5'
        if val > 0.0: return 'background-color: #064e3b; color: #6ee7b7'
        # NEGATIVE (Red)
        if val < -3.0: return 'background-color: #b91c1c; color: #ffffff; font-weight: bold'
        if val < -1.0: return 'background-color: #991b1b; color: #fecaca'
        if val < 0.0:  return 'background-color: #7f1d1d; color: #fca5a5'
        # Flat
        return 'background-color: #020617; color: #9ca3af'

    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    display_time = datetime.now().strftime('%Y-%m-%d %H:%M')

    # 使用 .map() 代替已廢棄的 .applymap()
    styler = df_heatmap.style \
        .format("{:.2f}%", subset=daily_cols_subset) \
        .format("{:.2f}%", subset=ytd_col_subset) \
        .map(pro_color_map, subset=daily_cols_subset) \
        .map(lambda x: 'background-color: #020617; color: #fbbf24; font-weight: bold;', subset=ytd_col_subset) \
        .map(lambda x: 'background-color: #020617;', subset=info_cols_subset) \
        .set_properties(**{
            'color': '#e2e8f0',
            'border': '1px solid #334155',
            'font-family': 'Roboto Mono, Consolas, monospace',
            'font-size': '12px',
            'padding': '8px',
            'text-align': 'center'
        }) \
        .set_table_styles([
            {'selector': 'thead th', 'props': [
                ('background-color', '#0f172a'),
                ('color', '#94a3b8'),
                ('font-weight', 'bold'),
                ('border-bottom', '2px solid #475569'),
                ('text-align', 'center'),
                ('padding', '12px'),
                ('text-transform', 'uppercase'),
                ('letter-spacing', '1px')
            ]},
            {'selector': 'td.col0', 'props': [
                ('font-weight', 'bold'),
                ('color', '#38bdf8'),
                ('background-color', '#020617'),
                ('text-align', 'left'),
                ('border-right', '1px solid #334155')
            ]},
            {'selector': 'td.col1', 'props': [
                ('text-align', 'left'),
                ('color', '#f1f5f9'),
                ('background-color', '#020617')
            ]},
            {'selector': 'td.col2', 'props': [
                ('text-align', 'left'),
                ('color', '#94a3b8'),
                ('font-style', 'italic')
            ]},
            {'selector': 'td.col3', 'props': [
                ('border-left', '1px solid #475569'),
                ('border-right', '1px solid #475569')
            ]},
        ]) \
        .set_caption(f"<div style='color: #64748b; margin-bottom: 10px; font-family: sans-serif;'>S&P 500 Top {top_n} Leaders (YTD) • Generated: {display_time} (HKT)</div>")

    output_filename = f"sp500_clean_heatmap_{timestamp_str}.html"
    styler.to_html(output_filename)
    print(f"Done! Saved to {output_filename}")


if __name__ == "__main__":
    generate_clean_sp500_heatmap(50)




