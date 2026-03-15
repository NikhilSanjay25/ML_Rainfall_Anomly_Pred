import pandas as pd
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl

# ── Global settings ────────────────────────────────────────────────────────────
mpl.rcParams['font.family'] = 'Times New Roman'

DISTRICT     = "Baksa"
TRAIN_CUTOFF = 2022          # years <= this go into train
TEST_START   = pd.to_datetime("2023-04-01")
TEST_END     = pd.to_datetime("2024-10-01")
ANOMALY_PCT  = 95            # percentile threshold for Mahalanobis distance
PLOT_MONTHS  = list(range(3, 12))   # Mar–Nov

FEATURES = [
    'Avg_rainfall', 'Avg_smlvl_at15cm', 'Annual_Temp', 'Temp_Seasonal',
    'Rainfall_lag_1d', 'Rainfall_lag_3d', 'Rainfall_lag_7d',
    'Rainfall_7d_sum', 'Rainfall_7d_avg', 'SoilMoisture_7d_avg',
    'day_of_year', 'week_of_year', 'month_sin', 'month_cos',
]

YEAR_COLORS = {2018: 'darkgreen', 2019: 'purple', 2020: 'orange',
               2021: 'brown',     2022: 'teal'}

# ── 1. Load & filter ───────────────────────────────────────────────────────────
df = pd.read_csv(
    r"dataset/final_dataset_with_features.csv",
    parse_dates=["Date"],
)
df = df[df['District'] == DISTRICT].copy()
df.sort_values('Date', inplace=True)
df['Month'] = df['Date'].dt.month
df.dropna(subset=FEATURES, inplace=True)

# ── 2. Missing-data table (before any row-dropping) ────────────────────────────
full_index  = pd.date_range(TEST_START, TEST_END, freq='D')
test_window = df[df['Date'].between(TEST_START, TEST_END)].copy()
test_dates  = set(test_window['Date'])

# Truly absent rows (date not in dataset at all)
absent_dates = pd.DataFrame({
    'Date': [d for d in full_index if d not in test_dates]
})
absent_dates['YearMonth'] = absent_dates['Date'].dt.to_period('M')

# Zero-rainfall rows (present but recorded as 0)
zero_rain = test_window[test_window['Avg_rainfall'] == 0.0][['Date']].copy()
zero_rain['YearMonth'] = zero_rain['Date'].dt.to_period('M')

absent_summary = (absent_dates.groupby('YearMonth').size()
                              .reset_index(name='Absent_Rows'))
zero_summary   = (zero_rain.groupby('YearMonth').size()
                            .reset_index(name='Zero_Rainfall_Days'))

all_periods = pd.DataFrame(
    {'YearMonth': pd.period_range(TEST_START, TEST_END, freq='M')}
)
missing_summary = (all_periods
                   .merge(absent_summary, on='YearMonth', how='left')
                   .merge(zero_summary,   on='YearMonth', how='left')
                   .fillna(0)
                   .astype({'Absent_Rows': int, 'Zero_Rainfall_Days': int}))
missing_summary['Total_Bad_Days'] = (missing_summary['Absent_Rows']
                                     + missing_summary['Zero_Rainfall_Days'])

print("\n── Data quality per month (test window) ──────────────────────────────")
print(missing_summary.to_string(index=False))
print(f"\nTotal absent rows      : {missing_summary['Absent_Rows'].sum()}")
print(f"Total zero-rain rows   : {missing_summary['Zero_Rainfall_Days'].sum()}")
print(f"Combined bad days      : {missing_summary['Total_Bad_Days'].sum()}\n")

# ── 3. Train / test split ──────────────────────────────────────────────────────
train_df = df[df['Year'] <= TRAIN_CUTOFF].copy()
test_df  = df[df['Date'].between(TEST_START, TEST_END)].copy()

X_train = train_df[FEATURES].values
X_test  = test_df[FEATURES].values

# ── 4. Mahalanobis anomaly detection ──────────────────────────────────────────
mean_vec       = X_train.mean(axis=0)
cov_matrix     = np.cov(X_train, rowvar=False)
inv_cov        = np.linalg.pinv(cov_matrix)   # pinv is safer than inv

def mahal(row):
    return distance.mahalanobis(row, mean_vec, inv_cov)

test_df['Mahalanobis_Distance'] = [mahal(x) for x in X_test]

train_distances = [mahal(x) for x in X_train]
threshold = np.percentile(train_distances, ANOMALY_PCT)

test_df['Predicted_Anomaly'] = test_df['Mahalanobis_Distance'] > threshold
anomalies = test_df[test_df['Predicted_Anomaly']].copy()

# ── 5. Historical monthly averages (2018–2022) ─────────────────────────────────
hist_df = df[df['Year'].between(2018, TRAIN_CUTOFF)]

monthly_avg      = hist_df.groupby('Month')['Avg_rainfall'].mean()
monthly_by_year  = (
    hist_df.groupby(['Year', 'Month'])['Avg_rainfall']
           .mean()
           .unstack(level=0)          # columns = years
           .loc[PLOT_MONTHS]          # keep only Mar–Nov rows
)

# ── 6. Build reference scatter points (15th of each plot-month, 2023 & 2024) ──
def reference_dates(months, years):
    return pd.to_datetime(
        [f"{y}-{m:02d}-15" for y in years for m in months]
    )

ref_dates = reference_dates(PLOT_MONTHS, [2023, 2024])

# Tile monthly_by_year rows to match [2023-months … 2024-months]
ref_monthly = pd.concat([monthly_by_year] * 2, ignore_index=True)
ref_monthly.index = ref_dates
ref_monthly = ref_monthly[(ref_monthly.index >= TEST_START) &
                           (ref_monthly.index <= TEST_END)]

avg_ref = pd.DataFrame({'Date': ref_dates,
                        'Month': PLOT_MONTHS * 2,
                        'Monthly_Avg': (PLOT_MONTHS * 2)})
avg_ref['Monthly_Avg'] = avg_ref['Month'].map(monthly_avg)
avg_ref = avg_ref[(avg_ref['Date'] >= TEST_START) &
                  (avg_ref['Date'] <= TEST_END)]

# ── 7. Build contiguous bad-data runs for shading ────────────────────────────
all_days = pd.DataFrame({'Date': full_index})
all_days['absent'] = all_days['Date'].isin(set(absent_dates['Date']))
all_days['zero']   = all_days['Date'].isin(set(zero_rain['Date']))

def contiguous_runs(flag_series, dates):
    """Return list of (start, end) pairs for contiguous True runs."""
    runs, in_run, start = [], False, None
    for d, v in zip(dates, flag_series):
        if v and not in_run:
            in_run, start = True, d
        elif not v and in_run:
            runs.append((start, d))
            in_run = False
    if in_run:
        runs.append((start, dates.iloc[-1]))
    return runs

absent_runs = contiguous_runs(all_days['absent'], all_days['Date'])
zero_runs   = contiguous_runs(all_days['zero'],   all_days['Date'])

# ── 8. Plot ───────────────────────────────────────────────────────────────────
fig, (ax, ax2) = plt.subplots(
    2, 1, figsize=(14, 8),
    gridspec_kw={'height_ratios': [4, 1]},
    sharex=True
)

# ── Main panel ────────────────────────────────────────────────────────────────
for i, (start, end) in enumerate(absent_runs):
    ax.axvspan(start, end, color='lightgrey', alpha=0.5,
               label='Missing Data' if i == 0 else '_nolegend_')

for i, (start, end) in enumerate(zero_runs):
    ax.axvspan(start, end, color='khaki', alpha=0.5,
               label='Zero-Recorded Days' if i == 0 else '_nolegend_')

ax.plot(test_df['Date'], test_df['Avg_rainfall'],
        label='Observed Rainfall (2023–24)', alpha=0.9, zorder=3)

ax.scatter(anomalies['Date'], anomalies['Avg_rainfall'],
           color='red', marker='x', zorder=5, s=60, label='Anomaly')

ax.scatter(avg_ref['Date'], avg_ref['Monthly_Avg'],
           color='blue', marker='o', s=35, zorder=4,
           label='Monthly Avg (2018–22)')

for year, color in YEAR_COLORS.items():
    if year in ref_monthly.columns:
        ax.scatter(ref_monthly.index, ref_monthly[year],
                   color=color, marker='o', s=35, zorder=4,
                   label=f'{year} Monthly Avg')

ax.set_title(f"Rainfall Anomalies in {DISTRICT} (Apr 2023 – Oct 2024)")
ax.set_ylabel("Avg Rainfall (mm)")
ax.legend(loc='upper right', fontsize=7, ncol=2)

# ── Data quality bar panel ────────────────────────────────────────────────────
bar_dates = [pd.to_datetime(str(p)) for p in missing_summary['YearMonth']]
bar_width = 20

ax2.bar(bar_dates, missing_summary['Absent_Rows'],
        width=bar_width, color='lightgrey', edgecolor='grey',
        label='Absent rows', align='center')
ax2.bar(bar_dates, missing_summary['Zero_Rainfall_Days'],
        width=bar_width, color='khaki', edgecolor='goldenrod',
        bottom=missing_summary['Absent_Rows'],
        label='Zero-recorded', align='center')

ax2.set_ylabel("Bad days / month", fontsize=8)
ax2.set_ylim(0, 35)
ax2.axhline(28, color='red', linewidth=0.8, linestyle='--', alpha=0.6,
            label='Full month threshold')
ax2.legend(fontsize=7, loc='upper right')
ax2.set_xlabel("Date")

# ── Shared x-axis formatting ──────────────────────────────────────────────────
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()