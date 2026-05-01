import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import time
from io import StringIO
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# Styling Helper
# ──────────────────────────────────────────────────────────────────────────────
PALETTE   = sns.color_palette("tab10")
STYLE     = "whitegrid"
FIG_DPI   = 130

sns.set_theme(style=STYLE, palette=PALETTE, font_scale=1.1)
plt.rcParams.update({'figure.dpi': FIG_DPI, 'axes.titleweight': 'bold'})

def divider(title=""):
    print("\n" + "=" * 65)
    if title:
        print(f"  {title}")
        print("=" * 65)

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
divider("1. DATA LOADING & PREPROCESSING")

PATH_DATASET = ('/kaggle/input/datasets/emonreza/'
                '65-years-of-weather-data-bangladesh-preprocessed/'
                '65 Years of Weather Data Bangladesh (1948 - 2013).csv')

t0 = time.time()
try:
    df = pd.read_csv(PATH_DATASET)
except FileNotFoundError:
    print(f"[ERROR] Dataset not found at:\n  {PATH_DATASET}")
    raise

print(f"  Loaded in {time.time()-t0:.2f}s  |  Shape: {df.shape}")

# -- Rename columns to standard names
df = df.rename(columns={'Station Names': 'Station',
                         'YEAR':          'Year',
                         'Rainfall':      'Monthly Total'})

# -- Build datetime index
df['Day']  = 1
df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

n_stations    = df['Station'].nunique()
year_range    = f"{df['Year'].min()} – {df['Year'].max()}"
total_records = len(df)

print(f"  Stations   : {n_stations}")
print(f"  Year range : {year_range}")
print(f"  Total rows : {total_records:,}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. EDA: ANNUAL RAINFALL TREND
# ══════════════════════════════════════════════════════════════════════════════
divider("2. EDA – ANNUAL RAINFALL TREND")

year_total = df.groupby(['Station', 'Year'])['Monthly Total'].sum().reset_index()

fig, ax = plt.subplots(figsize=(14, 6))
sns.lineplot(data=year_total, x='Year', y='Monthly Total', hue='Station',
             linewidth=1.4, alpha=0.85, ax=ax)
ax.set_title('Total Annual Rainfall by Station (1948 – 2013)', pad=12)
ax.set_ylabel('Total Annual Rainfall (mm)')
ax.set_xlabel('Year')
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), fontsize=8)
plt.tight_layout()
plt.savefig('01_Annual_Rainfall_Trend.png')
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 3. ALTITUDE FILTERING – SELECT LOW-LYING STATIONS
# ══════════════════════════════════════════════════════════════════════════════
divider("3. ALTITUDE FILTERING")

unique_stations = df[['Station', 'ALT', 'LATITUDE', 'LONGITUDE']].drop_duplicates()
low_lying = unique_stations.query('ALT < 5').sort_values('ALT')
print(low_lying[['Station', 'ALT', 'LATITUDE', 'LONGITUDE']].to_string(index=False))

# ── Station altitude overview chart
fig, ax = plt.subplots(figsize=(12, 5))
alt_all = unique_stations.sort_values('ALT')
colors  = ['#e74c3c' if a < 5 else '#3498db' for a in alt_all['ALT']]
bars = ax.barh(alt_all['Station'], alt_all['ALT'], color=colors, edgecolor='white', linewidth=0.5)
ax.axvline(5, color='red', linestyle='--', linewidth=1.5, label='5 m threshold')
ax.set_xlabel('Altitude (m)')
ax.set_title('Weather Station Altitudes  (Red = Selected for Model)', pad=10)
ax.legend()
plt.tight_layout()
plt.savefig('02_Station_Altitudes.png')
plt.show()

TARGET_STATIONS = [
    "Cox's Bazar", 'Khulna', 'Barisal', 'Hatiya',
    'Patuakhali', 'Khepupara', 'Sitakunda', 'Teknaf', 'Mongla'
]

# ══════════════════════════════════════════════════════════════════════════════
# 4. BUILD PIVOT & SELECT LOW-AREA DATE RANGE
# ══════════════════════════════════════════════════════════════════════════════
divider("4. BUILD PIVOT TABLE")

df_min = df[['Date', 'Station', 'Monthly Total']]
pivot  = pd.pivot_table(df_min, values='Monthly Total',
                        columns='Station', fill_value=0, index='Date')

TARGET_STATIONS = [s for s in TARGET_STATIONS if s in pivot.columns]
low_areas = pivot[TARGET_STATIONS].loc['1992-01-01':]

print(f"  Shape of modelling data : {low_areas.shape}")
print(f"  Date range              : {low_areas.index[0].date()} → {low_areas.index[-1].date()}")

# ── Monthly seasonal profile heatmap
monthly_mean = low_areas.copy()
monthly_mean.index = pd.to_datetime(monthly_mean.index)
monthly_mean['Month'] = monthly_mean.index.month
heatmap_data = monthly_mean.groupby('Month')[TARGET_STATIONS].mean()

fig, ax = plt.subplots(figsize=(13, 5))
sns.heatmap(heatmap_data.T, annot=True, fmt='.0f', cmap='YlOrRd',
            linewidths=0.4, linecolor='white', ax=ax,
            xticklabels=['Jan','Feb','Mar','Apr','May','Jun',
                         'Jul','Aug','Sep','Oct','Nov','Dec'])
ax.set_title('Mean Monthly Rainfall Heatmap by Station (1992 – 2013)', pad=10)
ax.set_ylabel('')
plt.tight_layout()
plt.savefig('03_Seasonal_Heatmap.png')
plt.show()

# ── Inter-station correlation heatmap
fig, ax = plt.subplots(figsize=(9, 7))
corr = low_areas.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5, linecolor='white',
            cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Station Rainfall Correlation Matrix', pad=10)
plt.tight_layout()
plt.savefig('04_Station_Correlation.png')
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAIN / TEST SPLIT & SCALING
# ══════════════════════════════════════════════════════════════════════════════
divider("5. TRAIN / TEST SPLIT")

TEST_SIZE  = 24
train      = low_areas.iloc[:-TEST_SIZE]
test       = low_areas.iloc[-TEST_SIZE:]

scaler     = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test  = scaler.transform(test)

SEQ_LEN    = 18
BATCH_SIZE = 1
N_FEATURES = len(TARGET_STATIONS)

generator            = TimeseriesGenerator(scaled_train, scaled_train,
                                           length=SEQ_LEN, batch_size=BATCH_SIZE)
validation_generator = TimeseriesGenerator(scaled_test,  scaled_test,
                                           length=SEQ_LEN, batch_size=BATCH_SIZE)

print(f"  Train samples : {len(train)}  ({train.index[0].date()} → {train.index[-1].date()})")
print(f"  Test  samples : {len(test)}  ({test.index[0].date()} → {test.index[-1].date()})")

# ══════════════════════════════════════════════════════════════════════════════
# 6. MODEL DEFINITION
# ══════════════════════════════════════════════════════════════════════════════
divider("6. MODEL DEFINITION")

model = Sequential([
    LSTM(500, activation='relu', return_sequences=True,
         input_shape=(SEQ_LEN, N_FEATURES)),
    LSTM(300, activation='relu', dropout=0.5, return_sequences=True),
    LSTM(100, activation='relu'),
    Dense(N_FEATURES)
])
model.compile(optimizer='adam', loss='mse')
model.summary()

arch_params = {
    'Architecture'   : 'Stacked LSTM (3 layers)',
    'LSTM Units'     : '500 → 300 → 100',
    'Dropout'        : '0.5 (Layer 2)',
    'Dense Output'   : str(N_FEATURES),
    'Optimizer'      : 'Adam',
    'Loss'           : 'MSE',
    'Sequence Length': str(SEQ_LEN),
    'Batch Size'     : str(BATCH_SIZE),
}

# ══════════════════════════════════════════════════════════════════════════════
# 7. MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
divider("7. MODEL TRAINING")

EPOCHS     = 10
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

t_train = time.time()
history = model.fit(generator, epochs=EPOCHS,
                    validation_data=validation_generator,
                    callbacks=[early_stop])
train_time = time.time() - t_train
epochs_run = len(history.history['loss'])

print(f"\n  Training completed in {train_time:.1f}s  |  Epochs run: {epochs_run}")

# ── Loss curve
losses = pd.DataFrame(history.history)
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(losses['loss'],     label='Train Loss', linewidth=2, color='#2ecc71')
ax.plot(losses['val_loss'], label='Val Loss',   linewidth=2, color='#e74c3c', linestyle='--')
ax.set_title('Training & Validation Loss (MSE)', pad=10)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (MSE)')
ax.legend()
plt.tight_layout()
plt.savefig('05_Training_Loss.png')
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 8. PREDICTIONS ON TEST SET
# ══════════════════════════════════════════════════════════════════════════════
divider("8. GENERATING PREDICTIONS")

test_preds, current_batch = [], scaled_train[-SEQ_LEN:].reshape(1, SEQ_LEN, N_FEATURES)
for _ in range(len(test)):
    pred = model.predict(current_batch, verbose=0)[0]
    test_preds.append(pred)
    current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)

true_predictions = pd.DataFrame(
    scaler.inverse_transform(test_preds),
    columns=test.columns, index=test.index
)

# ══════════════════════════════════════════════════════════════════════════════
# 9. PERFORMANCE METRICS  (per station + aggregate)
# ══════════════════════════════════════════════════════════════════════════════
divider("9. PERFORMANCE METRICS")

rows = []
for col in TARGET_STATIONS:
    y_true, y_pred = test[col].values, true_predictions[col].values
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mae   = mean_absolute_error(y_true, y_pred)
    r2    = r2_score(y_true, y_pred)
    mape  = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    rows.append({
        'Station'   : col,
        'RMSE (mm)' : round(rmse,  2),
        'MAE (mm)'  : round(mae,   2),
        'R²'        : round(r2,    4),
        'MAPE (%)'  : round(mape,  2),
    })

metrics_df = pd.DataFrame(rows)

# Rainy-season integrated rainfall (slots 9–14 in the 24-month test window)
metrics_df['True Season (mm)']  = [int(test.iloc[9:15][c].sum()) for c in TARGET_STATIONS]
metrics_df['Pred Season (mm)']  = [round(true_predictions.iloc[9:15][c].sum(), 1) for c in TARGET_STATIONS]
metrics_df['Season Error (mm)'] = (metrics_df['True Season (mm)']
                                   - metrics_df['Pred Season (mm)']).round(1)

print(metrics_df.to_string(index=False))
print(f"\n  Mean RMSE : {metrics_df['RMSE (mm)'].mean():.2f} mm")
print(f"  Mean MAE  : {metrics_df['MAE (mm)'].mean():.2f} mm")
print(f"  Mean R²   : {metrics_df['R²'].mean():.4f}")
print(f"  Mean MAPE : {metrics_df['MAPE (%)'].mean():.2f} %")

# ══════════════════════════════════════════════════════════════════════════════
# 10. PAPER-STYLE SUMMARY TABLE (printed + figure)
# ══════════════════════════════════════════════════════════════════════════════
divider("10. PAPER-STYLE SUMMARY METRICS FIGURE")

dataset_meta = {
    'Source Dataset'         : '65 Years of Weather Data Bangladesh',
    'Data Provider'          : 'Bangladesh Meteorological Department',
    'Temporal Coverage'      : year_range,
    'Modelling Period'       : '1992 – 2013',
    'No. of Stations'        : str(n_stations),
    'Stations Used (model)'  : str(N_FEATURES),
    'Station Criterion'      : 'Altitude < 5 m a.s.l.',
    'Total Records'          : f'{total_records:,}',
}

summary_rows  = list(dataset_meta.items())
arch_rows     = list(arch_params.items())
perf_rows     = [
    ('Mean RMSE', f"{metrics_df['RMSE (mm)'].mean():.2f} mm"),
    ('Mean MAE',  f"{metrics_df['MAE (mm)'].mean():.2f} mm"),
    ('Mean R²',   f"{metrics_df['R²'].mean():.4f}"),
    ('Mean MAPE', f"{metrics_df['MAPE (%)'].mean():.2f} %"),
    ('Epochs Run', str(epochs_run)),
    ('Training Time', f"{train_time:.1f} s"),
]

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#1a1a2e')

def make_table(ax, rows, title, header_color='#e94560', row_colors=None):
    ax.axis('off')
    ax.set_facecolor('#16213e')
    col_labels = ['Parameter', 'Value']
    if row_colors is None:
        row_colors = [['#0f3460', '#16213e'][i % 2] for i in range(len(rows))]
    tab = ax.table(
        cellText=rows, colLabels=col_labels,
        cellLoc='left', loc='center',
        bbox=[0, 0, 1, 1]
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(9.5)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor('#2a2a4a')
        cell.set_text_props(color='white')
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_facecolor(row_colors[r - 1])
    ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=8)

gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.06)
ax1, ax2, ax3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
make_table(ax1, summary_rows, '📂  Dataset Overview',    header_color='#e94560')
make_table(ax2, arch_rows,    '🧠  Model Architecture',  header_color='#533483')
make_table(ax3, perf_rows,    '📊  Performance Summary', header_color='#0f9b8e')

fig.suptitle('Bangladesh Flood Prediction — LSTM Model Report',
             color='white', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('06_Paper_Metrics_Table.png', bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 11. QUALITATIVE: PREDICTED vs ACTUAL  (side-by-side + overlay per station)
# ══════════════════════════════════════════════════════════════════════════════
divider("11. QUALITATIVE ANALYSIS PLOTS")

# ── Side-by-side overview
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
for col in TARGET_STATIONS:
    ax1.plot(true_predictions[col], label=col)
    ax2.plot(test[col],             label=col)
ax1.set_title('LSTM Predicted Monthly Rainfall')
ax2.set_title('Actual Monthly Rainfall')
for ax in (ax1, ax2):
    ax.set_ylabel('Rainfall (mm)')
    ax.set_xlabel('Date')
    ax.tick_params(axis='x', rotation=30)
handles, labels = ax2.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=8.5,
           bbox_to_anchor=(0.5, -0.01))
plt.tight_layout()
plt.savefig('07_Predicted_vs_Actual_Overview.png', bbox_inches='tight')
plt.show()

# ── Per-station actual vs predicted overlays (3×3 grid)
fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex=True)
axes = axes.flatten()
for idx, col in enumerate(TARGET_STATIONS):
    ax = axes[idx]
    ax.plot(test.index,             test[col].values,             'b-',  lw=1.8, label='Actual')
    ax.plot(true_predictions.index, true_predictions[col].values, 'r--', lw=1.8, label='Predicted')
    ax.fill_between(test.index,
                    test[col].values, true_predictions[col].values,
                    alpha=0.18, color='purple')
    ax.set_title(col, fontsize=10)
    ax.set_ylabel('mm')
    ax.tick_params(axis='x', rotation=30)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=10,
           bbox_to_anchor=(0.5, 1.01))
fig.suptitle('Per-Station Actual vs Predicted Monthly Rainfall', y=1.04, fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('08_Per_Station_Overlay.png', bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 12. QUANTITATIVE METRICS CHARTS
# ══════════════════════════════════════════════════════════════════════════════
divider("12. METRIC BAR CHARTS")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

for ax, metric, color in zip(
        axes.flatten(),
        ['RMSE (mm)', 'MAE (mm)', 'R²', 'MAPE (%)'],
        ['#e74c3c',   '#f39c12',  '#2ecc71', '#3498db']):
    bars = ax.bar(metrics_df['Station'], metrics_df[metric],
                  color=color, edgecolor='white', linewidth=0.6, alpha=0.88)
    ax.set_title(metric, fontweight='bold')
    ax.set_ylabel(metric)
    ax.tick_params(axis='x', rotation=35)
    for bar in bars:
        ax.annotate(f'{bar.get_height():.1f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8)

plt.suptitle('Per-Station Performance Metrics', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('09_Metric_Barcharts.png', bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 13. RAINY SEASON TOTAL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
divider("13. RAINY SEASON TOTAL COMPARISON")

x  = np.arange(len(TARGET_STATIONS))
w  = 0.38
fig, ax = plt.subplots(figsize=(13, 6))
b1 = ax.bar(x - w/2, metrics_df['True Season (mm)'],  w, label='Actual',    color='#3498db', alpha=0.88)
b2 = ax.bar(x + w/2, metrics_df['Pred Season (mm)'],  w, label='Predicted', color='#e74c3c', alpha=0.88)
ax.set_xticks(x); ax.set_xticklabels(TARGET_STATIONS, rotation=30, ha='right')
ax.set_ylabel('Total Rainfall (mm)')
ax.set_title('Rainy Season (Apr–Sep) Total Rainfall: Actual vs Predicted', fontweight='bold')
ax.legend()
for bar in list(b1) + list(b2):
    ax.annotate(f'{bar.get_height():.0f}',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords='offset points',
                ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig('10_Rainy_Season_Comparison.png', bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 14. RESIDUAL / ERROR ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
divider("14. RESIDUAL & ERROR ANALYSIS")

residuals_df = (test - true_predictions)

# ── Residual time-series
fig, ax = plt.subplots(figsize=(14, 5))
for col in TARGET_STATIONS:
    ax.plot(residuals_df.index, residuals_df[col], label=col, alpha=0.75, lw=1.3)
ax.axhline(0, color='black', lw=1.5, linestyle='--')
ax.fill_between(residuals_df.index, -200, 200, alpha=0.05, color='gray',
                label='±200 mm band')
ax.set_title('Prediction Residuals Over Time  (Actual − Predicted)', fontweight='bold')
ax.set_ylabel('Residual (mm)')
ax.legend(loc='upper right', fontsize=8, ncol=3)
ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig('11_Residuals_Timeseries.png', bbox_inches='tight')
plt.show()

# ── Residual distribution (kde + histogram)
fig, axes = plt.subplots(3, 3, figsize=(16, 11))
axes = axes.flatten()
for idx, col in enumerate(TARGET_STATIONS):
    axes[idx].hist(residuals_df[col], bins=12, color='#9b59b6',
                   edgecolor='white', alpha=0.8, density=True)
    residuals_df[col].plot.kde(ax=axes[idx], color='#e74c3c', linewidth=2)
    axes[idx].axvline(0, color='black', linestyle='--', lw=1.5)
    axes[idx].set_title(col, fontsize=10)
    axes[idx].set_xlabel('Residual (mm)')
fig.suptitle('Residual Distributions per Station', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('12_Residual_Distributions.png', bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 15. FLOOD RISK THRESHOLD EXCEEDANCE ANALYSIS  (unique to time-series floods)
# ══════════════════════════════════════════════════════════════════════════════
divider("15. FLOOD RISK THRESHOLD ANALYSIS")

# Define per-station flood thresholds (using 90th percentile of historical training data)
thresholds = {col: np.percentile(low_areas[col], 90) for col in TARGET_STATIONS}

fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex=False)
axes = axes.flatten()
for idx, col in enumerate(TARGET_STATIONS):
    ax  = axes[idx]
    thr = thresholds[col]
    ax.plot(test.index, test[col], 'b-', lw=1.6, label='Actual')
    ax.plot(true_predictions.index, true_predictions[col], 'r--', lw=1.6, label='Predicted')
    ax.axhline(thr, color='orange', linestyle=':', lw=2, label=f'Flood threshold ({thr:.0f} mm)')
    # Shade actual flood months
    actual_flood = test[col] > thr
    ax.fill_between(test.index, 0, test[col].max(),
                    where=actual_flood, color='blue', alpha=0.12, label='Actual flood month')
    pred_flood = true_predictions[col] > thr
    ax.fill_between(true_predictions.index, 0, test[col].max(),
                    where=pred_flood, color='red', alpha=0.10, label='Predicted flood month')
    ax.set_title(col, fontsize=10)
    ax.set_ylabel('mm')
    ax.tick_params(axis='x', rotation=30, labelsize=7)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=9,
           bbox_to_anchor=(0.5, 1.02))
fig.suptitle('Flood Risk Threshold Exceedance  (90th Percentile)', y=1.05,
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('13_Flood_Threshold_Analysis.png', bbox_inches='tight')
plt.show()

# -- Flood hit/miss summary
print("\n  Flood Event Detection Summary  (actual vs predicted, 90th pct threshold):")
print(f"  {'Station':<22} {'Actual Floods':>13} {'Pred Floods':>11} {'Correct Hits':>12}")
print("  " + "-" * 60)
for col in TARGET_STATIONS:
    act = (test[col] > thresholds[col]).sum()
    pre = (true_predictions[col] > thresholds[col]).sum()
    hit = ((test[col] > thresholds[col]) & (true_predictions[col] > thresholds[col])).sum()
    print(f"  {col:<22} {act:>13} {pre:>11} {hit:>12}")

# ══════════════════════════════════════════════════════════════════════════════
# 16. FORECAST 24 MONTHS BEYOND DATASET
# ══════════════════════════════════════════════════════════════════════════════
divider("16. FORECASTING 24 MONTHS AHEAD")

full_scaler = MinMaxScaler()
full_scaler.fit(low_areas)
scaled_full = full_scaler.transform(low_areas)

gen_full = TimeseriesGenerator(scaled_full, scaled_full, length=SEQ_LEN, batch_size=BATCH_SIZE)

model_full = Sequential([
    LSTM(500, activation='relu', return_sequences=True, input_shape=(SEQ_LEN, N_FEATURES)),
    LSTM(300, activation='relu', dropout=0.5, return_sequences=True),
    LSTM(100, activation='relu'),
    Dense(N_FEATURES)
])
model_full.compile(optimizer='adam', loss='mse')

early_stop_full = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
model_full.fit(gen_full, epochs=10, callbacks=[early_stop_full], verbose=1)

forecast, cb_full = [], scaled_full[-SEQ_LEN:].reshape(1, SEQ_LEN, N_FEATURES)
for _ in range(24):
    p = model_full.predict(cb_full, verbose=0)[0]
    forecast.append(p)
    cb_full = np.append(cb_full[:, 1:, :], [[p]], axis=1)

forecast_df = pd.DataFrame(
    full_scaler.inverse_transform(forecast),
    columns=TARGET_STATIONS,
    index=pd.date_range(
        start=low_areas.index[-1] + pd.DateOffset(months=1),
        periods=24, freq='MS'
    )
)

fig, ax = plt.subplots(figsize=(13, 6))
for col in TARGET_STATIONS:
    ax.plot(forecast_df.index, forecast_df[col], label=col, lw=1.8)
ax.set_title('24-Month Rainfall Forecast — Low-Lying Bangladesh Stations', fontweight='bold')
ax.set_ylabel('Predicted Rainfall (mm)')
ax.set_xlabel('Date')
ax.legend(loc='upper right', fontsize=8, ncol=3)
ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig('14_Forecast_24Months.png', bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 17. FINAL PAPER-STYLE PER-STATION METRICS TABLE
# ══════════════════════════════════════════════════════════════════════════════
divider("17. FINAL PER-STATION METRICS TABLE")

fig, ax = plt.subplots(figsize=(15, 4))
ax.axis('off')
col_labels = list(metrics_df.columns)
cell_text  = metrics_df.values.tolist()
row_colors = [['#dff0d8' if i % 2 == 0 else '#f9f9f9'] * len(col_labels)
              for i in range(len(cell_text))]
tab = ax.table(cellText=cell_text, colLabels=col_labels,
               cellLoc='center', loc='center',
               rowColours=['#f8f8f8']*len(cell_text))
tab.auto_set_font_size(False)
tab.set_fontsize(9.5)
tab.auto_set_column_width(col=list(range(len(col_labels))))
for (r, c), cell in tab.get_celld().items():
    if r == 0:
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', weight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#ecf0f1')
    cell.set_edgecolor('#bdc3c7')

ax.set_title('Per-Station Model Performance Metrics',
             fontsize=12, fontweight='bold', pad=14)
plt.tight_layout()
plt.savefig('15_Per_Station_Metrics_Table.png', bbox_inches='tight')
plt.show()

divider("ALL DONE")
print("  All 15 output figures saved successfully.")
print(f"  Final Mean RMSE : {metrics_df['RMSE (mm)'].mean():.2f} mm")
print(f"  Final Mean R²   : {metrics_df['R²'].mean():.4f}")
