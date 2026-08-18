import base64
import io
import matplotlib
import numpy as np
import pandas as pd
from flask import Flask, render_template
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)


def generate_performance_chart(results_df):
    plt.figure(figsize=(9, 4.2), dpi=200)
    models = results_df.index.tolist()
    r2_scores = results_df['R2 Score'].tolist()

    colors = ['#2563eb' if v >= 0 else '#ef4444' for v in r2_scores]
    bars = plt.bar(
        models,
        r2_scores,
        color=colors,
        width=0.45,
        edgecolor='#1e293b',
        linewidth=0.7,
        alpha=0.9,
    )

    plt.axhline(0, color='#1e293b', linewidth=1, linestyle='--')
    plt.title(
        'Algorithm Performance Benchmark ($R^2$ Score)',
        fontsize=13,
        fontweight='bold',
        pad=15,
        color='#0f172a',
    )
    plt.ylabel(
        '$R^2$ Score', fontsize=11, fontweight='600', color='#334155'
    )
    plt.ylim(-0.5, 1.1)
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.xticks(fontsize=10, fontweight='500', color='#334155')
    plt.yticks(fontsize=10, color='#334155')

    for bar in bars:
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        plt.annotate(
            f'{height:.2f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4 if height >= 0 else -14),
            textcoords='offset points',
            ha='center',
            va=va,
            fontsize=9,
            fontweight='bold',
            color='#1e293b',
        )

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', transparent=True)
    buffer.seek(0)
    plot_url = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    return plot_url


def generate_modal_split_chart(total_train, total_metro, total_monorail, total_bus):
    plt.figure(figsize=(9, 4.2), dpi=200)
    modes = ['Local Train', 'Metro', 'Monorail', 'Bus Stops']
    counts = [total_train, total_metro, total_monorail, total_bus]
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']

    bars = plt.bar(
        modes,
        counts,
        color=colors,
        width=0.45,
        edgecolor='#1e293b',
        linewidth=0.7,
        alpha=0.9,
    )

    plt.title(
        'Transit Infrastructure Distribution by Mode',
        fontsize=13,
        fontweight='bold',
        pad=15,
        color='#0f172a',
    )
    plt.ylabel(
        'Count / Stops', fontsize=11, fontweight='600', color='#334155'
    )
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.xticks(fontsize=10, fontweight='500', color='#334155')
    plt.yticks(fontsize=10, color='#334155')

    for bar in bars:
        height = bar.get_height()
        plt.annotate(
            f'{height:,}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold',
            color='#1e293b',
        )

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', transparent=True)
    buffer.seek(0)
    chart2_url = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    return chart2_url


def run_model_pipeline():
    # 1. Load All Datasets
    df_ward = pd.read_csv('ward_level_collated.csv')
    df_train = pd.read_csv('Mumbai Local Train Dataset.csv', encoding='latin1')
    
    try:
        df_metro = pd.read_csv('Mumbai Metro Stations Dataset.csv', encoding='latin1')
    except Exception:
        df_metro = pd.DataFrame()

    try:
        df_monorail = pd.read_csv('Mumbai Monorail Stations Dataset.csv', encoding='latin1')
    except Exception:
        df_monorail = pd.DataFrame()

    try:
        df_bus = pd.read_csv('BEST Bus Stops.csv', encoding='latin1')
    except Exception:
        df_bus = pd.DataFrame()

    total_pop = int(df_ward['TOT_P'].sum()) if 'TOT_P' in df_ward.columns else 0
    total_working_pop = int(df_ward['TOT_WORK_P'].sum()) if 'TOT_WORK_P' in df_ward.columns else 0
    total_train = len(df_train)
    total_metro = len(df_metro) if not df_metro.empty else 0
    total_monorail = len(df_monorail) if not df_monorail.empty else 0
    total_bus = len(df_bus) if not df_bus.empty else 0

    df_ward.columns = df_ward.columns.str.strip()

    station_to_ward = {
        'Churchgate': 'A', 'Marine Lines': 'A', 'Charni Road': 'D', 'Grant Road': 'D',
        'Mumbai Central': 'E', 'Mahalakshmi': 'G/S', 'Lower Parel': 'G/S', 'Prabhadevi': 'G/S',
        'Dadar': 'G/N', 'Matunga Road': 'G/N', 'Mahim Jn': 'G/N', 'Bandra': 'H/W',
        'Khar Road': 'H/W', 'Santacruz': 'H/W', 'Vile Parle': 'K/W', 'Andheri': 'K/W',
        'Jogeshwari': 'K/W', 'Ram Mandir': 'K/W', 'Goregaon': 'P/N', 'Malad': 'P/N',
        'Kandivli': 'R/S', 'Borivali': 'R/S', 'Dahisar': 'R/N', 'Mira Road': 'R/N',
        'CSMT': 'A', 'Masjid': 'B', 'Sandhurst Road': 'B', 'Byculla': 'E',
        'Chinchpokli': 'E', 'Currey Road': 'G/S', 'Parel': 'G/S', 'Matunga': 'F/N',
        'Sion': 'F/N', 'Kurla': 'L', 'Vidhyavihar': 'N', 'Ghatkopar': 'N',
        'Vikhroli': 'N', 'Kanjurmarg': 'S', 'Bhandup': 'S', 'Nahur': 'S',
        'Mulund': 'T', 'Reay Road': 'E', 'Cotton Green': 'E', 'Sewri': 'E',
        'Wadala Road': 'F/N', 'GTB Nagar': 'F/N', 'Chunabhatti': 'L', 'Chembur': 'M/W',
        'Govandi': 'M/W', 'Mankhurd': 'M/W'
    }

    df_train['Ward_Alphabet'] = df_train['Station'].map(station_to_ward).fillna('K/E')
    
    station_agg = (
        df_train.groupby('Ward_Alphabet')
        .agg(
            Train_Stations=('Station', 'count'),
            Avg_Platforms=('Platforms', 'mean'),
        )
        .reset_index()
    )

    merged_df = pd.merge(df_ward, station_agg, on='Ward_Alphabet', how='left')
    merged_df['Train_Stations'] = merged_df['Train_Stations'].fillna(1)
    merged_df['Avg_Platforms'] = merged_df['Avg_Platforms'].fillna(2.0)

    if 'MAIN_OT_P' in merged_df.columns and 'Land_Area' in merged_df.columns:
        merged_df['Commercial_Density'] = merged_df['MAIN_OT_P'] / (merged_df['Land_Area'] + 1e-5)
    else:
        merged_df['Commercial_Density'] = 0.5

    # PBL Core Calculation: Public Transport Accessibility Score based on Population vs Station Density
    if 'TOT_P_DEN' in merged_df.columns and 'TOT_P' in merged_df.columns:
        merged_df['Accessibility_Score'] = (
            (merged_df['Train_Stations'] * 15000.0) / (merged_df['TOT_P_DEN'] + 100)
        )
    else:
        merged_df['Accessibility_Score'] = np.random.uniform(5, 50, len(merged_df))

    # Sort wards by accessibility score ascending to highlight underserved areas for priority connectivity
    underserved_df = merged_df.sort_values(by='Accessibility_Score', ascending=True).head(5)

    feature_cols = [c for c in ['TOT_P', 'TOT_M', 'TOT_F', 'Train_Stations', 'Avg_Platforms', 'Commercial_Density'] if c in merged_df.columns]
    X = merged_df[feature_cols]
    y = merged_df['Accessibility_Score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Linear Reg': LinearRegression(),
        'Ridge Reg': Ridge(alpha=1.0),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boost': GradientBoostingRegressor(random_state=42),
    }

    results = {}
    model_metrics = []
    for name, model in models.items():
        if name in ['Linear Reg', 'Ridge Reg']:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        rmse = round(float(np.sqrt(mean_squared_error(y_test, preds))), 4)
        r2 = round(float(r2_score(y_test, preds)), 4)
        results[name] = {'R2 Score': r2}
        model_metrics.append({'name': name, 'rmse': rmse, 'r2': r2})

    results_df = pd.DataFrame(results).T
    chart_base64 = generate_performance_chart(results_df)
    chart2_base64 = generate_modal_split_chart(total_train, total_metro, total_monorail, total_bus)

    priority_wards = underserved_df.to_dict(orient='records')

    return (
        model_metrics,
        len(merged_df),
        total_pop,
        total_train,
        total_metro,
        total_monorail,
        total_bus,
        total_working_pop,
        chart_base64,
        chart2_base64,
        priority_wards,
    )


@app.route('/')
def home():
    (
        metrics,
        total_wards,
        total_pop,
        total_train,
        total_metro,
        total_monorail,
        total_bus,
        total_working_pop,
        chart,
        chart2,
        priority_wards,
    ) = run_model_pipeline()
    return render_template(
        'index.html',
        metrics=metrics,
        total_wards=total_wards,
        total_pop=total_pop,
        total_train=total_train,
        total_metro=total_metro,
        total_monorail=total_monorail,
        total_bus=total_bus,
        total_working_pop=total_working_pop,
        chart=chart,
        chart2=chart2,
        priority_wards=priority_wards,
    )


if __name__ == '__main__':
    app.run(debug=True)
