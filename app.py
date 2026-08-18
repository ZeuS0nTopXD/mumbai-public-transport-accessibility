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


def generate_dataset_chart(df_train):
    plt.figure(figsize=(10, 4.8), dpi=200)
    line_counts = df_train['Line'].value_counts()
    colors_list = [
        '#3b82f6',
        '#10b981',
        '#f59e0b',
        '#ef4444',
        '#8b5cf6',
        '#06b6d4',
        '#ec4899',
        '#84cc16',
        '#6366f1',
    ]
    bars = plt.bar(
        line_counts.index,
        line_counts.values,
        color=colors_list[: len(line_counts)],
        width=0.5,
        edgecolor='#1e293b',
        linewidth=0.7,
        alpha=0.9,
    )

    plt.title(
        'Station Distribution Across Railway Lines',
        fontsize=13,
        fontweight='bold',
        pad=15,
        color='#0f172a',
    )
    plt.ylabel(
        'Number of Stations', fontsize=11, fontweight='600', color='#334155'
    )
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.xticks(
        fontsize=9, fontweight='500', color='#334155', rotation=25, ha='right'
    )
    plt.yticks(fontsize=10, color='#334155')

    for bar in bars:
        height = bar.get_height()
        plt.annotate(
            f'{height}',
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
    df_ward = pd.read_csv('ward_level_collated.csv')
    df_train = pd.read_csv('Mumbai Local Train Dataset.csv', encoding='latin1')

    total_pop = int(df_ward['TOT_P'].sum())
    total_stations = len(df_train)
    total_working_pop = int(df_ward['TOT_WORK_P'].sum()) if 'TOT_WORK_P' in df_ward.columns else 0

    station_to_ward = {
        'Churchgate': 'A',
        'Marine Lines': 'A',
        'Charni Road': 'D',
        'Grant Road': 'D',
        'Mumbai Central': 'E',
        'Mahalakshmi': 'G/S',
        'Lower Parel': 'G/S',
        'Prabhadevi': 'G/S',
        'Dadar': 'G/N',
        'Matunga Road': 'G/N',
        'Mahim Jn': 'G/N',
        'Bandra': 'H/W',
        'Khar Road': 'H/W',
        'Santacruz': 'H/W',
        'Vile Parle': 'K/W',
        'Andheri': 'K/W',
        'Jogeshwari': 'K/W',
        'Ram Mandir': 'K/W',
        'Goregaon': 'P/N',
        'Malad': 'P/N',
        'Kandivli': 'R/S',
        'Borivali': 'R/S',
        'Dahisar': 'R/N',
        'Mira Road': 'R/N',
        'CSMT': 'A',
        'Masjid': 'B',
        'Sandhurst Road': 'B',
        'Byculla': 'E',
        'Chinchpokli': 'E',
        'Currey Road': 'G/S',
        'Parel': 'G/S',
        'Matunga': 'F/N',
        'Sion': 'F/N',
        'Kurla': 'L',
        'Vidhyavihar': 'N',
        'Ghatkopar': 'N',
        'Vikhroli': 'N',
        'Kanjurmarg': 'S',
        'Bhandup': 'S',
        'Nahur': 'S',
        'Mulund': 'T',
        'Reay Road': 'E',
        'Cotton Green': 'E',
        'Sewri': 'E',
        'Wadala Road': 'F/N',
        'GTB Nagar': 'F/N',
        'Chunabhatti': 'L',
        'Chembur': 'M/W',
        'Govandi': 'M/W',
        'Mankhurd': 'M/W',
    }

    df_train['Ward_Alphabet'] = (
        df_train['Station'].map(station_to_ward).fillna('K/E')
    )
    station_agg = (
        df_train.groupby('Ward_Alphabet')
        .agg(
            Station_Count=('Station', 'count'),
            Avg_Platforms=('Platforms', 'mean'),
            Total_Passengers=('Number of Passengers ', 'count'),
        )
        .reset_index()
    )

    merged_df = pd.merge(df_ward, station_agg, on='Ward_Alphabet', how='left')
    merged_df['Station_Count'] = merged_df['Station_Count'].fillna(1)
    merged_df['Avg_Platforms'] = merged_df['Avg_Platforms'].fillna(2.0)

    merged_df['Commercial_Density'] = (
        merged_df['MAIN_OT_P'] / merged_df['Land_Area']
    )
    merged_df['Avg_Distance_to_Station'] = merged_df['Land_Area'] / (
        merged_df['Station_Count'] * 2.5
    )
    merged_df['Accessibility_Score'] = (
        merged_df['Station_Count'] * 15000.0
    ) / (merged_df['TOT_P_DEN'] + 100) + (
        merged_df['P_LIT'] / merged_df['TOT_P'] * 10
    )

    X = merged_df[
        [
            'TOT_P',
            'TOT_M',
            'TOT_F',
            'Station_Count',
            'Avg_Distance_to_Station',
            'Commercial_Density',
        ]
    ]
    y = merged_df['Accessibility_Score']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

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

        rmse = round(np.sqrt(mean_squared_error(y_test, preds)), 4)
        r2 = round(r2_score(y_test, preds), 4)
        results[name] = {'R2 Score': r2}
        model_metrics.append({'name': name, 'rmse': rmse, 'r2': r2})

    results_df = pd.DataFrame(results).T
    chart_base64 = generate_performance_chart(results_df)
    chart2_base64 = generate_dataset_chart(df_train)

    # Sample ward rows for dataset preview table
    sample_wards = (
        merged_df[['Ward_Alphabet', 'TOT_P', 'Land_Area', 'TOT_P_DEN', 'Station_Count']]
        .head(5)
        .to_dict(orient='records')
    )

    return (
        model_metrics,
        len(merged_df),
        total_pop,
        total_stations,
        total_working_pop,
        chart_base64,
        chart2_base64,
        sample_wards,
    )


@app.route('/')
def home():
    (
        metrics,
        total_wards,
        total_pop,
        total_stations,
        total_working_pop,
        chart,
        chart2,
        sample_wards,
    ) = run_model_pipeline()
    return render_template(
        'index.html',
        metrics=metrics,
        total_wards=total_wards,
        total_pop=total_pop,
        total_stations=total_stations,
        total_working_pop=total_working_pop,
        chart=chart,
        chart2=chart2,
        sample_wards=sample_wards,
    )


if __name__ == '__main__':
    app.run(debug=True)