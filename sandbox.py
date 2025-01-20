import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import ipywidgets as widgets
from IPython.display import display

# Seitentitel setzen
st.title('Meine erste Streamlit App')

# Sidebar mit Eingabemöglichkeiten
st.sidebar.header('Einstellungen')
name = st.sidebar.text_input('Wie heißt du?', 'Max')
alter = st.sidebar.slider('Wähle dein Alter:', 0, 100, 25)

# Hauptbereich
st.write(f'Hallo {name}, willkommen in deiner ersten Streamlit App!')

# Tabs erstellen
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(['Daten', 'Chart', 'Über', 'Bruh', 'what', 'ehm', 'ok'])

# Erstelle einen DataFrame mit zufälligen Daten
df = pd.DataFrame({
    'Datum': pd.date_range('2024-01-01', periods=10),
    'Werte': np.random.randn(10)
})

df_target = pd.read_csv("lucas_organic_carbon_target.csv")
df_test = pd.read_csv("lucas_organic_carbon_training_and_test_data.csv")

if len(df_test) > 200:
    df_test_small = df_test.iloc[::len(df_test)//200]  # Reduziere auf 200 Punkte
else:
    df_test_small = df_test

if len(df_target) > 200:
    df_target_small = df_target.iloc[::len(df_target)//200]  # Reduziere auf 200 Punkte
else:
    df_target_small = df_target

def func1():
    st.header('Zufällige Daten')
    st.dataframe(df_target)
    st.dataframe(df)

def func2():
    st.header('Liniendiagramm')
    # Erstelle ein Liniendiagramm
    st.line_chart(df.set_index('Datum'))    

def func3():
    st.header('Über diese App')
    st.write("""
    Dies ist eine einfache Demo-App erstellt mit Streamlit.
    Sie zeigt einige grundlegende Funktionen wie:
    - Sidebar-Eingaben
    - Tabs
    - Dataframe-Anzeige
    - Diagramme
    """)

def func4():
    st.header("Bruh")
    fig, ax = plt.subplots()
    ax.bar(df['Datum'].dt.strftime('%Y-%m-%d'), df['Werte'])
    ax.set_xlabel('Datum')
    ax.set_ylabel('Werte')
    ax.set_title('Barplot der zufälligen Werte')
    plt.xticks(rotation=45)
    st.pyplot(fig)

def analyze_feature_importance(df_features, target_series, num_features=10):
    """
    Parameters:
    df_features: DataFrame mit Features
    target_series: Series mit Zielvariable
    num_features: Anzahl der top Features die angezeigt werden sollen
    """
    
    # Container für die Analyse
    with st.container():
        st.header("Random Forest Feature Importance Analyse")
        
        # Fortschrittsbalken
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Daten aufteilen
        status_text.text("Teile Daten auf...")
        X_train, X_test, y_train, y_test = train_test_split(
            df_features, target_series, 
            test_size=0.3, 
            random_state=42
        )
        progress_bar.progress(25)
        
        # Model Training
        status_text.text("Trainiere Random Forest...")
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        progress_bar.progress(50)
        
        # Feature Importance berechnen
        status_text.text("Berechne Feature Importance...")
        importances = model.feature_importances_
        feature_names = df_features.columns
        sorted_idx = importances.argsort()[::-1][:num_features]
        
        top_features = feature_names[sorted_idx]
        top_importances = importances[sorted_idx]
        progress_bar.progress(75)
        
        # Visualisierung mit Plotly
        status_text.text("Erstelle Visualisierung...")
        fig = go.Figure(go.Bar(
            x=top_importances,
            y=top_features,
            orientation='h',
            marker_color='skyblue'
        ))
        
        fig.update_layout(
            title="Top {} Feature Importances".format(num_features),
            xaxis_title="Relative Importance",
            yaxis_title="Features",
            height=500,
            yaxis={'autorange': 'reversed'},  # Wichtigste Features oben
            margin=dict(l=20, r=20, t=40, b=20)
        )
        progress_bar.progress(100)
        status_text.text("Analyse abgeschlossen!")
        
        # Plot anzeigen
        st.plotly_chart(fig, use_container_width=True)
        
        # Metriken anzeigen
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Train Score", 
                f"{model.score(X_train, y_train):.2%}"
            )
        with col2:
            st.metric(
                "Test Score", 
                f"{model.score(X_test, y_test):.2%}"
            )
        
        # Feature Importance als Tabelle
        st.subheader("Feature Importance Details")
        importance_df = pd.DataFrame({
            'Feature': top_features,
            'Importance': top_importances
        })
        st.dataframe(importance_df)
        
        return model, importance_df

def analyze_spectral_intervals(df_test, df_target):
    """
    Parameters:
    df_test: DataFrame mit den spektralen Daten
    df_target: DataFrame mit den Zielkategorien
    """
    st.header("Spektrale Intervallanalyse")
    
    # Sidebar für Kontrollelemente
    st.sidebar.header("Analyseparameter")
    
    # Wellenlängenbereich auswählen
    wavelengths = [float(col) for col in df_test.columns]
    min_wavelength = min(wavelengths)
    max_wavelength = max(wavelengths)
    
    wavelength_range = st.sidebar.slider(
        "Wellenlängenbereich (nm)",
        min_value=float(min_wavelength),
        max_value=float(max_wavelength),
        value=(500.0, 550.0),
        step=0.1
    )
    
    num_intervals = st.sidebar.slider(
        "Anzahl der Intervalle",
        min_value=2,
        max_value=20,
        value=5
    )
    
    # Container für die Visualisierung
    with st.container():
        # Spalten im gewählten Wellenlängenbereich auswählen
        selected_columns = [col for col in df_test.columns 
                          if wavelength_range[0] <= float(col) <= wavelength_range[1]]
        
        if not selected_columns:
            st.warning("Keine Daten im ausgewählten Wellenlängenbereich gefunden.")
            return
        
        # Mittelwert für jede Zeile im ausgewählten Bereich
        mean_values = df_test[selected_columns].mean(axis=1)
        
        # Erste Wellenlänge im Bereich für x-Achse
        x_values = df_test[selected_columns[0]]
        
        # Plotly Figure erstellen
        fig = go.Figure()
        
        # Scatter Plots für jede Kategorie
        colors = {'very_low': 'blue', 'low': 'green', 'moderate': 'red'}
        
        for category in colors:
            mask = df_target['x'] == category
            fig.add_trace(go.Scatter(
                x=x_values[mask],
                y=mean_values[mask],
                mode='markers',
                name=category,
                marker=dict(color=colors[category], size=8, opacity=0.6)
            ))
        
        # Intervalle erstellen und visualisieren
        x_min, x_max = x_values.min(), x_values.max()
        interval_width = (x_max - x_min) / num_intervals
        
        for i in range(num_intervals):
            start = x_min + i * interval_width
            end = x_min + (i + 1) * interval_width
            
            # Vertikale Linien für Intervallgrenzen
            fig.add_vline(x=start, line_dash="dash", line_color="gray", opacity=0.3)
            
            # Mittelwertlinien für jedes Intervall
            mask = (x_values >= start) & (x_values <= end)
            if any(mask):
                y_mean = mean_values[mask].mean()
                fig.add_shape(
                    type="line",
                    x0=start,
                    x1=end,
                    y0=y_mean,
                    y1=y_mean,
                    line=dict(color="black", width=1, dash="solid")
                )
        
        # Letzte vertikale Linie
        fig.add_vline(x=x_max, line_dash="dash", line_color="gray", opacity=0.3)
        
        # Layout anpassen
        fig.update_layout(
            title=f"Spektrale Analyse ({wavelength_range[0]:.1f}-{wavelength_range[1]:.1f} nm)",
            xaxis_title="Wellenlänge (nm)",
            yaxis_title="Mittlere Intensität",
            showlegend=True,
            hovermode='closest',
            height=600,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        # Plot anzeigen
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistiken anzeigen
        st.subheader("Statistiken für den ausgewählten Bereich")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Durchschnittliche Intensität",
                f"{mean_values.mean():.2f}"
            )
        with col2:
            st.metric(
                "Standardabweichung",
                f"{mean_values.std():.2f}"
            )
        with col3:
            st.metric(
                "Anzahl Datenpunkte",
                len(mean_values)
            )
        
        # Intervall-Details
        st.subheader("Intervall-Details")
        interval_stats = []
        for i in range(num_intervals):
            start = x_min + i * interval_width
            end = x_min + (i + 1) * interval_width
            mask = (x_values >= start) & (x_values <= end)
            interval_stats.append({
                'Intervall': f"{start:.1f}-{end:.1f}",
                'Mittlere Intensität': mean_values[mask].mean(),
                'Anzahl Punkte': sum(mask)
            })
        
        st.dataframe(pd.DataFrame(interval_stats))

with tab1:
    func1()

with tab2:
    func2()

with tab3:
    func3()

with tab4:
    func4()

with tab5:
    analyze_feature_importance(df_test_small, df_target_small['x'], num_features=10)

with tab6:
    analyze_spectral_intervals(df_test_small, df_target_small)

with tab7:
    st.write("Ok")

# Footer
st.markdown('---')
st.markdown('Erstellt mit ❤️ und Streamlit')