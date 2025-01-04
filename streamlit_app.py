import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import confusion_matrix
from joblib import load

st.set_page_config(page_title="Lucas Dataset", page_icon="🎈")
st.title("🎈Semesterprojekt Gruppe E")
st.write(
    "Dies ist das Semesterprojekt der Gruppe E."
)

# Load the data from a CSV. We're caching this so it doesn't reload every time the app
# reruns (e.g. if the user interacts with the widgets).
@st.cache_data
def load_data_df():
    df=pd.read_csv('/home/nicob/OneDrive/documents/Obsidian Vault/02 - Areas/Education/Informatik Master (HU Berlin)/HU Berlin/Module/Semester-1/Visual Analytics/Aufgaben/Erste Aufgabe Exploration/Datensatz Lucas Organic Carbon für Classifikationstask/lucas_organic_carbon/training_test/lucas_organic_carbon_training_and_test_data.csv')
    return df

def load_data_df_target():
    df_target=pd.read_csv('/home/nicob/OneDrive/documents/Obsidian Vault/02 - Areas/Education/Informatik Master (HU Berlin)/HU Berlin/Module/Semester-1/Visual Analytics/Aufgaben/Erste Aufgabe Exploration/Datensatz Lucas Organic Carbon für Classifikationstask/lucas_organic_carbon/target/lucas_organic_carbon_target.csv')
    return df_target

df = load_data_df()
df_target = load_data_df_target()

st.sidebar.header('Target Classes Distribution')

plt.figure(figsize=(8, 6))
df_target['x'].value_counts().plot(kind='bar',color=sns.palettes.mpl_palette('Dark2'))
plt.title('Verteilung der Zielklassen')
plt.xlabel('Zielklasse')
plt.ylabel('Anzahl')

st.pyplot(plt.gcf())

st.sidebar.header('Random Forest Classifier')



X = df
y = df_target['x']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, criterion="gini", max_features=5, bootstrap=True, oob_score=True, random_state=42)

rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)

oob_score = rf_classifier.oob_score_
st.write("OOB Score:", oob_score)

accuracy = accuracy_score(y_test, y_pred)
st.write("Accuracy:", accuracy)
st.text(classification_report(y_test, y_pred))

st.sidebar.subheader('Confusion Matrix Parameters')

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=[f'Class {i}' for i in range(5)],
            yticklabels=[f'Class {i}' for i in range(5)])
plt.title('Confusion Matrix for 5-Class Classification')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
st.pyplot(fig)

top_n = 20
feature_importance = rf_classifier.feature_importances_
sorted_idx = np.argsort(feature_importance)[-top_n:]

fi_fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(top_n), feature_importance[sorted_idx], align='center', color='skyblue')
ax.set_yticks(range(top_n))
ax.set_yticklabels([X.columns[i] for i in sorted_idx])
ax.set_xlabel('Feature Importance')
ax.set_title(f'Top {top_n} Feature Importance')

st.pyplot(fi_fig)