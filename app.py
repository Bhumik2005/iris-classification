import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Page config
st.set_page_config(
    page_title="Iris Classification",
    page_icon="🌸",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0f0f1a; }
    .stApp { background-color: #0f0f1a; color: #f0f0f5; }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    h1, h2, h3 { color: #a78bfa !important; }
</style>
""", unsafe_allow_html=True)

# Load and cache data
@st.cache_data
def load_data():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["species"] = pd.Categorical.from_codes(data.target, data.target_names)
    df["target"] = data.target
    return df, data.target_names

# Train and cache models
@st.cache_resource
def train_models():
    data = load_iris()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM":                 SVC(kernel="rbf", probability=True),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        results[name] = {
            "model":    model,
            "accuracy": acc,
            "preds":    preds,
            "y_test":   y_test,
        }

    return results

df, target_names = load_data()
results = train_models()

# Header
st.markdown("# 🌸 Iris Flower Classification")
st.markdown("**ML model comparing Logistic Regression, Random Forest and SVM on the classic Iris dataset.**")
st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Model Results", "🔍 Predict Species", "📈 Data Explorer"])

# ── Tab 1: Model Results ──
with tab1:
    st.markdown("### Model Comparison")

    cols = st.columns(3)
    for i, (name, res) in enumerate(results.items()):
        with cols[i]:
            st.metric(label=name, value=f"{res['accuracy']*100:.1f}%", delta="accuracy")

    st.markdown("---")

    # Best model
    best_name = max(results, key=lambda x: results[x]["accuracy"])
    st.success(f"🏆 Best Model: **{best_name}** with **{results[best_name]['accuracy']*100:.1f}% accuracy**")

    st.markdown("---")

    # Confusion matrices
    st.markdown("### Confusion Matrices")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor("#0f0f1a")

    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(res["y_test"], res["preds"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                    xticklabels=target_names, yticklabels=target_names, ax=ax)
        ax.set_title(name, color="#a78bfa", fontsize=11)
        ax.set_facecolor("#0f0f1a")
        ax.tick_params(colors="white", labelsize=8)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### Classification Report")
    selected = st.selectbox("Select model", list(results.keys()))
    report = classification_report(
        results[selected]["y_test"],
        results[selected]["preds"],
        target_names=target_names,
        output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)

# ── Tab 2: Predict ──
with tab2:
    st.markdown("### Predict Iris Species")
    st.markdown("Adjust the sliders to input flower measurements and get a prediction.")

    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
        sepal_width  = st.slider("Sepal Width (cm)",  2.0, 4.5, 3.0, 0.1)
    with col2:
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
        petal_width  = st.slider("Petal Width (cm)",  0.1, 2.5, 1.2, 0.1)

    model_choice = st.selectbox("Choose model for prediction", list(results.keys()))

    if st.button("🌸 Predict Species", use_container_width=True):
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        model = results[model_choice]["model"]
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]
        species = target_names[prediction]

        emoji = {"setosa": "🌺", "versicolor": "🌷", "virginica": "🌸"}
        st.markdown(f"## {emoji.get(species, '🌸')} Predicted Species: **{species.capitalize()}**")

        st.markdown("### Confidence")
        for i, name in enumerate(target_names):
            st.progress(float(proba[i]), text=f"{name}: {proba[i]*100:.1f}%")

# ── Tab 3: Data Explorer ──
with tab3:
    st.markdown("### Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", len(df))
    col2.metric("Features", "4")
    col3.metric("Classes", "3")

    st.markdown("### Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### Feature Distribution")
    feature = st.selectbox("Select feature", df.columns[:-2])
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    fig2.patch.set_facecolor("#0f0f1a")
    ax2.set_facecolor("#0f0f1a")
    for species in df["species"].unique():
        subset = df[df["species"] == species][feature]
        ax2.hist(subset, alpha=0.6, label=species, bins=15)
    ax2.legend()
    ax2.tick_params(colors="white")
    ax2.set_xlabel(feature, color="white")
    ax2.set_ylabel("Count", color="white")
    ax2.set_title(f"{feature} Distribution by Species", color="#a78bfa")
    st.pyplot(fig2)

    st.markdown("### Pairplot")
    st.markdown("*Relationship between all features colored by species*")
    fig3 = sns.pairplot(df.drop("target", axis=1), hue="species", palette="husl")
    fig3.fig.patch.set_facecolor("#0f0f1a")
    st.pyplot(fig3)

# Footer
st.markdown("---")
st.markdown("Built by **Bhumik Kumta** · [GitHub](https://github.com/Bhumik2005) · [Portfolio](https://portfolio-5zuhn6y87-bhumikkumta-2843s-projects.vercel.app)")