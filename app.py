import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

st.title("🔢 Digit Recognition App")

# model yükle
model = joblib.load("digit_model.pkl")

# veri yükle
df = pd.read_csv("train.csv")

st.subheader("Dataset Preview")
st.write(df.head())

# örnek görüntü göster
st.subheader("Sample Digit")

X = df.drop(columns=["label"])
y = df["label"]

img = X.iloc[0].values.reshape(28, 28)

fig, ax = plt.subplots()
ax.imshow(img, cmap="gray")
ax.axis("off")

st.pyplot(fig)

# tahmin demo
sample = X.iloc[[0]]
pred = model.predict(sample)

st.write("Prediction:", pred[0])

st.markdown("---")
st.caption("Digit Recognition ML Demo • Kaggle Project")
