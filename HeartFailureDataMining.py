import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


# 1. Veri Yükleme
df = pd.read_csv("heart.csv")
print(df.columns)

# Cinsiyet dağılımı - Kalp hastalığına göre
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Sex', hue='HeartDisease')
plt.xlabel("Cinsiyet")
plt.ylabel("Kalp Yetmezliği")
plt.savefig('cinsiyet_dagilimi.png')
plt.close()

sns.boxplot(x='HeartDisease', y='Age', data=df)
plt.title("Kalp Yetmezliğine Göre Yaş Dağılımı")
plt.xlabel("Kalp Yetmezliği")
plt.ylabel("Yaş")
plt.show()


# Göğüs ağrısı tipine göre kalp hastalığı dağılımı
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='ChestPainType', hue='HeartDisease')
plt.title('Göğüs Ağrısı Tipine Göre Kalp Hastalığı Dağılımı')
plt.savefig('gogus_agrisi_dagilimi.png')
plt.close()


# 2. Sütunları düzenleme
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# 3. Kategorik sütunları one-hot encoding (get_dummies)
df = pd.get_dummies(df, drop_first=True)

# 3. Giriş ve çıkış verilerini ayır
X = df.drop("heartdisease", axis=1)
y = df["heartdisease"]

# 5. Standardizasyon
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 1. KBest ile özellik seçimi
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=5)
X_kbest = selector.fit_transform(X_scaled, y)
selected_features_kbest = X.columns[selector.get_support()]
print("K-Best'e Göre Seçilen Özellikler:", selected_features_kbest.tolist())

# 2. RandomForest özellik önemine göre seçim
model_fs = RandomForestClassifier(random_state=42)
model_fs.fit(X_scaled, y)

feature_importances = pd.Series(model_fs.feature_importances_, index=X.columns)
selected_features_rf = feature_importances.sort_values(ascending=False).head(5).index
print("\nRandom Forest'a Göre Seçilen En Önemli Özellikler:", selected_features_rf.tolist())

# KBest seçilen özelliklerle modeli eğitelim
X_selected_kbest = X[selected_features_kbest]
X_selected_kbest_scaled = scaler.fit_transform(X_selected_kbest)

# Random Forest seçilen özelliklerle modeli eğitelim
X_selected_rf = X[selected_features_rf]
X_selected_rf_scaled = scaler.fit_transform(X_selected_rf)

# --- Model 1: Tüm özelliklerle ---
print("\n--- Model 1: Tüm Özelliklerle ---")
# [1] Eğitim = Test
print("\n[1] Eğitim = Test (Overfitting riski yüksek)")
model_all_full = RandomForestClassifier(random_state=42)
model_all_full.fit(X_scaled, y)
y_pred_all_full = model_all_full.predict(X_scaled)
print("Confusion Matrix:\n", confusion_matrix(y, y_pred_all_full))
print("\nAccuracy:", accuracy_score(y, y_pred_all_full))
print("\nClassification Report:\n", classification_report(y, y_pred_all_full))

# [2] %70 Eğitim - %30 Test
print("\n[2] %70 Eğitim - %30 Test")
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
model_split_full = RandomForestClassifier(random_state=42)
model_split_full.fit(X_train_full, y_train_full)
y_pred_split_full = model_split_full.predict(X_test_full)
print("Confusion Matrix:\n", confusion_matrix(y_test_full, y_pred_split_full))
print("\nAccuracy:", accuracy_score(y_test_full, y_pred_split_full))
print("\nClassification Report:\n", classification_report(y_test_full, y_pred_split_full))

# --- Model 2: KBest seçilen özelliklerle ---
print("\n--- Model 2: KBest Seçilen Özelliklerle ---")
# [1] Eğitim = Test
print("\n[1] Eğitim = Test (Overfitting riski yüksek)")
model_all_kbest = RandomForestClassifier(random_state=42)
model_all_kbest.fit(X_selected_kbest_scaled, y)
y_pred_all_kbest = model_all_kbest.predict(X_selected_kbest_scaled)
print("Confusion Matrix:\n", confusion_matrix(y, y_pred_all_kbest))
print("\nAccuracy:", accuracy_score(y, y_pred_all_kbest))
print("\nClassification Report:\n", classification_report(y, y_pred_all_kbest))

# [2] %70 Eğitim - %30 Test
print("\n[2] %70 Eğitim - %30 Test")
X_train_kbest, X_test_kbest, y_train_kbest, y_test_kbest = train_test_split(X_selected_kbest_scaled, y, test_size=0.3, random_state=42)
model_split_kbest = RandomForestClassifier(random_state=42)
model_split_kbest.fit(X_train_kbest, y_train_kbest)
y_pred_split_kbest = model_split_kbest.predict(X_test_kbest)
print("Confusion Matrix:\n", confusion_matrix(y_test_kbest, y_pred_split_kbest))
print("\nAccuracy:", accuracy_score(y_test_kbest, y_pred_split_kbest))
print("\nClassification Report:\n", classification_report(y_test_kbest, y_pred_split_kbest))

# --- Model 3: RandomForest seçilen özelliklerle ---
print("\n--- Model 3: RandomForest Seçilen Özelliklerle ---")
# [1] Eğitim = Test
print("\n[1] Eğitim = Test (Overfitting riski yüksek)")
model_all_rf = RandomForestClassifier(random_state=42)
model_all_rf.fit(X_selected_rf_scaled, y)
y_pred_all_rf = model_all_rf.predict(X_selected_rf_scaled)
print("Confusion Matrix:\n", confusion_matrix(y, y_pred_all_rf))
print("\nAccuracy:", accuracy_score(y, y_pred_all_rf))
print("\nClassification Report:\n", classification_report(y, y_pred_all_rf))

# [2] %70 Eğitim - %30 Test
print("\n[2] %70 Eğitim - %30 Test")
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_selected_rf_scaled, y, test_size=0.3, random_state=42)
model_split_rf = RandomForestClassifier(random_state=42)
model_split_rf.fit(X_train_rf, y_train_rf)
y_pred_split_rf = model_split_rf.predict(X_test_rf)
print("Confusion Matrix:\n", confusion_matrix(y_test_rf, y_pred_split_rf))
print("\nAccuracy:", accuracy_score(y_test_rf, y_pred_split_rf))
print("\nClassification Report:\n", classification_report(y_test_rf, y_pred_split_rf))


# --- [3] 10-Fold Çapraz Doğrulama ---
print("\n--- [3] 10-Fold Çapraz Doğrulama ---")
model_cv = RandomForestClassifier(random_state=42)
cv_scores = cross_val_score(model_cv, X_scaled, y, cv=10, scoring='accuracy')

print("10-fold CV Accuracy Mean: %0.4f" % cv_scores.mean())
print("Standard Deviation     : %0.4f" % cv_scores.std())

sns.countplot(x='heartdisease', data=df)
plt.title("Kalp Yetmezliği Sonuç Dağılımı")
plt.show()

# Korelasyon ısı haritası
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Değişkenler Arası Korelasyon Matrisi")
plt.show()





