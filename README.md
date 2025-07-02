# 🧬 Makine Öğrenmesi ile Meme Kanseri Teşhisi Karar Destek Sistemi

Bu proje, **Baykar Teknolojileri Bitirme Programı** kapsamında, Baykar altında bitirme projesi olarak geliştirilmiş olup, meme kanseri teşhisinde farklı makine öğrenmesi modellerinin başarımlarını karşılaştırarak en iyi modeli belirlemeyi ve bu modeli kullanıcı dostu bir web arayüzü ile sunmayı amaçlamaktadır.

---

## 🎯 Projenin Amacı

- Meme kanserinin iyi huylu (benign) ya da kötü huylu (malignant) olup olmadığını hücre özelliklerine göre tahmin etmek
- Farklı makine öğrenmesi modellerini (Lojistik Regresyon, Random Forest, SVM, KNN, vb.) karşılaştırmak
- En yüksek başarıyı gösteren modeli **Streamlit** ile geliştirilen bir arayüze entegre ederek kullanıcıların kullanımına sunmak

---

## 📁 Kullanılan Veri Seti

- **Veri Seti**: Breast Cancer Wisconsin (Diagnostic)  
- **Toplam Örnek Sayısı**: 569  
- **Özellik Sayısı**: 30 (hücre çekirdeği ölçümleri)  
- **Hedef Değişken**:  
  - `M` (Malignant - Kötü huylu) → 1  
  - `B` (Benign - İyi huylu) → 0  

---

## ⚙️ Uygulanan Adımlar

### 1. Veri Ön İşleme
- Gereksiz sütunlar (`id`, `Unnamed: 32`) çıkarıldı
- Kategorik hedef değişken (`diagnosis`) sayısal değerlere dönüştürüldü
- Özellikler `StandardScaler` ile normalize edildi
- Eğitim/Test verisi %80/%20 oranında bölündü

### 2. Kullanılan Makine Öğrenmesi Modelleri
- Lojistik Regresyon
- Karar Ağaçları
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Yapay Sinir Ağı (MLPClassifier)
- AdaBoost, GradientBoosting, Naive Bayes, LDA vb.

### 3. Model Optimizasyonu
- `GridSearchCV` ile hiperparametre optimizasyonu yapıldı
- 5 katlı çapraz doğrulama (`k-fold CV`) uygulandı

### 4. Performans Değerlendirme Metrikleri
- Doğruluk (Accuracy)
- Kesinlik (Precision)
- Duyarlılık (Recall)
- F1 Skoru
- ROC-AUC Skoru
- Confusion Matrix

---

## 🌐 Streamlit Web Uygulaması

Uygulama özellikleri:

- Gerçek zamanlı tahmin sistemi
- Farklı modelleri karşılaştırma imkânı
- Model doğruluk, ROC-AUC ve F1 skorlarını gösteren grafikler
- Radar Chart ile çok boyutlu özellik görselleştirmesi
- Eğitim amaçlı uyarı ve yasal bilgilendirme

---

## 📦 Kurulum

Projenin çalışabilmesi için aşağıdaki paketlerin yüklü olması gerekir:

```bash
pip install scikit-learn

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

pip install pandas
pip install joblib
pip install numpy
pip install plotly
import streamlit as st

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import sys
import json
import os
import random

pip install matplotlib  # İsteğe bağlı, bazı modellerde grafik gerekebilir
pip install seaborn  # Eğer ek görselleştirme istiyorsanız
```

Uygulamayı başlatmak için:
```bash
streamlit run app/app.py
```

## 📊 Örnek Model Performansları

| Model                | Doğruluk (Accuracy) |
|----------------------|---------------------|
| 🎯 Random Forest        | 0.97                |
| 🚀 Gradient Boosting    | 0.96                |
| 💡 SVM (Support Vector Machine) | 0.95        |
| 🧮 Logistic Regression  | 0.94                |
| 🔍 K-Nearest Neighbors  | 0.91                |
| 🧠 MLPClassifier (Yapay Sinir Ağı) | 0.90      |
| 📦 Naive Bayes          | 0.89                |
| 🧭 LDA (Linear Discriminant Analysis) | 0.88   |

> Not: Performans değerleri eğitim/test bölünmesi sonrası elde edilen doğruluk oranlarıdır. F1, Recall ve ROC-AUC gibi diğer metrikler uygulama içinden detaylı incelenebilir.

---

## ⚠️ Uyarı ve Yasal Bilgilendirme

> 🚨 **Bu uygulama yalnızca eğitim, araştırma ve demonstrasyon amacıyla geliştirilmiştir.**  
> Sunulan tahminler, makine öğrenmesi algoritmaları tarafından üretilen **istatistiksel tahminlerdir** ve **kesin tıbbi teşhis veya tedavi yerine geçemez**.  
> Sağlık durumunuzla ilgili her türlü karar için mutlaka **bir sağlık profesyoneline danışınız**.  
> Geliştirici, bu yazılımın kullanımından doğabilecek sonuçlardan **sorumlu tutulamaz**.

---

## 📚 Referanslar

- Mangasarian, O. L., Street, W. N., & Wolberg, W. H. (1995). *Breast cancer diagnosis and prognosis via linear programming.*
- Polat, K., & Güneş, S. (2007). *Breast cancer diagnosis using least square support vector machine.*
- [Breast Cancer Wisconsin (Diagnostic) Dataset – UCI](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- [Streamlit Resmi Web Sitesi](https://streamlit.io/)
- [Alejandro AO - Streamlit Kanseri Tahmin Uygulaması Videosu](https://www.youtube.com/watch?v=Fz5wUuSjeG4)

---

## ✍️ Geliştirici

Bu proje, **Baykar Teknolojileri** eğitimi kapsamında  
🧑‍💻 Mehmet Emin Palabıyık  
tarafından hazırlanmıştır.  

🔗 [LinkedIn Profili](https://www.linkedin.com/in/emin-35bnry72698265/)

---
