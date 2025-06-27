import os
import json
import joblib
import pandas as pd

# Scaler'ı yükleme
def load_scaler():
    return joblib.load('models/trained_models/scaler.joblib')

# Modeli yükleme
def load_model(model_key):
    file_path = f'models/trained_models/{model_key}.joblib'

    # Dosya var mı kontrol et
    if not os.path.exists(file_path):
        # Alternatif dosya adları dene, sadece boşluk kaldırılmış hali de ekleniyor
        alternative_paths = [
            f'models/trained_models/{model_key.lower()}.joblib', # küçük harfli hali
            f'models/trained_models/{model_key.replace("_", " ")}.joblib', # alt çizgiler boşlukla
            f'models/trained_models/{model_key.replace("_", "-")}.joblib', # alt çizgiler tire ile
            f'models/trained_models/{model_key.replace(" ", "")}.joblib', # boşluklar kaldırılmış hali
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                file_path = alt_path
                break
        else:
            available_files = os.listdir('models/trained_models') if os.path.exists('models/trained_models') else []
            raise FileNotFoundError(
                f"Model dosyası bulunamadı: {file_path}\n"
                f"Aranan model: {model_key}\n"
                f"Mevcut dosyalar: {available_files}"
            )
    
    return joblib.load(file_path)

# Model sonuçlarını ve yollarını yükleme
def load_model_results():
    with open('assets/model_results.json', 'r') as f:
        return json.load(f)

# Model yollarını yükleme
def load_model_paths():
    with open('assets/model_paths.json', 'r') as f:
        return json.load(f)

# Özellik aralıklarını yükleme
def load_feature_ranges():
    df = pd.read_csv('dataset/data.csv')
    df = df.drop(['Unnamed: 32', 'id'], axis=1)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    feature_ranges = {}
    for column in df.columns:
        if column != 'diagnosis':
            feature_ranges[column] = {
                'min': float(df[column].min()),
                'max': float(df[column].max()),
                'mean': float(df[column].mean()),
                'std': float(df[column].std())
            }
    return feature_ranges

# Model açıklamaları
MODEL_DESCRIPTIONS = {
    'Logistic Regression': "Lojistik Regresyon, sınıflandırma problemlerinde kullanılan istatistiksel bir modeldir. Sigmoid fonksiyonu kullanarak olasılık tahmini yapar.",
    'Random Forest': "Random Forest, çok sayıda karar ağacının oluşturduğu bir topluluk modelidir. Demokratik oylama ile karar verir.",
    'Extra Trees': "Extra Trees, rastgeleleştirilmiş çok sayıda karar ağacıyla çalışan bir topluluk yöntemidir. Genellikle daha hızlı ve çeşitlidir.",
    'Bagging': "Bagging, aynı türden birçok modelin farklı veri alt kümeleri üzerinde eğitildiği bir topluluk yöntemidir. Aşırı öğrenmeyi azaltabilir.",
    'Gradient Boosting': "Gradient Boosting, zayıf öğrenicileri sıralı olarak eğiten ve hataları düzelten bir topluluk yöntemidir.",
    'Ada Boost': "AdaBoost, zayıf sınıflandırıcıları art arda eğiterek hata yapan örneklere daha fazla ağırlık verir.",
    'SVM': "Destek Vektör Makineleri (SVM), veriyi en geniş marjla ayıran hiperdüzlemi bulur. Yüksek boyutlu verilerde etkilidir.",
    'KNN': "K-En Yakın Komşu (KNN), bir veri noktasının sınıfını en yakın komşularının çoğunluğuna göre belirler.",
    'Naive Bayes': "Naive Bayes, Bayes teoremine dayanan ve özelliklerin bağımsız olduğu varsayımıyla çalışan bir sınıflandırıcıdır.",
    'LDA': "Linear Discriminant Analysis (LDA), sınıflar arası ayrımı en üst düzeye çıkarmak için doğrusal kombinasyonlar oluşturur.",
    'MLP': "Multi-Layer Perceptron (MLP), ileri beslemeli yapay sinir ağıdır. Karmaşık desenleri öğrenmek için gizli katmanlar kullanır."
}

# Her model için detaylı bilgi linkleri
MODEL_LINKS = {
    'Logistic Regression': "https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression",
    'Random Forest': "https://scikit-learn.org/stable/modules/ensemble.html#random-forests",
    'Extra Trees': "https://scikit-learn.org/stable/modules/ensemble.html#extra-trees",
    'Bagging': "https://scikit-learn.org/stable/modules/ensemble.html#bagging",
    'Gradient Boosting': "https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting",
    'Ada Boost': "https://scikit-learn.org/stable/modules/ensemble.html#adaboost",
    'SVM': "https://scikit-learn.org/stable/modules/svm.html#svm-classification",
    'KNN': "https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-classification",
    'Naive Bayes': "https://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes",
    'LDA': "https://scikit-learn.org/stable/modules/lda_qda.html#linear-discriminant-analysis",
    'MLP': "https://scikit-learn.org/stable/modules/neural_networks_supervised.html#mlp"
}