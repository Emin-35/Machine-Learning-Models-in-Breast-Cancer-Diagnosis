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

import pandas as pd
import joblib
import json
import os

# Klasör yapısını oluştur
os.makedirs('models', exist_ok=True)
os.makedirs('models/trained_models', exist_ok=True)
os.makedirs('assets', exist_ok=True)

# Veri setini temizleme ve hazırlama
def clean_data():
    df = pd.read_csv('dataset/data.csv') # Veri seti yolu
    df = df.drop(['Unnamed: 32', 'id'], axis=1) # Gereksiz sütunları kaldır
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0}) # M ve B etiketlerini sayısal değerlere dönüştür
    return df

# Modelleri eğitme ve değerlendirme
# Güncellenmiş model eğitim ve değerlendirme fonksiyonu
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, C=1.0, solver='liblinear', penalty='l2'), 
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=4, max_features='sqrt'),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=150, max_depth=10),
        'Bagging': BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=5), n_estimators=100),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8),
        'AdaBoost': AdaBoostClassifier(n_estimators=200, learning_rate=0.5),
        'SVM': SVC(C=1.0, kernel='rbf', gamma='scale', probability=True),
        'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance', metric='minkowski'),
        'NaiveBayes': GaussianNB(),
        'LDA': LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
        'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1500, alpha=0.0005),
    }

    results = {}
    trained_models = {}

    # GridSearch örnekleri (sadece bazı güçlü modeller için çok zaman alıyor)
    param_grids = {
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10],
            'max_features': ['sqrt'],
            'min_samples_split': [2, 4]
        },
        'GradientBoosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 4]
        },
        'SVM': {
            'C': [0.5, 1.0],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }
    }

    # Modelleri eğit ve değerlendir
    for name, model in models.items():
        print(f"Eğitiliyor: {name}...")
        # GridSearch kullanımı
        if name in param_grids:
            grid = GridSearchCV(model, param_grids[name], cv=5, n_jobs=-1, scoring='accuracy')
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
        else:
            model.fit(X_train, y_train)
            best_model = model

        # Tahminler
        y_pred = best_model.predict(X_test)
        
        # Temel metrikler
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # ROC-AUC skoru hesapla
        try:
            if hasattr(best_model, "predict_proba"):
                y_proba = best_model.predict_proba(X_test)[:, 1]
            elif hasattr(best_model, "decision_function"):
                y_proba = best_model.decision_function(X_test)
            else:
                y_proba = None
            
            if y_proba is not None:
                roc_auc = roc_auc_score(y_test, y_proba)
            else:
                roc_auc = None
        except Exception as e:
            print(f"ROC-AUC hesaplanamadı {name}: {e}")
            roc_auc = None

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Specificity ve Sensitivity hesapla
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Modeli kaydet
        model_path = f'models/trained_models/{name}.joblib'
        joblib.dump(best_model, model_path)

        # Sonuçları sakla (genişletilmiş)
        results[name] = {
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1': report['weighted avg']['f1-score'],
            'roc_auc': roc_auc,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'confusion_matrix': {
                'tn': int(tn), 'fp': int(fp), 
                'fn': int(fn), 'tp': int(tp)
            }
        }
        
        # Konsol çıktısı
        print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {report['weighted avg']['precision']:.4f}, Recall: {report['weighted avg']['recall']:.4f}, F1: {report['weighted avg']['f1-score']:.4f}")
        if roc_auc is not None:
            print(f"  ROC-AUC: {roc_auc:.4f}, Specificity: {specificity:.4f}, Sensitivity: {sensitivity:.4f}")
        
        trained_models[name] = model_path

    return results, trained_models

def main():
    data = clean_data()
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']

    # Standardizasyon
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Scaler'ı kaydet
    joblib.dump(scaler, 'models/trained_models/scaler.joblib')

    # Modelleri eğit ve değerlendir
    results, model_paths = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # JSON dosyalarına yaz
    with open('assets/model_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    with open('assets/model_paths.json', 'w') as f:
        json.dump(model_paths, f, indent=4)

    print("Tüm modeller başarıyla eğitildi ve kaydedildi!")

if __name__ == "__main__":
    main()