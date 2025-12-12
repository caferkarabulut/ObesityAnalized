# ObesityAnalized - Obezite Veri Analizi Projesi

Bu proje, obezite veri seti üzerinde sınıflandırma, regresyon ve kümeleme analizleri gerçekleştirir.

## 🚀 Kurulum

### 1. Repoyu Klonla

```bash
git clone https://github.com/caferkarabulut/ObesityAnalized.git
cd ObesityAnalized
```

### 2. Sanal Ortamı Aktif Et

```bash
# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Bağımlılıkları Yükle

```bash
pip install -r requirements.txt
```

## 📊 Kullanım

```bash
python main.py
```

## 📁 Proje Yapısı

```
ObesityAnalized/
├── main.py              # Ana çalıştırma dosyası
├── classification.py    # Random Forest sınıflandırma
├── regression.py        # BMI regresyon analizi
├── clustering.py        # K-Means kümeleme
├── ObesityDataSet.csv   # Veri seti
├── requirements.txt     # Bağımlılıklar
├── .gitignore           # Git ignore dosyası
└── venv/                # Sanal ortam
```

## 📈 Analizler

- **Sınıflandırma**: Random Forest ile obezite seviyesi tahmini
- **Regresyon**: Linear Regression ve Random Forest ile BMI tahmini
- **Kümeleme**: K-Means ile veri kümeleme ve PCA görselleştirmesi

## 📋 Çıktılar

| Dosya | Açıklama |
|-------|----------|
| `confusion_matrix_ve_metrikler.png` | Sınıflandırma performans metrikleri |
| `bmi_regression_sonuclari.png` | Regresyon sonuçları |
| `kmeans_elbow_silhouette.png` | K-Means optimizasyonu |
| `kmeans_pca_visualization.png` | Küme görselleştirmesi |
