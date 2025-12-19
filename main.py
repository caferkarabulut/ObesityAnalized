import pandas as pd
import matplotlib.pyplot as plt
from classification import run_classification
from regression import run_regression
from clustering import run_clustering


def show_dataset_columns(csv_path):
    """Veri setindeki sütun başlıklarını görsel olarak göster"""
    df = pd.read_csv(csv_path)
    
    # Türkçe açıklamalar
    descriptions = {
        'Gender': 'Cinsiyet',
        'Age': 'Yaş',
        'Height': 'Boy (metre)',
        'Weight': 'Kilo (kg)',
        'family_history_with_overweight': 'Ailede obezite geçmişi',
        'FAVC': 'Yüksek kalorili yiyecek tüketimi',
        'FCVC': 'Sebze tüketim sıklığı',
        'NCP': 'Günlük ana öğün sayısı',
        'CAEC': 'Öğün arası yeme alışkanlığı',
        'SMOKE': 'Sigara kullanımı',
        'CH2O': 'Günlük su tüketimi',
        'SCC': 'Kalori takibi yapma',
        'FAF': 'Fiziksel aktivite sıklığı',
        'TUE': 'Teknoloji kullanım süresi',
        'CALC': 'Alkol tüketimi',
        'MTRANS': 'Ulaşım aracı tercihi',
        'NObeyesdad': 'Obezite seviyesi (Hedef değişken)'
    }
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Tablo verisi
    table_data = [[i+1, col, descriptions.get(col, '')] for i, col in enumerate(df.columns)]
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Numara', 'Sütun Adı', 'Açıklama'],
        loc='center',
        cellLoc='left',
        colColours=['#4CAF50', '#4CAF50', '#4CAF50']
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    plt.title('ObesityDataSet - Veri Seti Kategorileri', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('dataset_categories.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()


def main():
    csv_path = "ObesityDataSet.csv"

    # Veri seti kategorilerini göster
    show_dataset_columns(csv_path)
    
    run_classification(csv_path)
    run_regression(csv_path)
    run_clustering(csv_path)


if __name__ == "__main__":
    main()
