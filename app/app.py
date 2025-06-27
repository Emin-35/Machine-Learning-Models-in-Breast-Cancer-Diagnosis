import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import sys
import os
import random

# Proje dizin yapısını ayarlama - models klasöründeki modülleri kullanabilmek için
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Modeller ve yardımcı fonksiyonları içe aktarma
from models.helper_functions import load_scaler, load_model, load_model_results, load_feature_ranges, MODEL_DESCRIPTIONS, MODEL_LINKS

#---------------------------------------------Header---------------------------------------------------------------------------------------

# Streamlit sayfa ayarlarını yapılandırma
st.set_page_config(
    page_title="Meme Kanseri Tanı Sistemi",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧬"
)

# Ana başlık ve açıklama bölümü
with st.container():
    st.title("🧬 Meme Kanseri Tanı Sistemi Baykar-Teknolojileri-Bitirme-Projesi")
    st.markdown("""
                Baykar Teknolojileri eğitimi üzerine verilmiş Wisconsin Teşhis Veri Seti kullanılarak geliştirilen makine öğrenimi modelleri ile hücre kümelerinin 
                **iyi huylu (benign)** veya **kötü huylu (malignant)** olarak sınıflandırılması için hazırlanmış bir streamlit proje sunumudur. \n
                Bu proje [**Mehmet Emin Palabıyık**](https://www.linkedin.com/in/emin-35bnry72698265/) tarafından hiçbir kar amacı gütmeksizin eğitim amaçlı olarak hazırlanmıştır. \n
                Bu uygulama, sitozis laboratuvarından aldığınız ölçümlere dayanarak bir meme kütlesinin iyi huylu mu yoksa kötü huylu mu olduğunu farklı makine öğrenimi modelleri kullanarak tahmin eder. \n 
                Ayrıca, kenar çubuğundaki kaydırıcıları kullanarak ölçümleri de güncelleyebilirsiniz.
                Güncellediğiniz ölçümlerle gerçek zamanlı analiz yapabilir ve farklı modellerin tahminlerini karşılaştırabilirsiniz.\n
                Bu uygulama sonucu, **dikkate alınmaması** gereken bir araçtır ve **kesin tıbbi tanı veya tedavi** için **kullanılmamalıdır.** Lütfen öncelikle doktorunuza danışınız\n
                """)
#---------------------------------------------Header---------------------------------------------------------------------------------------

#---------------------------------------------Side Bar---------------------------------------------------------------------------------------

# Yan çubuk başlığı ve model seçim alanı
st.sidebar.header("⚙️ Model ve Parametreler")

# Kullanıcının ML modelini seçmesi için dropdown menü
selected_model = st.sidebar.selectbox(
    "Model Seçiniz:",
    list(MODEL_DESCRIPTIONS.keys()),  # Mevcut tüm model isimlerini listele
    format_func=lambda x: x  # Model ismini olduğu gibi göster
)

# Seçilen modelin açıklamasını yan çubukta gösterme
st.sidebar.markdown("### 📝 Model Açıklaması")
st.sidebar.info(MODEL_DESCRIPTIONS[selected_model])

# Model hakkında daha detaylı bilgi için harici link
st.sidebar.markdown(f"📖 **[Daha Fazla Bilgi: {selected_model}]({MODEL_LINKS[selected_model]})**")

# Hücre özellikleri için slider'lar bölümü
st.sidebar.header("🔬 Hücre Özellikleri")

# Hücre özelliklerini mantıklı gruplara ayırma
feature_groups = {
    "Temel Ölçümler": [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean'
    ],
    "Doku Özellikleri (Ortalama)": [
        'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean',
        'symmetry_mean', 'fractal_dimension_mean'
    ],
    "Standart Hatalar": [
        'radius_se', 'texture_se', 'perimeter_se', 'area_se',
        'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
        'symmetry_se', 'fractal_dimension_se'
    ],
    "En Kötü Değerler": [
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
        'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst',
        'symmetry_worst', 'fractal_dimension_worst'
    ]
}

# Belirtilen min-max aralığında rastgele değer üretme fonksiyonu
def generate_random_value(min_val, max_val):
    return round(random.uniform(min_val, max_val), 4)

# Özellik isimlerini daha okunabilir hale getirme fonksiyonu
def prettify_feature_name(feature):
    parts = feature.split('_') # Özellik adını alt çizgiye göre ayır
    if parts[-1] in ['mean', 'se', 'worst']: # Özellik adının sonundaki kısımları ayır
        suffix = parts.pop(-1) # Son kısımları kaldır
        name = ' '.join(parts).title() # Kalan kısımları baş harfleri büyük olacak şekilde birleştir
        return f"{name} ({suffix})" # Son kısımları parantez içinde ekle
    else:
        return feature.title()

# Özellik değer aralıklarını yükleme
feature_ranges = load_feature_ranges()

# Kullanıcı girişlerini saklamak için session state başlatma
if 'input_features' not in st.session_state:
    st.session_state.input_features = {}

# Her özellik grubu için expandable section ve kontrol butonları oluşturma
for group_name, features in feature_groups.items():
    with st.sidebar.expander(group_name):
        # Her grup için "Orijinal Değerlere Dön" ve "Rastgele Değer" butonları
        col1, col2 = st.columns(2)
        with col1:
            reset_group = st.button("🔁 Orijinal", key=f"{group_name}_reset")
        with col2:
            random_group = st.button("🎲 Rastgele", key=f"{group_name}_random")

        # Orijinal değerlere dönme işlemi
        if reset_group:
            for feature in features:
                slider_key = f"{feature}_slider"
                st.session_state[slider_key] = float(feature_ranges[feature]['mean'])
            st.rerun()

        # Rastgele değer atama işlemi
        if random_group:
            for feature in features:
                slider_key = f"{feature}_slider"
                st.session_state[slider_key] = generate_random_value(
                    float(feature_ranges[feature]['min']), 
                    float(feature_ranges[feature]['max'])
                )
            st.rerun()

        # Her özellik için slider oluşturma
        for feature in features:
            f_range = feature_ranges[feature]
            label = prettify_feature_name(feature)
            slider_key = f"{feature}_slider"
            
            # Slider için varsayılan değer belirleme
            if slider_key not in st.session_state:
                st.session_state[slider_key] = float(f_range['mean'])

            # Slider widget'ı oluşturma
            value = st.slider(
                label=label,
                min_value=float(0),
                max_value=float(f_range['max']),
                value=st.session_state[slider_key],
                step=float((f_range['max'] - f_range['min']) / 100),
                key=slider_key
            )
            
            # Güncellenmiş değeri session state'e kaydetme
            st.session_state.input_features[feature] = value

# Analiz için kullanılacak giriş değerlerini alma
input_features = st.session_state.input_features

#---------------------------------------------Side Bar---------------------------------------------------------------------------------------

#---------------------------------------------Gerçek Zamanlı Analiz---------------------------------------------------------------------------------------

# Model verilerini ve scaler'ı yükleme
scaler = load_scaler()
model_results = load_model_results()

# Gerçek zamanlı analiz açık/kapalı durumunu session state'de saklama
if 'show_realtime_analysis' not in st.session_state:
    st.session_state.show_realtime_analysis = True

# Gerçek zamanlı analizi açıp kapatmak için toggle butonu
col1, col2, col3 = st.columns([3, 1, 3])
with col2:
    if st.button(
        "🔬 Gerçek Zamanlı Analiz " + ("(Açık)" if st.session_state.show_realtime_analysis else "(Kapalı)"),
        type="primary" if st.session_state.show_realtime_analysis else "secondary"
    ):
        st.session_state.show_realtime_analysis = not st.session_state.show_realtime_analysis
        st.rerun()

# Gerçek zamanlı analiz paneli - sadece toggle açıksa göster
if st.session_state.show_realtime_analysis:
    st.markdown("---")

    st.info("💡 Sol taraftaki **Hücre Özellikleri** bölümünden değerleri güncelleyerek, bu değerler ile analiz yapabilirsiniz. \n")
    
    # Model seçimi kontrolü (daha önce tanımlanmadıysa)
    if 'selected_model' not in locals():
        selected_model = st.selectbox(
            "🤖 Model Seçin:",
            options=list(MODEL_DESCRIPTIONS.keys()),
            index=0,
            help="Analiz için kullanılacak makine öğrenmesi modelini seçin"
        )
    
    try:
        # Mevcut slider değerlerini alma
        input_features = st.session_state.input_features
        
        # Eğer değerler mevcutsa analiz başlatma
        if input_features:
            with st.spinner("Gerçek zamanlı analiz yapılıyor..."):
                # Giriş verilerini ML modeline uygun formata dönüştürme
                input_df = pd.DataFrame([input_features])
                scaled_input = scaler.transform(input_df)
                
                # Seçili modeli yükleme ve tahmin yapma
                model_name = selected_model.replace(" ", "")
                model = load_model(model_name)
                
                prediction = model.predict(scaled_input)
                prediction_proba = model.predict_proba(scaled_input)
        
                # Ana sonuçları gösterme bölümü
                st.subheader("🔬 Gerçek Zamanlı Analiz Sonuçları")
                
                # Üç sütunlu sonuç kartları
                col1, col2, col3 = st.columns(3)
                with col1:
                    # Tahmin sonucunu gösterme
                    st.metric(
                        "Tahmin Sonucu", 
                        "Kötü Huylu ⚠️" if prediction[0] == 1 else "İyi Huylu ✅",
                        help="Modelin hücre kümesi hakkındaki tahmini"
                    )
                
                with col2:
                    # Modelin tahmininden emin olma derecesi
                    confidence = max(prediction_proba[0])
                    st.metric(
                        "Güven Skoru", 
                        f"{confidence:.2%}",
                        help="Modelin tahmininden ne kadar emin olduğu"
                    )
                
                with col3:
                    # Modelin genel doğruluk oranı
                    accuracy = model_results[model_name]['accuracy']
                    st.metric(
                        "Model Doğruluğu", 
                        f"{accuracy:.2%}",
                        help="Bu modelin test verisi üzerindeki doğruluk oranı"
                    )
                
                # Olasılık dağılımını bar chart ile görselleştirme
                prob_df = pd.DataFrame({
                    'Sınıf': ['İyi Huylu', 'Kötü Huylu'],
                    'Olasılık': prediction_proba[0]
                })
                
                # Plotly ile interaktif bar chart oluşturma
                fig = px.bar(
                    prob_df, x='Sınıf', y='Olasılık', 
                    color='Sınıf', text='Olasılık',
                    color_discrete_map={
                        'İyi Huylu': '#4CAF50',
                        'Kötü Huylu': '#F44336'
                    },
                    height=300
                )
                
                # Chart görünüm ayarları
                fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
                fig.update_layout(
                    yaxis_tickformat='.0%',
                    showlegend=False,
                    margin=dict(t=10, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Detaylı olasılık bilgilerini info box'ta gösterme
                with st.info("📊 Detaylı Olasılık Bilgileri"):
                    st.markdown(f"""
                    - **İyi huylu (benign) olma olasılığı:** {prediction_proba[0][0]:.4f} ({prediction_proba[0][0]:.2%})
                    - **Kötü huylu (malignant) olma olasılığı:** {prediction_proba[0][1]:.4f} ({prediction_proba[0][1]:.2%})
                    - **Kullanılan model:** {selected_model}
                    - **Toplam özellik sayısı:** {len(input_features)}
                    """)
                
                # Diğer tüm modellerin aynı veriler için tahminlerini gösterme
                with st.expander("🔍 **DİĞER MODELLERİN SONUÇLARI**", expanded=True):
                    st.markdown("**Aynı değerler için diğer modellerin tahminleri:**")
                    
                    # Model karşılaştırma listesi
                    model_comparison = []
                    
                    # Mevcut model hariç tüm modelleri test etme
                    for model_name in MODEL_DESCRIPTIONS.keys():
                        if model_name != selected_model:
                            try:
                                # Karşılaştırma modelini yükleme
                                comparison_model_name = model_name.replace(" ", "")
                                comparison_model = load_model(comparison_model_name)
                                
                                # Tahmin yapma
                                comp_prediction = comparison_model.predict(scaled_input)
                                comp_prediction_proba = comparison_model.predict_proba(scaled_input)
                                
                                # Sonuçları kaydetme
                                result = {
                                    'Model': model_name,
                                    'Tahmin': "Kötü Huylu" if comp_prediction[0] == 1 else "İyi Huylu",
                                    'İyi Huylu %': f"{comp_prediction_proba[0][0]:.2%}",
                                    'Kötü Huylu %': f"{comp_prediction_proba[0][1]:.2%}",
                                    'Güven': f"{max(comp_prediction_proba[0]):.2%}",
                                    'Model Doğruluğu': f"{model_results[comparison_model_name]['accuracy']:.2%}"
                                }
                                model_comparison.append(result)
                                
                            except Exception as e:
                                # Hata durumunda boş sonuç ekleme
                                model_comparison.append({
                                    'Model': model_name,
                                    'Tahmin': "❌ Hata",
                                    'İyi Huylu %': "-",
                                    'Kötü Huylu %': "-",
                                    'Güven': "-",
                                    'Model Doğruluğu': "-"
                                })
                    
                    # Karşılaştırma sonuçları varsa tablo halinde gösterme
                    if model_comparison:
                        # DataFrame oluşturma
                        comparison_df = pd.DataFrame(model_comparison)
                        
                        # Tahmin sonuçlarına göre renklendirme fonksiyonu
                        def color_predictions(val):
                            if val == "İyi Huylu":
                                return 'background-color: #d4edda; color: #155724'
                            elif val == "Kötü Huylu":
                                return 'background-color: #f8d7da; color: #721c24'
                            else:
                                return ''
                        
                        # Stillendirilmiş tablo gösterimi
                        styled_df = comparison_df.style.map(
                            color_predictions, subset=['Tahmin']
                        )
                        
                        st.dataframe(styled_df, use_container_width=True, hide_index=True)
                        
                        # Modeller arası konsensüs istatistikleri
                        benign_count = sum(1 for result in model_comparison if result['Tahmin'] == "İyi Huylu")
                        malignant_count = sum(1 for result in model_comparison if result['Tahmin'] == "Kötü Huylu")
                        total_models = len(model_comparison)
                        
                        # Özet metrikleri gösterme
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("İyi Huylu Diyen", f"{benign_count}/{total_models}")
                        with col2:
                            st.metric("Kötü Huylu Diyen", f"{malignant_count}/{total_models}")
                        with col3:
                            # Genel konsensüs gücünü hesaplama
                            if total_models > 0:
                                consensus = "Güçlü" if max(benign_count, malignant_count) >= total_models * 0.8 else "Zayıf"
                                st.metric("Genel Karar", consensus)
                    else:
                        st.warning("Diğer modeller yüklenemedi.")
        
        else:
            st.info("⏳ Slider değerleri yükleniyor...")
    
    # Hata yakalama ve kullanıcıya bilgi verme
    except FileNotFoundError as e:
        st.error(f"❌ Model dosyası bulunamadı: {str(e)}")
        # Mevcut model dosyalarını listeleme
        if os.path.exists('models'):
            available_models = [f for f in os.listdir('models') if f.endswith('.joblib')]
            st.info(f"📁 Mevcut model dosyaları: {available_models}")
    
    except Exception as e:
        st.error(f"❌ Analiz sırasında hata oluştu: {str(e)}")
        st.error("🔧 Lütfen slider değerlerinin doğru ayarlandığından emin olun.")

else:
    # Gerçek zamanlı analiz kapalıysa kullanıcıya bilgi verme
    st.info("💡 Gerçek zamanlı analiz için yukarıdaki butona tıklayın!")

#---------------------------------------------Gerçek Zamanlı Analiz---------------------------------------------------------------------------------------

#---------------------------------------------Plotly ile Radar Chart Analizi------------------------------------------------------------------------------
# Kullandığım kütüphane için bu linkten daha fazla bilgiye erişebilirsiniz
# https://plotly.com/python/radar-chart/

#----------------------------------------------
import plotly.graph_objects as go

st.subheader("📊 Model Radar-Chart Grafiği")

# Global min/max değerleri ile normalizasyon yapan fonksiyon
def custome_normalize(input_values):
    data = pd.read_csv("dataset/data.csv")  # Veri setini oku
    data = data.drop(['Unnamed: 32', 'id'], axis=1)  # Gereksiz sütunları kaldır

    scaled_dict = {}

    for key, val in input_values.items():
        max_val = data[key].max()
        min_val = data[key].min()
        scaled_value = (val - min_val) / (max_val - min_val)  # Min-max normalizasyon
        scaled_dict[key] = scaled_value
        
    return scaled_dict

# Logaritmik + min-max normalizasyon uygulayan fonksiyon
def log_normalize(input_values):
    data = pd.read_csv("dataset/data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1)

    scaled_dict = {}

    for key, val in input_values.items():
        data_col = data[key]
        data_log = np.log1p(data_col)  # log(1+x) uygulayarak büyük değer farklarını azalt
        val_log = np.log1p(val)
        
        max_log = data_log.max()
        min_log = data_log.min()
        
        if max_log == min_log:
            scaled_value = 0.5  # Tüm değerler aynıysa orta değeri ata
        else:
            scaled_value = (val_log - min_log) / (max_log - min_log)
            scaled_value = scaled_value * 0.8 + 0.1  # 0.1-0.9 aralığına yeniden ölçekle
        
        scaled_dict[key] = scaled_value
        
    return scaled_dict

# Percentile tabanlı normalizasyon yapan fonksiyon
def percentile_normalize(input_values):
    data = pd.read_csv("dataset/data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1)

    scaled_dict = {}

    for key, val in input_values.items():
        data_col = data[key]
        p25 = np.percentile(data_col, 25)  # 25. yüzdelik dilim
        p75 = np.percentile(data_col, 75)  # 75. yüzdelik dilim
        
        if p75 == p25:
            scaled_value = 0.5  # Tüm değerler aynıysa orta değeri ata
        else:
            scaled_value = (val - p25) / (p75 - p25)  # IQR ile normalizasyon
            scaled_value = np.clip(scaled_value, 0, 1)  # 0 ile 1 arası sınırla
            scaled_value = scaled_value * 0.8 + 0.1  # 0.1-0.9 aralığına yeniden ölçekle
        
        scaled_dict[key] = scaled_value
        
    return scaled_dict

# Radar chart oluşturma fonksiyonu
def get_radar_chart(input_features):

    # Radar grafikte gösterilecek kategoriler
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                  'Smoothness', 'Compactness',
                  'Concavity', 'Concave Points',
                  'Symmetry', 'Fractal Dimension']

    fig = go.Figure()  # Radar chart figürü oluştur

    # Normalizasyon türünü belirleyen buton callback fonksiyonları
    def set_log_normalization():
        st.session_state["use_log"] = True

    def set_percentile_normalization():
        st.session_state["use_log"] = False

    # Varsayılan normalizasyon türü (log)
    if "use_log" not in st.session_state:
        st.session_state["use_log"] = True

    # Butonları iki sütun halinde yerleştir
    col1, col2 = st.columns(2)

    with col1:
        st.button(
            "🔄 Logaritmik Normalizasyon",
            key="toggle_log",
            on_click=set_log_normalization
        )
        # Bilgilendirme kutusu
        st.info("**🔄 Logaritmik Normalizasyon:** Değerler arasında büyük farklar varsa (örneğin 0.01 ile 1000 gibi), bu yöntemde logaritma kullanılarak ölçek farkı azaltılır. Böylece uç değerlerin etkisi düşer ve veriler daha dengeli hale gelir.")

    with col2:
        st.button(
            "📊 Percentile Normalizasyon",
            key="toggle_percentile",
            on_click=set_percentile_normalization
        )
        st.info("**📊 Percentile Normalizasyon:** Her değerin verinin dağılımı içindeki yüzdelik konumu (percentile) hesaplanır. Özellikle sıralı verilerde daha adil bir karşılaştırma sağlar. Aykırı değerlerin etkisini azaltır, veriler daha dengeli hale gelir.")

    # Seçilen normalizasyona göre işlem yap
    use_log = st.session_state["use_log"]

    if use_log:
        normalized_features = log_normalize(input_features)
    else:
        normalized_features = percentile_normalize(input_features)

    # Normalize edilmiş değerleri gruplara ayır
    mean_vals = []
    se_vals = []
    worst_vals = []
    
    for key, val in normalized_features.items():
        if key.endswith('_mean'):
            mean_vals.append(val)
        elif key.endswith('_se'):
            se_vals.append(val)
        elif key.endswith('_worst'):
            worst_vals.append(val)
    
    # Mean değerler için radar grafiği çiz
    fig.add_trace(go.Scatterpolar(
        r=mean_vals,
        theta=categories,
        fill='toself',
        name='Mean Values',
        line_color='blue',
        fillcolor='rgba(0, 0, 255, 0.1)'
    ))
    
    # Standart hata değerleri için çizim
    fig.add_trace(go.Scatterpolar(
        r=se_vals,
        theta=categories,
        fill='toself',
        name='Standard Errors',
        line_color='green',
        fillcolor='rgba(0, 255, 0, 0.1)'
    ))
    
    # Worst (en kötü) değerler için çizim
    fig.add_trace(go.Scatterpolar(
        r=worst_vals,
        theta=categories,
        fill='toself',
        name='Worst Values',
        line_color='red',
        fillcolor='rgba(255, 0, 0, 0.1)'
    ))

    # Grafiğin genel görünüm ayarları
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0.1, 0.3, 0.5, 0.7, 0.9],
                ticktext=['Düşük', 'Düşük-Orta', 'Orta', 'Orta-Yüksek', 'Yüksek'],
                tickfont=dict(size=10, color='black'),
            )
        ),
        showlegend=True,
        title="Kanser Hücresi Özellikleri - Radar Chart",
        width=600,
        height=600
    )

    return fig

#----------------------------------------------Plotly ile Radar Chart Analizi------------------------------------------------------------------------------

#----------------------------------------------Model Performans Sonuçları----------------------------------------------------------------------------------

def add_predictions(input_features, model_results):
    # Custom CSS stilleri
    st.markdown("""
    <style>
    .prediction-container {
        padding: 1.5rem;
        border-radius: 0.8rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .prediction-title {
        color: #fff;
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .diagnosis {
        color: #fff;
        padding: 0.8rem 1.2rem;
        border-radius: 0.5rem;
        font-size: 1.3rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        animation: fadeIn 0.5s ease-in;
    }
    
    .diagnosis.benign {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        border-left: 5px solid #2E7D32;
    }
    
    .diagnosis.malignant {
        background: linear-gradient(135deg, #f44336, #d32f2f);
        border-left: 5px solid #C62828;
    }
    
    .probability-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #fff;
    }
    
    .probability-value {
        font-size: 1.1rem;
        font-weight: bold;
        color: #fff;
    }
    
    .accuracy-badge {
        background: linear-gradient(135deg, #FF6B6B, #4ECDC4);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #FFA726, #FF7043);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        border-left: 4px solid #F57C00;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .result-icon {
        font-size: 2rem;
        margin-right: 0.5rem;
        vertical-align: middle;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Modeli ve scaler'ı yükle
    model_name = selected_model.replace(" ", "")
    model = load_model(model_name)
    scaler = load_scaler()

    # Girdileri hazırla
    input_features = np.array([list(input_features.values())]).reshape(1, -1)

    # Özellikleri ölçeklendir
    df_scaled_input = pd.DataFrame(input_features)
    scaled_input = scaler.transform(df_scaled_input)

    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)

    # Ana container
    st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
    
    # Başlık
    st.markdown('<div class="prediction-title">🔍 Model Tahmin Sonuçları</div>', unsafe_allow_html=True)
    
    # Tahmin sonucu
    if prediction[0] == 1:
        diagnosis_class = "malignant"
        diagnosis_text = '<span class="result-icon">⚠️</span>Kötü Huylu (Malignant)'
        result_color = "#f44336"
    else:
        diagnosis_class = "benign"
        diagnosis_text = '<span class="result-icon">✅</span>İyi Huylu (Benign)'
        result_color = "#4CAF50"
    
    st.markdown(f'<div class="diagnosis {diagnosis_class}">{diagnosis_text}</div>', unsafe_allow_html=True)
    
    # Olasılık kartları
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f'''
        <div class="probability-card">
            <div style="color: #4CAF50; font-weight: bold; margin-bottom: 0.5rem;">
                ✅ İyi Huylu Olasılığı
            </div>
            <div class="probability-value">{prediction_proba[0][0]:.2%}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="probability-card">
            <div style="color: #f44336; font-weight: bold; margin-bottom: 0.5rem;">
                ⚠️ Kötü Huylu Olasılığı
            </div>
            <div class="probability-value">{prediction_proba[0][1]:.2%}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Model doğruluğu
    st.markdown(f'''
    <div style="text-align: center; margin: 1rem 0;">
        <span class="accuracy-badge">
            🎯 Model Doğruluğu: {model_results[model_name]['accuracy']:.2%}
        </span>
        <div style="color: #fff; margin-top: 0.5rem; font-size: 1.1rem;">
            <strong>📊 Kullanılan Model: {model_name}</strong>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Uyarı mesajı
    st.markdown('''
    <div class="warning-box">
        <strong>⚠️ Önemli Uyarı:</strong> Bu tahmin, modelin eğitim verilerine dayanmaktadır ve gerçek tıbbi kararlar için kullanılmamalıdır. Lütfen doktorunuza danışın.
    </div>
    ''', unsafe_allow_html=True)

#---------------------------------------------Plotly ile Radar Chart Analizi------------------------------------------------------------------------------

#---------------------------------------------Model performans karşılaştırmaları--------------------------------------------------------------------------

from plotly.subplots import make_subplots

def create_performance_summary_cards(metrics_df):
    # En yüksek doğruluğa sahip modeli al
    best_model = metrics_df.loc[metrics_df['Doğruluk'].idxmax()]
    # Ortalama doğruluğu hesapla
    avg_accuracy = metrics_df['Doğruluk'].mean()
    # En yüksek kesinliğe sahip modeli al
    best_precision = metrics_df.loc[metrics_df['Kesinlik'].idxmax()]
    # En yüksek hassasiyete sahip modeli al
    best_recall = metrics_df.loc[metrics_df['Hassasiyet'].idxmax()]
    
    # 4 sütunlu kart yapısı oluştur
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # En iyi modeli ve doğruluğunu göster
        st.metric(
            label="🏆 En İyi Model",
            value=best_model['Model'],
            delta=f"{best_model['Doğruluk']:.1%}"
        )
    
    with col2:
        # Ortalama doğruluk ve en iyi modelle farkı göster
        st.metric(
            label="📊 Ortalama Doğruluk",
            value=f"{avg_accuracy:.1%}",
            delta=f"{best_model['Doğruluk'] - avg_accuracy:+.1%}"
        )
    
    with col3:
        # En iyi kesinliğe sahip modeli göster
        st.metric(
            label="🎯 En İyi Kesinlik",
            value=best_precision['Model'],
            delta=f"{best_precision['Kesinlik']:.1%}"
        )
    
    with col4:
        # En iyi hassasiyete sahip modeli göster
        st.metric(
            label="🔍 En İyi Hassasiyet",
            value=best_recall['Model'],
            delta=f"{best_recall['Hassasiyet']:.1%}"
        )
# ----------------------------------------------Model performans karşılaştırmaları--------------------------------------------------------------------------

# ---------------------------------------------Model dashboard'u ve gauge chart'lar--------------------------------------------------------------s

# Model metriklerini tabloya dönüştürür ve renk sözlüğü oluşturur
def create_model_performance_dashboard(model_results):
    metrics = []
    for model, scores in model_results.items():
        metrics.append({
            'Model': model,
            'Doğruluk': scores['accuracy'],
            'Kesinlik': scores['precision'],
            'Hassasiyet': scores['recall'],
            'F1-Skoru': scores['f1']
        })
    
    metrics_df = pd.DataFrame(metrics)

    # Model adlarına karşılık gelen renkler
    model_colors = {
        'LogisticRegression': '#FF6347',
        'RandomForest': '#2E8B57',
        'ExtraTrees': '#6B8E23',
        'Bagging': '#A0522D',
        'GradientBoosting': '#20B2AA',
        'AdaBoost': '#DC143C',
        'SVM': '#4169E1',
        'KNN': '#9370DB',
        'NaiveBayes': '#FF8C00',
        'LDA': '#8B0000',
        'MLP': "#02F8F8"
    }
    
    return metrics_df, model_colors

def create_accuracy_gauge_chart(metrics_df, model_colors):
    # En yüksek doğruluğa sahip modeli bul
    best_model = metrics_df.loc[metrics_df['Doğruluk'].idxmax()]
    
    # Gauge chart için subplot yapısı oluştur
    models = metrics_df['Model'].tolist()
    n_models = len(models)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{"type": "indicator"}] * cols for _ in range(rows)],
        vertical_spacing=0.25,
        horizontal_spacing=0.1
    )
    
    for i, (_, row) in enumerate(metrics_df.iterrows()):
        row_num = i // cols + 1
        col_num = i % cols + 1
        
        accuracy = row['Doğruluk']
        model_name = row['Model']
        
        # En iyi model altın rengiyle gösterilir
        if model_name == best_model['Model']:
            color = "gold"
            bar_color = "rgba(255, 215, 0, 0.8)"
            display_name = f"🏆 {model_name}"
        else:
            color = model_colors.get(model_name, "#636EFA")
            bar_color = f"rgba{tuple(list(px.colors.hex_to_rgb(color)) + [0.8])}"
            display_name = model_name

        # Her model için gauge chart oluştur
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=accuracy * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={
                    'text': f"<b>{display_name}</b>", 
                    'font': {'size': 16}
                },
                number={
                    'suffix': "%", 
                    'font': {'size': 20, 'color': 'white'}
                },
                gauge={
                    'axis': {
                        'range': [None, 100], 
                        'tickwidth': 1, 
                        'tickcolor': "white",
                        'tickfont': {'size': 10}
                    },
                    'bar': {'color': bar_color},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 70], 'color': 'rgba(255, 0, 0, 0.1)'},
                        {'range': [70, 85], 'color': 'rgba(255, 255, 0, 0.1)'},
                        {'range': [85, 100], 'color': 'rgba(0, 255, 0, 0.1)'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=row_num, col=col_num
        )
    
    # Grafik genel ayarları
    fig.update_layout(
        title={
            'text': f"Model Doğruluk Performansı</b><br><sub>🏆 En İyi: {best_model['Model']} ({best_model['Doğruluk']:.1%})</sub>",
            'x': 0.5,
            'font': {'size': 18}
        },
        height=400 * rows,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=160, b=20, l=20, r=20)
    )
    
    return fig

# ---------------------------------------------Model performans dashboard'u ve gauge chart'lar------------------------------------------------------------

# ---------------------------------------------Detaylı Metrik Analizi ve Heatmap--------------------------------------------------------------------------

def create_detailed_metrics_heatmap(metrics_df):
    # Metrikleri modele göre satırlara ayır
    metrics_only = metrics_df.set_index('Model')[['Doğruluk', 'Kesinlik', 'Hassasiyet', 'F1-Skoru']]
    
    # Isı haritası oluştur
    fig = go.Figure(data=go.Heatmap(
        z=metrics_only.values,
        x=metrics_only.columns,
        y=metrics_only.index,
        colorscale='RdYlGn',
        text=[[f'{val:.1%}' for val in row] for row in metrics_only.values],
        texttemplate="%{text}",
        textfont={"size": 12, "color": "white"},
        colorbar=dict(
            title="Performans",
            tickvals=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            ticktext=['50%', '60%', '70%', '80%', '90%', '100%']
        ),
        hoverongaps=False,
        hovertemplate="<b>%{y}</b><br>" +
                     "Metrik: %{x}<br>" +
                     "Değer: %{text}<br>" +
                     "<extra></extra>"
    ))
    
    fig.update_layout(
        title={
            'text': "<b>Model Performans Detayları</b><br><sub>Tüm Metrikler Isı Haritası</sub>",
            'x': 0.45,
            'font': {'size': 18}
        },
        xaxis_title="Performans Metrikleri",
        yaxis_title="Modeller",
        height=400,
        width=500
    )
    
    return fig

# ---------------------------------------------Detaylı Metrik Analizi ve Heatmap--------------------------------------------------------------------------

# ---------------------------------------------ROC-AUC Performans Görselleştirmesi------------------------------------------------------------

def create_roc_auc_bar_chart(metrics_df, model_colors):
    """ROC-AUC skorlarını bar chart olarak gösterir"""
    
    # ROC-AUC verilerini al (train_models.py'dan gelen sonuçlarda var)
    model_results = load_model_results()
    
    # ROC-AUC değerlerini topla
    roc_data = []
    for model_name, results in model_results.items():
        if 'roc_auc' in results and results['roc_auc'] is not None:
            roc_data.append({
                'Model': model_name,
                'ROC-AUC': results['roc_auc']
            })
    
    if not roc_data:
        st.warning("ROC-AUC verileri bulunamadı.")
        return None
    
    roc_df = pd.DataFrame(roc_data)
    roc_df = roc_df.sort_values('ROC-AUC', ascending=True)
    
    # En iyi ROC-AUC skorunu bul
    best_roc_model = roc_df.loc[roc_df['ROC-AUC'].idxmax()]
    
    # Bar chart oluştur
    fig = px.bar(
        roc_df, 
        x='ROC-AUC', 
        y='Model',
        orientation='h',
        title=f"<b>ROC-AUC Performans Karşılaştırması</b><br><sub>🏆 En İyi: {best_roc_model['Model']} ({best_roc_model['ROC-AUC']:.3f})</sub>",
        text='ROC-AUC',
        color='Model',
        color_discrete_map=model_colors
    )
    
    # Grafik düzenlemeleri
    fig.update_traces(
        texttemplate='%{text:.3f}', 
        textposition='inside',
        textfont=dict(color='white', size=12)
    )
    
    fig.update_layout(
        xaxis_title="ROC-AUC Skoru",
        yaxis_title="Modeller",
        height=400,
        showlegend=False,
        xaxis=dict(range=[0, 1]),
        margin=dict(t=80, b=40, l=120, r=40)
    )
    
    # En iyi modeli vurgula
    fig.add_vline(
        x=best_roc_model['ROC-AUC'], 
        line_dash="dash", 
        line_color="gold", 
        line_width=3,
        annotation_text=f"En İyi: {best_roc_model['ROC-AUC']:.3f}",
        annotation_position="top"
    )
    
    return fig

# ---------------------------------------------Confusion Matrix Heatmap Görselleştirmesi------------------------------------------------

def create_confusion_matrix_heatmap(model_results, selected_models=None):
    """Seçilen modeller için confusion matrix heatmap'i oluşturur"""
    
    if selected_models is None:
        # Tüm modelleri al
        selected_models = list(model_results.keys())
    
    # Subplot sayısını hesapla
    n_models = len(selected_models)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    # Subplot başlıkları
    subplot_titles = [f"{model}" for model in selected_models]
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subplot_titles,
        specs=[[{"type": "heatmap"}] * cols for _ in range(rows)],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    for i, model_name in enumerate(selected_models):
        if 'confusion_matrix' not in model_results[model_name]:
            continue
            
        cm_data = model_results[model_name]['confusion_matrix']
        
        # Confusion matrix'i 2x2 array olarak düzenle
        cm_matrix = [
            [cm_data['tn'], cm_data['fp']],
            [cm_data['fn'], cm_data['tp']]
        ]
        
        # Yüzdelik hesaplama
        total = sum(cm_data.values())
        cm_percentages = [
            [f"{cm_data['tn']}<br>({cm_data['tn']/total:.1%})", f"{cm_data['fp']}<br>({cm_data['fp']/total:.1%})"],
            [f"{cm_data['fn']}<br>({cm_data['fn']/total:.1%})", f"{cm_data['tp']}<br>({cm_data['tp']/total:.1%})"]
        ]
        
        row_num = i // cols + 1
        col_num = i % cols + 1

        # Custom koyu mavi-mor tonları, farklı renkler grafikte okunmadığı için daha belirgin renkler kullandım
        custome_colorscale=[[0, 'rgb(8,29,88)'],    # Koyu mavi
            [0.25, 'rgb(37,52,148)'], # Mavi
            [0.5, 'rgb(68,1,84)'],    # Koyu mor
            [0.75, 'rgb(122,7,75)'],  # Mor
            [1, 'rgb(177,56,42)']]    # Koyu kırmızı
        
        # Heatmap ekle
        fig.add_trace(
            go.Heatmap(
                z=cm_matrix,
                x=["Tahmin: İyi Huylu", "Tahmin: Kötü Huylu"],
                y=["Gerçek: İyi Huylu", "Gerçek: Kötü Huylu"],
                colorscale= custome_colorscale,
                text=cm_percentages,
                texttemplate="%{text}",
                textfont={"size": 10, "color": "white"},
                showscale=i==0,  # Sadece ilk grafikte renk skalası göster
                hoverongaps=False,
                hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>" +
                             "Sayı: %{z}<br>" +
                             "<extra></extra>",
                colorbar=dict(
                    title="Sayı",
                    x=1.02 if i == 0 else None
                ) if i == 0 else None
            ),
            row=row_num, col=col_num
        )
    
    fig.update_layout(
        title={
            'text': "<b>Confusion Matrix Karşılaştırması</b><br><sub>TN: Doğru Negatif, FP: Yanlış Pozitif, FN: Yanlış Negatif, TP: Doğru Pozitif</sub>",
            'x': 0.5,
            'font': {'size': 16}
        },
        height=300 * rows,
        showlegend=False
    )
    
    return fig

# ---------------------------------------------Gelişmiş Metrik Özet Kartları-----------------------------------------------------------

def create_advanced_performance_cards(model_results):
    """ROC-AUC, Sensitivity, Specificity için özet kartları oluşturur"""
    
    # En iyi performansları bul
    best_roc = max([(name, data.get('roc_auc', 0)) for name, data in model_results.items() 
                   if data.get('roc_auc') is not None], key=lambda x: x[1], default=("N/A", 0))
    # Ortalama değerleri hesapla
    avg_roc = np.mean([data.get('roc_auc', 0) for data in model_results.values() 
                      if data.get('roc_auc') is not None])
    
    # 4 sütunlu kart yapısı
    col1, col2= st.columns(2)
    
    with col1:
        st.metric(
            label="🎯 En İyi ROC-AUC",
            value=best_roc[0],
            delta=f"{best_roc[1]:.3f}" if best_roc[1] > 0 else "N/A"
        )
    
    with col2:
        st.metric(
            label="📊 Ortalama ROC-AUC",
            value=f"{avg_roc:.3f}" if avg_roc > 0 else "N/A",
            delta=f"{best_roc[1] - avg_roc:+.3f}" if best_roc[1] > 0 and avg_roc > 0 else None
        )

# ---------------------------------------------ROC-AUC Performans Görselleştirmesi------------------------------------------------------------

# ---------------------------------------------Display Dashboard Interface--------------------------------------------------------------------------------
# Ana kullanım fonksiyonu
def display_model_performance_dashboard(model_results):
    """Gelişmiş model performans dashboard'u - ROC-AUC ve Confusion Matrix dahil"""
    
    metrics_df, model_colors = create_model_performance_dashboard(model_results)
    
    # Mevcut performans özeti
    st.subheader("📈 Temel Performans Özeti")
    create_performance_summary_cards(metrics_df)
    
    # Yeni gelişmiş performans kartları
    st.subheader("🎯 Gelişmiş Performans Metrixleri")
    create_advanced_performance_cards(model_results)
    
    st.divider()
    
    # Mevcut doğruluk gauge'ları
    st.subheader("🎯 Doğruluk Performansı")
    gauge_fig = create_accuracy_gauge_chart(metrics_df, model_colors)
    st.plotly_chart(gauge_fig, use_container_width=True)
    
    st.divider()

    # Mevcut detaylı metrik heatmap
    st.subheader("🔥 Detaylı Metrik Analizi")  
    heatmap_fig = create_detailed_metrics_heatmap(metrics_df)
    st.plotly_chart(heatmap_fig, use_container_width=True)

    st.divider()
    
    # ROC-AUC bar chart
    st.subheader("📊 ROC-AUC Performans Analizi")
    roc_fig = create_roc_auc_bar_chart(metrics_df, model_colors)
    if roc_fig:
        st.plotly_chart(roc_fig, use_container_width=True)
        
        # ROC-AUC hakkında bilgi kutusu
        st.info("""
        **📊 ROC-AUC Skoru Nedir?**
        - **0.5**: Rastgele tahmin (hiç iyi değil)
        - **0.7-0.8**: Kabul edilebilir performans  
        - **0.8-0.9**: Mükemmel performans
        - **0.9-1.0**: Olağanüstü performans
        
        ROC-AUC, modelin farklı eşik değerlerinde ne kadar iyi sınıflandırma yaptığını ölçer.
        """)
    
    st.divider()
    
    # Confusion Matrix karşılaştırması
    st.subheader("🔥 Confusion Matrix Analizi")
    
    # Model seçim kutusu
    available_models = [name for name in model_results.keys() 
                       if 'confusion_matrix' in model_results[name]]
    
    if available_models:
        selected_models_for_cm = st.multiselect(
            "Confusion Matrix için modelleri seçin:",
            available_models,
            default=available_models[:6] if len(available_models) > 6 else available_models,
            help="En fazla 9 model seçebilirsiniz"
        )
        
        if selected_models_for_cm:
            cm_fig = create_confusion_matrix_heatmap(model_results, selected_models_for_cm[:9])
            st.plotly_chart(cm_fig, use_container_width=True)
            
            # Confusion Matrix açıklama
            st.info("""
            **🔥 Confusion Matrix Açıklaması:**
            - **TN (True Negative)**: Gerçekten iyi huylu olanları doğru tespit
            - **FP (False Positive)**: İyi huyluyu kötü huylu olarak yanlış tespit  
            - **FN (False Negative)**: Kötü huyluyu iyi huylu olarak yanlış tespit ⚠️ Tehlikeli!
            - **TP (True Positive)**: Gerçekten kötü huylu olanları doğru tespit
            """)
    else:
        st.warning("Confusion Matrix verileri bulunamadı.")
    
    st.divider()
    
    return metrics_df

# Radar grafik ve tahmin modülünü göster
col1, col2 = st.columns([3, 1])

with col1:
    radar_chart = get_radar_chart(input_features)
    st.plotly_chart(radar_chart, use_container_width=True)
with col2:
    add_predictions(input_features, model_results)

st.divider()

# Model performans dashboard'u başlat
st.header("🤖 Model Performans Analizi")
display_model_performance_dashboard(model_results)

# ---------------------------------------------Display Dashboard Interface--------------------------------------------------------------------------------

#---------------------------------------------Footer---------------------------------------------------------------------------------------

# Uyarı
st.warning("""
🚨 **Önemli Uyarı:**  
Bu uygulama, yalnızca eğitim, araştırma ve demonstrasyon amacıyla geliştirilmiştir.  

Sunulan tahminler, makine öğrenimi algoritmaları tarafından sağlanan *istatistiksel tahminlerdir* ve **herhangi bir şekilde kesin tıbbi teşhis, tedavi önerisi veya profesyonel sağlık hizmeti** olarak yorumlanmamalıdır.

Lütfen bu uygulamadan elde ettiğiniz sonuçları kendi başınıza tıbbi bir karar vermek için kullanmayınız.  
Özellikle sağlık durumunuzla ilgili endişeleriniz varsa, mutlaka **nitelikli bir sağlık profesyoneline veya doktora danışınız**.  

Uygulama geliştiricisi, bu yazılımın kullanımından doğabilecek doğrudan veya dolaylı sonuçlardan **hiçbir şekilde sorumlu değildir**.
""")

#---------------------------------------------Footer---------------------------------------------------------------------------------------