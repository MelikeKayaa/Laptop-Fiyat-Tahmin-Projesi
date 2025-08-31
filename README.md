# Laptop Fiyat Tahmin Projesi 💻

Bu proje, farklı laptop özelliklerine göre satış fiyatlarını tahmin etmek için bir makine öğrenmesi modeli oluşturmayı amaçlamaktadır. Python ve popüler veri bilimi kütüphaneleri kullanılarak geliştirilmiştir.

---

## 📝 Proje Amacı

Laptopların özellikleri (marka, RAM, depolama türü, ekran boyutu, işlemci ve GPU gibi) kullanılarak **fiyat tahmini yapmak**.  Bu proje, özellikle veri analizi, özellik mühendisliği ve makine öğrenmesi modellerinin uygulamalı kullanımını göstermektedir.

---

## 📂 Veri Seti

- **Dosya Adı:** `laptop_price`  
- **Kaynak:** Kaggle  
- **İçerik:** Laptop markaları, modeli, teknik özellikleri ve satış fiyatları (Euro cinsinden).  

**Özellikler:**
- `Company` → Laptop markası  
- `TypeName` → Laptop türü (Notebook, Ultrabook, Gaming vs.)  
- `Inches` → Ekran boyutu  
- `Cpu` / `Gpu` → İşlemci ve grafik kartı bilgileri  
- `Ram` → Bellek miktarı  
- `Memory` → Depolama bilgisi (SSD, HDD, Flash, Hybrid)  
- `Weight` → Laptop ağırlığı  
- `Price_euros` → Satış fiyatı  

---

## Kullanılan Kütüphaneler

- `pandas`, `numpy` → Veri işleme  
- `matplotlib`, `seaborn` → Görselleştirme  
- `sklearn` → Makine öğrenmesi ve model değerlendirme  
- `missingno` → Eksik veri analizi  
- `re` → Düzenli ifadeler ile veri temizleme  

---

## Veri Ön İşleme

1. **Eksik verilerin kontrolü:** `df.isnull().sum()`  
2. **Aykırı değer analizi:** Boxplot ve IQR yöntemi ile fiyat değişkenindeki uç değerler incelendi.  
3. **Tip dönüşümleri:**  
   - `Ram` → int (GB)  
   - `Weight` → float (kg)  
4. **Memory sütunlarının ayrıştırılması:** SSD, HDD, Flash Storage ve Hybrid değerleri ayrı sütunlara bölündü.  
5. **Depolama türleri ve markalar analizi:** SSD, HDD, Flash ve Hybrid dağılımları görselleştirildi.  

---

## Görselleştirmeler

- Fiyat dağılımı (aykırı değerler dahil)

- Markalara göre SSD ve HDD oranları

- Özelliklerin fiyat üzerindeki etkisi

- Modellerin R² ve RMSE karşılaştırmaları

---

## Özellik Mühendisliği (Feature Engineering)

1. Hedef değişken (`y`) olarak `Price_euros` seçildi.  
2. Bağımsız değişkenler (`X`) olarak fiyatı etkileyecek sütunlar seçildi.  
3. CPU ve GPU markaları çıkarıldı ve `Cpu_brand`, `Gpu_brand` sütunları oluşturuldu.  
4. Kategorik değişkenler için **one-hot encoding** uygulandı (`Company`, `TypeName`, `Cpu_brand`, `Gpu_brand`).  
5. Sayısal sütunlar (`Ram`, `SSD`, `HDD`, `Flash_Storage`, `Hybrid`, `Weight`, `Inches`) **StandardScaler** ile ölçeklendirildi.  

---

## Modeller ve Değerlendirme

Kullanılan modeller:  

- **Linear Regression**  
- **Random Forest Regressor**  
- **Gradient Boosting Regressor**  
- **Decision Tree Regressor**  
- **SVR (Support Vector Regressor)**  

**Performans ölçümleri:**  
- `R²` → Modelin veri üzerindeki açıklama gücü  
- `RMSE` → Tahmin hatası  

**Sonuçlar:**  
- En iyi performans **Random Forest** modeli tarafından elde edildi.  
- Modellerin R² ve RMSE değerleri grafiklerle karşılaştırıldı.  

---

## 💻 Yeni Laptop Fiyat Tahmini

Yeni bir laptop için fiyat tahmini adımları:  

1. Yeni laptop verisi oluşturuldu (marka, RAM, CPU, GPU, depolama vb.)  
2. CPU/GPU markası çıkarıldı  
3. One-hot encoding uygulandı ve eğitim verisi ile hizalandı  
4. Sayısal sütunlar ölçeklendirildi  
5. Random Forest modeli ile tahmin yapıldı  

**Örnek Çıktı:**

```text
🎯 YENİ LAPTOP FİYAT TAHMİNİ
==================================================
- Company: Asus
- TypeName: Gaming
- Inches: 17.3
- Ram: 32
- SSD: 1000
- HDD: 0
- Flash_Storage: 0
- Hybrid: 0
- Weight: 3.5
- Cpu_brand: Intel
- Gpu_brand: Nvidia
==================================================
TAHMİNİ FİYAT: 3182.13 €
==================================================

---

## Kullanım

1. laptop_price.csv dosyasını proje dizinine koyun

2. Kodları çalıştırın, sonuçlar ve tahminler otomatik olarak üretilecektir

3. Yeni laptop tahmini yapmak için new_laptop DataFrame’ini güncelleyebilirsiniz.

---

## Özet

Bu proje sayesinde:

- Veri ön işleme ve eksik/aykırı değer analizi yapılmıştır

- Özellik mühendisliği ve one-hot encoding uygulanmıştır

- Birden fazla regresyon modeli ile fiyat tahmini gerçekleştirilmiştir

- Random Forest ile başarılı tahmin sonuçları elde edilmiştir

