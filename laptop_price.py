# ----- laptop satış verileri ile bir makine öğrenmesi projesi yapacağız. -----

# kullanılacak kütüphaneleri yükleyelim.
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

"""
Bu kısıma aslında gerek yok ama başka projelerde kullanılabilir.Fikir olması için silmeyeceğim.

pd.set_option('display.max_columns',None) # bütün sütunları göster
pd.set_option('display.max_rows',None) # bütün satırları göster
pd.set_option('display.float_format', lambda x: '%.3f' % x) # virgülden sonra 3 basamak göster
pd.set_option('display.width',500) # gösterilen sütunlar 500 ile sınırlansın

"""
df = pd.read_csv("laptop_price.csv", encoding="ISO-8859-1")  # veri setini yükleyelim.
df.head()  # ilk 5 veri ne ?
df.info()  # veri seti hakkında temel bilgiler neler ?
df.isnull().sum()  # eksik veri var mı ?

# aykırı değer analizi yapalım.
sns.boxplot(data=df, x='Price_euros')
plt.title('Fiyat Dağılımı (Aykırı Değerler)')
plt.show()

# aykırı değerleri yakalayalım.
q1 = df['Price_euros'].quantile(0.25)
q3 = df['Price_euros'].quantile(0.75)
iqr = q3-q1
up = q3 + 1.5 * iqr  # üst sınır
low = q1 - 1.5 * iqr  # alt sınır

# aykırı değerler = alt sınırdan küçük olanlar veya üst sınırdan büyük olanlar
aykirilar = df[(df['Price_euros'] < low) | (df['Price_euros'] > up)]
print(aykirilar.sort_values(by='Price_euros', ascending=False))

# aykırı değerlere bakalım.
df[(df['Price_euros'] < low) | (df['Price_euros'] > up)].index

# aykırı değer var mı yok mu ?
df[(df['Price_euros'] < low) | (df['Price_euros'] > up)].any(axis=None)

# ram kategorisi ile işlem yapabilmek için object olan tipini int olarak değiştirelim..
df['Ram'] = df['Ram'].str.replace('GB', "").astype(int)

# weight kategorisinin de tipini değiştirelim.
df['Weight'] = df['Weight'].str.replace('kg', "").astype(float)

df['Memory'].unique()  # benzersiz değişkenlere baktık.
# bu değişkenlerde işlem yapabilmek için bunları biraz düzenleyeceğiz.
df['Memory'].sample(10, random_state=1)  # rastgele 10 tane veriye baktım gözlem yapabilmek için.


# bu fonksiyon ile memorydeki her değeri ayrı ayrı gb cinsinden yeni yerlere ayırabiliriz.
def parse_memory(memory_str):
    ssd = 0
    hdd = 0
    flash = 0
    hybrid = 0

    # '+' ile parçalayalım.
    parts = memory_str.split('+')
    for part in parts:
        part = part.strip()  # boşlukları temizler
        part_lower = part.lower()

        match = re.search(r'(\d+\.?\d*)\s*(gb|tb)', part_lower, re.IGNORECASE)
        if match:
            size = float(match.group(1))  # sayı kısmı
            unit = match.group(2).upper()  # GB veya TB (büyük harfe çevirir.)
            if unit == 'TB':
                size *= 1000  # TB- GB dönüşümü yapar.

        else:
                size = 0  # eğer değer bulunmazsa 0 girer.

        if 'ssd' in part_lower:
                ssd += size

        elif 'hdd' in part_lower:
                hdd += size

        elif 'flash' in part_lower:
                flash += size

        elif 'hybrid' in part_lower:
                hybrid += size

    return {'SSD': ssd, 'HDD': hdd, 'Flash_Storage': flash, 'Hybrid': hybrid}


# fonksiyonumuz istediğimiz gibi mi diye kontrol edelim.
print(parse_memory("256GB SSD + 1TB HDD"))
print(parse_memory("128GB Flash Storage"))
print(parse_memory("512GB SSD"))

# yazdığımız fonksiyonu bütün verilere uygulayalım.
memory_parsed = df['Memory'].apply(parse_memory)

# yeni bir dataframe oluşturalım.
memory_df = pd.DataFrame(memory_parsed.tolist())  # memory parsedin içinde listeler dict olduğu için tolist dedik.

# orijinal dataframe ile birleştirelim.
df = pd.concat([df, memory_df], axis=1)

df.tail()  # sondan 5 değer

# en yüksek ssd ye sahip 5 bilgisayar nedir ?
df.sort_values(by='SSD', ascending=False).head()

# ortalama hdd kapasitesi nedir ?
df['HDD'].mean()

# en yüksek hdd değeri nedir ?
df['HDD'].max()

# en yüksek ssd değeri nedir ?
df['SSD'].max()

# ssd değeri 512 gb dan büyük kaç tane bilgisayar var ?
df[df['SSD'] > 512]

# her tür kaç laptopta var ? tür= ssd,hdd vs
disk_counts = (df[['SSD', 'HDD', 'Flash_Storage', 'Hybrid']] > 0).sum()

plt.figure(figsize=(6, 6))
plt.pie(disk_counts, labels=disk_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Depolama Türlerinin Laptop Sayısına Göre Dağılımı', pad=35)
plt.axis('equal')  # daire şeklinde olması için
plt.show()
# açıklama: ssd veri setimizdeki bilgisayarların yarısndan fazlasında varmış
# daha sonra hdd daha sonra flash ve hybrid geliyor ama flash ve hybrid oranı fazlasıyla düşük.

# markalara göre ssd oranına bakalım.
total_laptop = df['Company'].value_counts()  # toplam laptop sayısı
ssd_counts = df[df['SSD'] > 0]['Company'].value_counts()  # ssd içeren laptop sayısı
ssd_ratio = (ssd_counts/total_laptop) * 100  # ssd oranı
# dataframe haline getirelim ve sıralayalım.
ssd_ratio_df = ssd_ratio.reset_index()
ssd_ratio_df.columns = ['Company', 'SSD Ratio (%)']
ssd_ratio_df = ssd_ratio_df.sort_values(by='SSD Ratio (%)', ascending=False)

# grafiğini çizelim
plt.figure(figsize=(10, 6))
sns.barplot(data=ssd_ratio_df, x='SSD Ratio (%)', y='Company')
plt.title('Markalara Göre SSD Oranı (%) ', fontsize=14)
plt.xlabel('SSD Oranı (%) ')
plt.ylabel('Marka')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# şimdi de hdd için bakalım.
hdd_counts = df[df['HDD'] > 0]['Company'].value_counts()
hdd_ratio = (hdd_counts/total_laptop) * 100
hdd_ratio_df = hdd_ratio.reset_index()
hdd_ratio_df.columns = ['Company', 'HDD Ratio (%)']
hdd_ratio_df = hdd_ratio_df.sort_values(by='HDD Ratio (%)', ascending=False)

# grafiğini çizelim.
plt.figure(figsize=(10, 6))
sns.barplot(data=hdd_ratio_df, x='HDD Ratio (%)', y='Company')
plt.title('Markalara Göre HDD Oranı (%) ', fontsize=14)
plt.xlabel('HDD Oranı (%)')
plt.ylabel('Marka')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Verisetindeki laptop markalarının, toplam satış fiyatından aldığı pay nedir ?
total_price = df['Price_euros'].sum()
company_price_total = df.groupby('Company')['Price_euros'].sum()
company_price_ratio = ((company_price_total/total_price)*100).sort_values(ascending=False)

# Laptoplarda kullanılan işletim sistemlerine göre (OpSys), ortalama SSD kapasitesi nasıldır?
# Hangi işletim sistemi ortalama olarak daha fazla SSD 'ye sahip?
df['OpSys'].unique()  # kaç farklı işletim sistemi var?

# işletim sistemlerine göre ssd ortalaması nasıl?
df.groupby('OpSys')['SSD'].mean()

# büyükten küçüğe sıralayalım
df.groupby('OpSys')['SSD'].mean().sort_values(ascending=False)

# fiyat değişkeninin genel sayısal işlemleri (ortalama, standart sapma , en büyük en küçük değer vs) nedir ?
df['Price_euros'].describe()

# bazı özelliklerin (ram,ssd,marka) fiyat ilişkisi nedir ?
df.groupby(['Ram', 'SSD', 'Company'])['Price_euros'].mean().reset_index()


# ----- özellik mühendisliği ve encoding ----

df.info()  # kategorik ve sayısal değişkenlere bakalım.
# model için gerekli sütunlar:
# kategorik(one-hot yapılacak: kategorik değişkenler sayısala dönüşecek ):Company, TypeName, Cpu, Gpu
# (her kategori için ayrı sütun oluşturulur ve ilgili satırda 1, diğerlerinde 0)
# Sayısal (Scaler yapılacak): Ram, SSD, HDD, Flash_Storage, Hybrid

# 1. ADIM: Önce y'yi tanımlayalım (Price_euros'u alalım)
y = df['Price_euros']

# 2. ADIM: Sonra X'i oluşturalım (Price_euros ve diğer istemediklerimizi çıkaralım)
X = df.drop(columns=['Price_euros', 'Product', 'laptop_ID', 'ScreenResolution', 'OpSys', 'Memory'])

# 3. ADIM: Feature engineering yapalım
X['Cpu_brand'] = X['Cpu'].apply(lambda x: x.split()[0])
X['Gpu_brand'] = X['Gpu'].apply(lambda x: x.split()[0])
X = X.drop(columns=['Cpu', 'Gpu'])

# 4. ADIM: One-hot encoding
categorical_cols = ['Company', 'TypeName', 'Cpu_brand', 'Gpu_brand']
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# 5. ADIM: Train-test ayıralım
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 6. ADIM: Scaling (ölçeklendirme) yapalım tüm sayısal değişkenleri aynı ölçeğe getirelim.
num_col = ['Ram', 'SSD', 'HDD', 'Flash_Storage', 'Hybrid', 'Weight', 'Inches']
scaler = StandardScaler()
X_train[num_col] = scaler.fit_transform(X_train[num_col])  # hem öğren hem uygula
X_test[num_col] = scaler.transform(X_test[num_col])  # sadece uygula öğrenme !

# 7. ADIM: Modeli eğitelim ve test edelim
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 8. ADIM: Sonuçları değerlendirelim
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print('RMSE:', rmse)
print("R²:", r2)

# özelliklerin önem sırası
feature_importance = pd.DataFrame({
    'Özellik': X_encoded.columns,
    'Katsayı': model.coef_,
    'Önem(Mutlak Değer)': abs(model.coef_)})

# öneme göre sıralama yapalım
feature_importance = feature_importance.sort_values('Önem(Mutlak Değer)', ascending=False)
print('En önemli 10 özellik :')
print(feature_importance.head(10))

# görselleştirilmesi
plt.figure(figsize=(12, 8))
feature_importance.head(15).sort_values('Katsayı').plot(y='Katsayı', x='Özellik', kind='barh', color='skyblue')
plt.title('En Önemli 15 Özelliğin Fiyat Üzerindeki Etkisi')
plt.xlabel('Katsayı Değeri (Pozitif: Fiyatı Artırır, Negatif: Azaltır)')
plt.tight_layout()
plt.show()

# cross-validation ile model güvenirliğine bakalım.
cv_scores = cross_val_score(model, X_encoded, y, cv=5, scoring='r2')
cv_rms_scores = - cross_val_score(model, X_encoded, y, cv=5, scoring='neg_root_mean_squared_error')

print('CROSS- VALIDATION SONUÇLARI :')
print('R² Scores:', cv_scores)
print('Ortalama R²:', cv_scores.mean())
print('RMSE Score: ', cv_rms_scores)
print('Ortalama RMSE: ', cv_rms_scores.mean())
print('Standart Sapma: ', cv_scores.std())

# Cross-validation sonuçlarını görselleştirelim
plt.figure(figsize=(10, 6))
plt.bar(range(1, 6), cv_scores, color='lightblue', alpha=0.7)
plt.axhline(y=cv_scores.mean(), color='red', linestyle='--', label=f'Ortalama: {cv_scores.mean(): 3f}')
plt.xlabel('Fold_Number')
plt.ylabel('R² Score')
plt.title('5-Fold Cross Validation Sonuçları:')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# --- farklı modellerle denemeler ---

# R² =  Modelin veriyi ne kadar açıkladığını gösteriyor.0-1 arası değerler alır.
# 1'e ne kadar yakınsa o kadar iyi sonuç alınmış demektir.
# RMSE : Hata ölçüsüdür, yani tahmin edilen değer ile gerçek değer arasındaki ortalama fark.
# Değerin küçük olması daha iyidir.

# deneyeceğimiz modeller
models = {'Lineer Regression': LinearRegression(),
          'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
          'Gradient Boosting': GradientBoostingRegressor(random_state=42),
          'Decision Tree': DecisionTreeRegressor(random_state=42),
          'SVR': SVR(kernel='rbf')}

# sonuçları saklayacağımız dictionary
results = {}

print('Farklı Modelleri Test Ediyoruz...\n')
for name, model in models.items():
    # modeli eğitelim
    model.fit(X_train, y_train)
    # tahmin yapalım
    y_pred = model.predict(X_test)
    # Metrikleri hesaplayalım
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # sonuçları kaydedelim
    results[name] = {'R²': r2, 'RMSE': rmse}

    print(f'{name}: ')
    print(f'R² ={r2: .4f}')
    print(f'RMSE = {rmse: .2f} €')
    print('--' * 40)

# sonuçları karşılaştıralım
print('\n MODEL KARŞILAŞTIRMASI:')
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('R²', ascending=False)
print(results_df)

# --- model kıyas açıklamaları ---
# modeller arasında kıyas yaparken baktığımız iki şey var :
# 1) R² yüksek olsun(daha fazla varyansı açıklasın).2) RMSE düşük olsun(daha az hata yapsın).
# o yüzden en iyi model: -- random forest --
# gradient boosting de iyi ama RF kadar değil.SVR bu veriseti için iyi bir seçenek değilmiş.

# modelleri grafik ile görelim.
# sonuçları grafik için uygun formata getirelim.
results_df.info()  # sonuçların olduğu df e baktık.
results_plot = results_df.reset_index()  # indexi sütun yapacağız. yani model isimlerini
results_plot = results_plot.rename(columns={'index': 'Model'})  # index sütununa model adını verdik.
print(results_plot)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 1. grafik: R²
sns.barplot(data=results_plot, x='Model', y='R²', ax=axes[0], palette='Blues_d')
axes[0].set_title('Modellere Göre R² Skorları')
axes[0].set_ylabel('R² Değerleri')

# 2. grafik: RMSE
sns.barplot(data=results_plot, x='Model', y='RMSE', ax=axes[1], palette='Reds_d')
axes[1].set_title('Modellere Göre RMSE Skorları')
axes[1].set_ylabel('RMSE Değerleri')

plt.tight_layout()
plt.show()

# tahmin verisi eklemeden önce sütunlara bakalım.
df.columns

# 1) modeli oluşturalım ve eğitelim
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 2) yeni laptop verisi oluşturalım
new_laptop = pd.DataFrame([{
    'Company': 'Asus',
    'TypeName': 'Gaming',
    'Inches': 17.3,
    'Cpu': 'Intel Core i7',
    'Ram': 32,
    'Gpu': 'Nvidia GeForce GTX',
    'SSD': 1000,
    'HDD': 0,
    'Flash_Storage': 0,
    'Hybrid': 0,
    'Weight': 3.5}])

# 3) CPU/GPU marka çıkarımı yapalım
new_laptop['Cpu_brand'] = new_laptop['Cpu'].apply(lambda x: x.split()[0])
new_laptop['Gpu_brand'] = new_laptop['Gpu'].apply(lambda x: x.split()[0])
new_laptop = new_laptop.drop(['Cpu', 'Gpu'], errors='ignore')

# 4) One-hot encoding (eğitim verisine uyumlu) yapalım
categorical_cols = ['Company', 'TypeName', 'Cpu_brand', 'Gpu_brand']
new_laptop_enc = pd.get_dummies(new_laptop, columns=categorical_cols, drop_first=True)

# Eğitimdeki sütunlarla hizalayalım
new_laptop_enc = new_laptop_enc.reindex(columns=X_train.columns, fill_value=0)

# 5) Sayısal sütunları ölçeklendirelim
num_cols = ['Ram', 'SSD', 'HDD', 'Flash_Storage', 'Hybrid', 'Weight', 'Inches']
new_laptop_enc[num_cols] = scaler.transform(new_laptop_enc[num_cols])

# 6) Tahmin
predicted_price = rf_model.predict(new_laptop_enc)[0]

# 7) Tahmin sonuçlarını yazdıralım.
print(' YENİ LAPTOP FİYAT TAHMİNİ')
print('=' * 50)
for col, val in new_laptop.iloc[0].items():
    print(f'- {col}: {val}')
print('=' * 50)
print(f'TAHMİNİ FİYAT: {predicted_price: .2f} €')
print('=' * 50)
