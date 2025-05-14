## Proje Kodlarının Detaylı Açıklaması (`prepare_artifacts.py` ve `app.py`)

Bu bölümde, projenizin kalbi olan iki ana Python betiğinin (`prepare_artifacts.py` ve `app.py`) kodlarını adım adım ve kolay anlaşılır bir şekilde inceleyeceğiz. Bu sayede, hocanıza sunum yaparken kodların nasıl çalıştığını kendinizden emin bir şekilde anlatabileceksiniz.

### 1. Model Hazırlık Betiği: `prepare_artifacts.py`

Bu betiğin temel amacı, ham müşteri verilerini alıp, makine öğrenimi modelini eğitmek ve bu modelin web uygulaması tarafından kullanılabilmesi için gerekli tüm "artifact"ları (yan ürünleri/dosyaları) hazırlayıp kaydetmektir. Bu betik bir kere çalıştırılır ve model ile diğer yardımcı dosyalar oluşturulur.

```python
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
```

*   **`import pandas as pd`**: `pandas` kütüphanesi, verileri tablolar (DataFrame) halinde kolayca işlemek, okumak ve analiz etmek için kullanılır. `pd` kısaltmasıyla çağrılır.
*   **`import xgboost as xgb`**: `xgboost` kütüphanesi, projemizde kullanılan XGBoost sınıflandırma modelini içerir. `xgb` kısaltmasıyla çağrılır.
*   **`from sklearn.model_selection import train_test_split`**: `scikit-learn` (sklearn) kütüphanesinden `train_test_split` fonksiyonu, veri setini model eğitimi ve model testi için iki ayrı parçaya bölmek amacıyla kullanılır.
*   **`from sklearn.preprocessing import StandardScaler`**: Yine `scikit-learn` kütüphanesinden `StandardScaler`, sayısal verilerin ölçeklenmesi (standartlaştırılması) için kullanılır.
*   **`import joblib`**: `joblib` kütüphanesi, Python nesnelerini (bu projede eğitilmiş model ve scaler gibi) dosya olarak kaydetmek ve daha sonra tekrar yüklemek için kullanılır. Büyük veri içeren nesneler için `pickle` kütüphanesine göre daha verimlidir.
*   **`import os`**: `os` kütüphanesi, işletim sistemiyle ilgili işlemler yapmak için kullanılır; örneğin, dosya yolları oluşturmak veya klasörlerin varlığını kontrol etmek gibi.

```python
# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model_artifacts")
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_churn_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
COLUMNS_PATH = os.path.join(MODEL_DIR, "model_columns.json")
TOTAL_CHARGES_MEAN_PATH = os.path.join(MODEL_DIR, "total_charges_mean.json")
NUMERICAL_COLS_PATH = os.path.join(MODEL_DIR, "numerical_cols.json")
```

*   Bu blokta, proje içinde kullanılacak önemli dosya ve klasörlerin yolları tanımlanır.
*   **`BASE_DIR = os.path.dirname(os.path.abspath(__file__))`**: Bu satır, `prepare_artifacts.py` betiğinin bulunduğu klasörün tam yolunu alır. `__file__` o an çalışan dosyanın adını verir, `os.path.abspath()` tam yolunu bulur, `os.path.dirname()` ise bu tam yoldan klasör kısmını alır.
*   **`DATA_FILE = os.path.join(BASE_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")`**: Ham veri setinin (`.csv` dosyası) tam yolunu oluşturur. `os.path.join()` farklı işletim sistemlerinde doğru şekilde yol birleştirmesi yapar.
*   **`MODEL_DIR = os.path.join(BASE_DIR, "model_artifacts")`**: Eğitilmiş model ve diğer yardımcı dosyaların kaydedileceği `model_artifacts` klasörünün yolunu tanımlar.
*   Diğer `_PATH` değişkenleri de benzer şekilde, bu `MODEL_DIR` içinde saklanacak belirli dosyaların (model dosyası, scaler dosyası, sütun listesi dosyası vb.) tam yollarını belirtir.

```python
os.makedirs(MODEL_DIR, exist_ok=True)
```

*   **`os.makedirs(MODEL_DIR, exist_ok=True)`**: `model_artifacts` klasörünü oluşturur. `exist_ok=True` parametresi, eğer klasör zaten varsa bir hata vermemesini, sessizce devam etmesini sağlar.

```python
# Load the dataset
df = pd.read_csv(DATA_FILE)
```

*   **`df = pd.read_csv(DATA_FILE)`**: `pandas` kütüphanesinin `read_csv` fonksiyonu kullanılarak, daha önce yolu tanımlanan `DATA_FILE` (yani `WA_Fn-UseC_-Telco-Customer-Churn.csv`) okunur ve içeriği `df` adlı bir DataFrame'e (veri tablosuna) yüklenir.

```python
# --- Preprocessing --- 
# Handle TotalCharges: Convert to numeric, coercing errors, then fill NaNs with the mean
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
total_charges_mean = df["TotalCharges"].mean()
df["TotalCharges"].fillna(total_charges_mean, inplace=True)
joblib.dump({"total_charges_mean": total_charges_mean}, TOTAL_CHARGES_MEAN_PATH)
```

*   Bu blok, veri ön işleme adımlarından ilkini gerçekleştirir: `TotalCharges` sütunundaki sorunları gidermek.
*   **`df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")`**: `TotalCharges` sütunundaki değerler metin olarak gelmiş olabilir (örneğin, içinde boşluklar olabilir). Bu satır, bu sütundaki tüm değerleri sayısal (numeric) tipe dönüştürmeye çalışır. `errors="coerce"` parametresi, eğer bir değer sayıya dönüştürülemiyorsa (örneğin, tamamen boş bir metinse), o değeri `NaN` (Not a Number - Sayı Değil, yani eksik değer) olarak işaretler.
*   **`total_charges_mean = df["TotalCharges"].mean()`**: Sayısala dönüştürülmüş `TotalCharges` sütunundaki tüm geçerli değerlerin ortalamasını hesaplar ve `total_charges_mean` değişkenine atar.
*   **`df["TotalCharges"].fillna(total_charges_mean, inplace=True)`**: `TotalCharges` sütunundaki tüm `NaN` (eksik) değerleri, bir önceki adımda hesaplanan `total_charges_mean` (ortalama) değeri ile doldurur. `inplace=True` parametresi, bu değişikliğin doğrudan `df` DataFrame'i üzerinde yapılmasını sağlar.
*   **`joblib.dump({"total_charges_mean": total_charges_mean}, TOTAL_CHARGES_MEAN_PATH)`**: Hesaplanan bu ortalama `TotalCharges` değerini, daha sonra web uygulamasında da kullanabilmek için `total_charges_mean.json` adlı bir dosyaya kaydeder. `{}` içinde bir sözlük (dictionary) olarak kaydedilir, böylece anahtar-değer çifti şeklinde erişilebilir.

```python
# Identify categorical columns for one-hot encoding (excluding customerID for now, will be dropped)
original_categorical_cols = df.select_dtypes(include="object").columns.tolist()
if "customerID" in original_categorical_cols:
    original_categorical_cols.remove("customerID")
```

*   Bu blok, makine öğrenimi modelinin anlayabilmesi için sayısal formata dönüştürülmesi gereken kategorik (metin tabanlı) sütunları belirler.
*   **`original_categorical_cols = df.select_dtypes(include="object").columns.tolist()`**: `df` DataFrame'indeki veri tipi `object` (genellikle metin veya karışık tipteki veriler için kullanılır) olan tüm sütunların isimlerini seçer ve bir liste (`original_categorical_cols`) haline getirir.
*   **`if "customerID" in original_categorical_cols: original_categorical_cols.remove("customerID")`**: `customerID` sütunu da metin içerir ancak bu bir tanımlayıcıdır ve modelde özellik olarak kullanılmayacaktır (daha sonra veri setinden atılacaktır). Bu nedenle, eğer `customerID` bu kategorik sütunlar listesindeyse, listeden çıkarılır.

```python
# One-hot encode categorical features
df = pd.get_dummies(df, columns=original_categorical_cols, drop_first=True)
```

*   **`df = pd.get_dummies(df, columns=original_categorical_cols, drop_first=True)`**: Bu çok önemli bir adımdır. `pandas`'ın `get_dummies` fonksiyonu, belirtilen `original_categorical_cols` listesindeki tüm kategorik sütunlara "one-hot encoding" uygular.
    *   **One-hot encoding nedir?** Örneğin, "InternetService" sütununda "DSL", "Fiber optic", "No" gibi 3 farklı değer varsa, `get_dummies` bu tek sütunu `InternetService_DSL`, `InternetService_Fiber optic`, `InternetService_No` gibi 3 yeni sütuna dönüştürür. Bir müşteri için hangi kategori geçerliyse o sütuna 1, diğerlerine 0 değeri atanır. Bu, modelin kategorik bilgiyi sayısal olarak işlemesini sağlar.
    *   **`drop_first=True`**: Bu parametre, one-hot encoding yaparken her bir orijinal kategorik sütun için oluşturulan yeni dummy (kukla) sütunlardan ilkini atar. Örneğin, "Gender" (Cinsiyet) sütununda "Male" ve "Female" varsa, `Gender_Male` ve `Gender_Female` yerine sadece `Gender_Male` (veya `Gender_Female`) sütunu oluşturulur. Eğer `Gender_Male` 0 ise, müşterinin Female olduğu anlaşılır. Bu, "dummy variable trap" (kukla değişken tuzağı) denilen ve modelde gereksiz bağımlılıklara yol açabilen bir durumu önlemek için yapılır.

```python
# Define target and features
TARGET_COL = None
if "Churn_Yes" in df.columns:
    TARGET_COL = "Churn_Yes"
elif "Churn" in df.columns and df["Churn"].dtype != 'object':
    TARGET_COL = "Churn"
else:
    possible_target_cols = [col for col in df.columns if "Churn" in col and (df[col].dtype == 'bool' or df[col].dtype == 'int')]
    if possible_target_cols:
        TARGET_COL = possible_target_cols[0]
    else:
        raise ValueError("Could not identify the binary target column for Churn after dummification.")

X = df.drop(columns=[TARGET_COL, "customerID"])
y = df[TARGET_COL]
```

*   Bu blok, modelin neyi tahmin edeceğini (hedef değişken, `y`) ve hangi bilgileri kullanarak tahmin yapacağını (özellikler, `X`) tanımlar.
*   **`TARGET_COL = None ... raise ValueError(...)`**: Bu kısım, `Churn` (müşteri kaybı) bilgisini içeren hedef sütunun adını bulmaya çalışır. `get_dummies` işlemi `Churn` sütununu da etkilemiş olabilir (örneğin, `Churn_Yes` gibi bir sütun oluşmuş olabilir). Kod, `Churn_Yes` adında bir sütun varsa onu hedef olarak alır. Yoksa, ve `Churn` adında sayısal (object olmayan) bir sütun varsa onu alır. Bunlar da yoksa, içinde 'Churn' geçen ve boolean (doğru/yanlış) veya integer (tamsayı) tipinde olan bir sütun arar. Hiçbiri bulunamazsa hata verir.
*   **`X = df.drop(columns=[TARGET_COL, "customerID"])`**: `X` (özellikler matrisi), ana DataFrame `df`'ten hedef sütun (`TARGET_COL`) ve `customerID` sütunu çıkarılarak oluşturulur. Geriye kalan tüm sütunlar, modelin churn tahminini yaparken kullanacağı giriş bilgileridir.
*   **`y = df[TARGET_COL]`**: `y` (hedef değişken vektörü), sadece `TARGET_COL` sütununu içerir. Bu, modelin tahmin etmeyi öğreneceği değerlerdir (müşteri churn etti mi, etmedi mi).

```python
# Save the column order and names for the Flask app
model_columns = X.columns.tolist()
joblib.dump(model_columns, COLUMNS_PATH)
```

*   **`model_columns = X.columns.tolist()`**: Özellik matrisi `X`'teki sütunların isimlerini ve sıralarını bir liste olarak alır.
*   **`joblib.dump(model_columns, COLUMNS_PATH)`**: Bu sütun listesini (`model_columns.json` dosyasına) kaydeder. Bu çok önemlidir, çünkü web uygulaması yeni bir müşterinin tahminini yaparken, gelen veriyi tam olarak modelin eğitildiği sıradaki ve sayıdaki sütunlara göre düzenlemek zorundadır. Bu dosya bu tutarlılığı sağlar.

```python
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

*   **`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)`**: `train_test_split` fonksiyonu kullanılarak `X` (özellikler) ve `y` (hedef) veri setleri, eğitim ve test setleri olarak ikiye ayrılır.
    *   `X_train`: Eğitim için kullanılacak özellikler.
    *   `X_test`: Test için kullanılacak özellikler.
    *   `y_train`: Eğitim için kullanılacak hedef değerler.
    *   `y_test`: Test için kullanılacak hedef değerler.
    *   **`test_size=0.2`**: Verinin %20'sinin test seti, kalan %80'inin ise eğitim seti olarak ayrılacağını belirtir.
    *   **`random_state=42`**: Bu parametre, bölme işleminin her çalıştırıldığında aynı şekilde yapılmasını sağlar (rastgeleliği sabitler). Bu, sonuçların tekrarlanabilir olması için önemlidir. `42` sıkça kullanılan rastgele bir sayıdır, herhangi bir özel anlamı yoktur.

```python
# Identify numerical columns for scaling
original_numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen'] 
numerical_cols_to_scale = []
for col in original_numeric_cols:
    if col in X_train.columns:
        numerical_cols_to_scale.append(col)

if 'SeniorCitizen' not in numerical_cols_to_scale and 'SeniorCitizen' in X_train.columns:
     numerical_cols_to_scale.append('SeniorCitizen')

numerical_cols_to_scale = [col for col in numerical_cols_to_scale if X_train[col].dtype in ['int64', 'float64']]
joblib.dump(numerical_cols_to_scale, NUMERICAL_COLS_PATH)
```

*   Bu blok, hangi sayısal sütunların ölçekleneceğini belirler.
*   **`original_numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']`**: Orijinal veri setindeki temel sayısal sütunların bir listesi tanımlanır. `SeniorCitizen` 0 veya 1 değerlerini alsa da, burada sayısal olarak ele alınıp ölçeklenmesi tercih edilmiş.
*   **`for col in original_numeric_cols: ...`**: Bu döngü, tanımlanan orijinal sayısal sütunların, one-hot encoding sonrası `X_train`'de hala var olup olmadığını kontrol eder ve varsa `numerical_cols_to_scale` listesine ekler.
*   **`if 'SeniorCitizen' not in ...`**: `SeniorCitizen`'ın kesinlikle listede olduğundan emin olmak için ek bir kontrol.
*   **`numerical_cols_to_scale = [col for col ...]`**: Son olarak, bu listedeki sütunların gerçekten sayısal tipte (`int64` veya `float64`) olup olmadığı bir kez daha kontrol edilir.
*   **`joblib.dump(numerical_cols_to_scale, NUMERICAL_COLS_PATH)`**: Ölçeklenecek sayısal sütunların bu son listesi (`numerical_cols.json` dosyasına) kaydedilir. Web uygulaması, yeni müşteri verisini ölçeklerken hangi sütunları ölçekleyeceğini bu dosyadan okuyacaktır.

```python
# Scale numerical features
scaler = StandardScaler()
if numerical_cols_to_scale:
    X_train[numerical_cols_to_scale] = scaler.fit_transform(X_train[numerical_cols_to_scale])
    X_test[numerical_cols_to_scale] = scaler.transform(X_test[numerical_cols_to_scale])

# Save the scaler
joblib.dump(scaler, SCALER_PATH)
```

*   Bu blok, belirlenen sayısal özellikleri ölçekler.
*   **`scaler = StandardScaler()`**: `StandardScaler` nesnesi oluşturulur. Bu nesne, veriyi ortalaması 0 ve standart sapması 1 olacak şekilde dönüştürecektir.
*   **`if numerical_cols_to_scale:`**: Eğer ölçeklenecek sayısal sütun varsa devam eder.
*   **`X_train[numerical_cols_to_scale] = scaler.fit_transform(X_train[numerical_cols_to_scale])`**: `StandardScaler`'ın `fit_transform` metodu, *sadece eğitim verisi (`X_train`)* üzerinde çağrılır. `fit` kısmı, `X_train`'deki belirtilen sütunların ortalamasını ve standart sapmasını hesaplar (öğrenir). `transform` kısmı ise bu öğrenilen değerleri kullanarak `X_train`'deki sütunları ölçekler.
*   **`X_test[numerical_cols_to_scale] = scaler.transform(X_test[numerical_cols_to_scale])`**: Test verisi (`X_test`) ise *sadece `transform`* metodu ile ölçeklenir. Burada `fit` kullanılmaz, çünkü test verisinin bilgilerini (ortalama, standart sapma) modele sızdırmamamız gerekir. Test verisi, eğitim verisinden öğrenilen ölçekleme parametreleriyle dönüştürülmelidir.
*   **`joblib.dump(scaler, SCALER_PATH)`**: Eğitilmiş (yani ortalama ve standart sapmaları öğrenmiş) `scaler` nesnesi, `scaler.joblib` dosyasına kaydedilir. Web uygulaması, yeni müşteri verisini ölçeklerken bu kaydedilmiş scaler'ı kullanacaktır.

```python
# --- Model Training --- 
# Initialize and train the XGBoost classifier
xgb_major_version = int(xgb.__version__.split('.')[0])
if xgb_major_version < 2:
    model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", use_label_encoder=False)
else:
    model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss")

model.fit(X_train, y_train)
```

*   Bu blok, XGBoost modelini kurar ve eğitir.
*   **`xgb_major_version = int(xgb.__version__.split('.')[0]) ... else: ...`**: Bu kısım, kurulu olan XGBoost kütüphanesinin sürümünü kontrol eder. XGBoost'un eski (2.0 öncesi) ve yeni sürümlerinde `XGBClassifier` başlatılırken kullanılan bazı parametreler farklılık gösterebilir. `use_label_encoder` parametresi eski sürümlerde bazen uyarıya neden olabiliyordu, bu yüzden sürüm kontrolü ile uygun şekilde model başlatılır.
    *   **`objective="binary:logistic"`**: Modelin amacının ikili sınıflandırma (churn edecek / etmeyecek) olduğunu ve lojistik regresyon tabanlı bir kayıp fonksiyonu kullanacağını belirtir.
    *   **`eval_metric="logloss"`**: Modelin performansını değerlendirmek için kullanılacak metrik olarak "logaritmik kayıp" (logloss) belirtilir.
*   **`model.fit(X_train, y_train)`**: Modelin asıl eğitildiği yer burasıdır. `fit` metodu, `X_train` (eğitim özellikleri) ve `y_train` (eğitim hedefleri) verilerini kullanarak modelin iç parametrelerini ayarlar ve müşteri özellikleriyle churn durumu arasındaki ilişkiyi öğrenir.

```python
# Save the trained model
joblib.dump(model, MODEL_PATH)
```

*   **`joblib.dump(model, MODEL_PATH)`**: Eğitilmiş olan `model` nesnesi, `xgboost_churn_model.joblib` dosyasına kaydedilir. Web uygulamasının kalbi bu dosyadır.

```python
print(f"Model saved to {MODEL_PATH}")
# ... (diğer print mesajları)

# (Optional) Evaluate the model to ensure consistency with notebook
from sklearn.metrics import accuracy_score, classification_report
y_pred_test = model.predict(X_test)
print("\nModel Evaluation on Test Set (from prepare_artifacts.py):")
print(f"Accuracy: {accuracy_score(y_test, y_pred_test):.2f}")
print(classification_report(y_test, y_pred_test))
```

*   Son kısımda, kaydedilen dosyaların yolları ekrana basılır.
*   Ardından, isteğe bağlı olarak, eğitilen modelin test seti (`X_test`, `y_test`) üzerindeki performansı değerlendirilir. `accuracy_score` (doğruluk) ve `classification_report` (hassasiyet, duyarlılık, F1-skoru gibi daha detaylı metrikler) hesaplanıp ekrana yazdırılır. Bu, modelin ne kadar iyi öğrendiğine dair bir fikir verir.

`prepare_artifacts.py` betiği tamamlandığında, `model_artifacts` klasörü içinde web uygulamasının ihtiyaç duyacağı tüm dosyalar hazır olmuş olur.




### 2. Web Uygulaması Betiği: `app.py`

Bu betik, kullanıcıların etkileşimde bulunduğu web uygulamasını çalıştırır. `Flask` adlı bir Python web çatısı kullanılarak yazılmıştır. Kullanıcıdan bir müşteri ID'si alır, bu müşterinin verilerini işler, eğitilmiş modeli kullanarak churn tahminini yapar ve sonucu kullanıcıya gösterir.

```python
import flask
import pandas as pd
import joblib
import os
import numpy as np
```

*   **`import flask`**: `Flask` kütüphanesini içe aktarır. Web uygulamasını oluşturmak ve yönetmek için temel araçları sağlar.
*   **`import pandas as pd`**: `pandas` kütüphanesi, özellikle müşteri verilerini DataFrame olarak işlemek ve `preprocess_single_customer` fonksiyonunda kullanmak için gereklidir.
*   **`import joblib`**: `prepare_artifacts.py` tarafından kaydedilen eğitilmiş modeli, scaler'ı ve diğer yardımcı dosyaları (`.joblib` ve `.json` uzantılı) yüklemek için kullanılır.
*   **`import os`**: Dosya yollarını yönetmek ve `model_artifacts` klasörünün yerini bulmak için kullanılır.
*   **`import numpy as np`**: `numpy` kütüphanesi, sayısal işlemler için kullanılır. Bu projede özellikle `preprocess_single_customer` fonksiyonunda veri tipleriyle veya eksik değerlerle çalışırken dolaylı olarak `pandas` tarafından kullanılabilir veya doğrudan sayısal dönüşümler için gerekebilir (bu kodda doğrudan `np` kullanımı belirgin olmasa da, `pandas`'ın birçok fonksiyonu arka planda `numpy` kullanır).

```python
app = flask.Flask(__name__)
```

*   **`app = flask.Flask(__name__)`**: Yeni bir Flask web uygulaması örneği (instance) oluşturur. `__name__` özel bir Python değişkenidir ve Flask'a uygulamanın kök yolunu bulmasında yardımcı olur. `app` değişkeni artık bizim web uygulamamızı temsil eder.

```python
# --- Configuration and Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model_artifacts")
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_churn_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
MODEL_COLUMNS_PATH = os.path.join(MODEL_DIR, "model_columns.json")
TOTAL_CHARGES_MEAN_PATH = os.path.join(MODEL_DIR, "total_charges_mean.json")
NUMERICAL_COLS_PATH = os.path.join(MODEL_DIR, "numerical_cols.json")
ORIGINAL_DATA_PATH = os.path.join(BASE_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
```

*   Bu blok, `prepare_artifacts.py` dosyasındakine çok benzer şekilde, uygulamanın ihtiyaç duyduğu dosyaların ve klasörlerin yollarını tanımlar.
*   **`BASE_DIR`**: `app.py` dosyasının bulunduğu klasörün yolu.
*   **`MODEL_DIR`**: `model_artifacts` klasörünün yolu.
*   Diğer `_PATH` değişkenleri: Eğitilmiş modelin, scaler'ın, model sütunlarının, `TotalCharges` ortalamasının, ölçeklenecek sayısal sütunların listesinin ve orijinal veri setinin (`.csv` dosyası) tam yollarını belirtir. Orijinal veri setine, kullanıcı tarafından girilen müşteri ID'sine ait ham verileri çekmek için ihtiyaç duyulur.

```python
# --- Load Artifacts --- 
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    model_columns = joblib.load(MODEL_COLUMNS_PATH)
    total_charges_mean_data = joblib.load(TOTAL_CHARGES_MEAN_PATH)
    total_charges_mean = total_charges_mean_data["total_charges_mean"]
    numerical_cols_to_scale = joblib.load(NUMERICAL_COLS_PATH)
    original_df = pd.read_csv(ORIGINAL_DATA_PATH)
except Exception as e:
    print(f"Error loading artifacts: {e}")
    model = None
```

*   Bu blok, uygulama başlarken `prepare_artifacts.py` tarafından oluşturulan ve kaydedilen tüm gerekli dosyaları yüklemeye çalışır.
*   **`try...except` bloğu**: Dosya yükleme işlemleri sırasında bir hata oluşursa (örneğin, dosya bulunamazsa), programın çökmesini engellemek ve hatayı yakalayıp bir mesaj yazdırmak için kullanılır.
*   **`model = joblib.load(MODEL_PATH)`**: Kaydedilmiş XGBoost modelini yükler.
*   **`scaler = joblib.load(SCALER_PATH)`**: Kaydedilmiş StandardScaler nesnesini yükler.
*   **`model_columns = joblib.load(MODEL_COLUMNS_PATH)`**: Modelin eğitildiği sütunların listesini ve sırasını yükler.
*   **`total_charges_mean_data = joblib.load(TOTAL_CHARGES_MEAN_PATH)`** ve **`total_charges_mean = total_charges_mean_data["total_charges_mean"]`**: `TotalCharges` için kaydedilmiş ortalama değeri yükler. `.json` dosyası bir sözlük olarak yüklendiği için `["total_charges_mean"]` ile değere erişilir.
*   **`numerical_cols_to_scale = joblib.load(NUMERICAL_COLS_PATH)`**: Ölçeklenecek sayısal sütunların listesini yükler.
*   **`original_df = pd.read_csv(ORIGINAL_DATA_PATH)`**: Orijinal `WA_Fn-UseC_-Telco-Customer-Churn.csv` dosyasını bir pandas DataFrame olarak yükler. Bu, kullanıcı bir müşteri ID'si girdiğinde o müşterinin ham verilerini bulmak için kullanılacaktır.
*   Eğer `try` bloğunda herhangi bir hata olursa, `except` bloğu çalışır, bir hata mesajı yazdırılır ve `model` değişkeni `None` olarak ayarlanır. Bu, modelin yüklenemediğini ve uygulamanın tahmin yapamayacağını gösterir.

```python
def preprocess_single_customer(customer_raw_data_series, original_df_columns_for_get_dummies):
    if model is None:
        raise Exception("Model artifacts not loaded. Cannot preprocess.")

    customer_df = customer_raw_data_series.to_frame().T
    customer_df["TotalCharges"] = pd.to_numeric(customer_df["TotalCharges"], errors="coerce")
    customer_df["TotalCharges"].fillna(total_charges_mean, inplace=True)

    original_categorical_cols = []
    for col in original_df.columns:
        if original_df[col].dtype == 'object' and col not in ["customerID", "Churn"]:
            original_categorical_cols.append(col)
    
    processed_df = pd.get_dummies(customer_df, columns=original_categorical_cols, drop_first=False)

    for col in model_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0 
            
    processed_df = processed_df.reindex(columns=model_columns, fill_value=0)

    cols_to_scale_for_this_customer = [col for col in numerical_cols_to_scale if col in processed_df.columns]
    if cols_to_scale_for_this_customer:
        processed_df[cols_to_scale_for_this_customer] = scaler.transform(processed_df[cols_to_scale_for_this_customer])

    return processed_df
```

*   Bu fonksiyon (`preprocess_single_customer`), kullanıcı tarafından girilen tek bir müşterinin ham verilerini alır ve eğitilmiş modelin anlayabileceği formata dönüştürür. Bu, `prepare_artifacts.py` içinde yapılan ön işleme adımlarının aynısının tek bir veri örneği için uygulanmasıdır.
*   **`if model is None: raise Exception(...)`**: Eğer model yüklenmemişse, işlem yapmanın anlamı olmadığı için bir hata fırlatır.
*   **`customer_df = customer_raw_data_series.to_frame().T`**: Fonksiyona gelen `customer_raw_data_series` (bu, orijinal DataFrame'den çekilmiş tek bir müşteri satırıdır, bir pandas Series nesnesidir) önce `.to_frame()` ile tek sütunlu bir DataFrame'e, sonra `.T` (transpose - devrik) ile tek satırlı bir DataFrame'e dönüştürülür. Bu, `pandas` fonksiyonlarının (özellikle `get_dummies`) DataFrame üzerinde daha kolay çalışmasını sağlar.
*   **`customer_df["TotalCharges"] = pd.to_numeric(...)`** ve **`customer_df["TotalCharges"].fillna(...)`**: `prepare_artifacts.py`'deki gibi, bu müşterinin `TotalCharges` değeri sayısal yapılır ve eğer eksikse (veya sayıya çevrilememişse) daha önce yüklenen `total_charges_mean` ile doldurulur.
*   **`original_categorical_cols = [] ... original_categorical_cols.append(col)`**: Bu kısım, orijinal veri setindeki (`original_df`) kategorik sütunların bir listesini (tıpkı `prepare_artifacts.py`'de olduğu gibi) yeniden oluşturur. Bu, `get_dummies` işleminin doğru sütunlara uygulanmasını sağlar.
*   **`processed_df = pd.get_dummies(customer_df, columns=original_categorical_cols, drop_first=False)`**: Tek müşterilik `customer_df` DataFrame'indeki kategorik sütunlara one-hot encoding uygular. `drop_first=False` kullanılmış. `prepare_artifacts.py`'de `drop_first=True` kullanılmıştı. Bu bir tutarsızlık olabilir ve modelin beklentileriyle eşleşmesi için `prepare_artifacts.py`'dekiyle aynı (`True`) olması daha doğru olurdu. Ancak, sonraki `reindex` adımı bu durumu büyük ölçüde düzeltecektir.
*   **`for col in model_columns: if col not in processed_df.columns: processed_df[col] = 0`**: Bu döngü, modelin eğitildiği sırada var olan (`model_columns` listesinden gelen) tüm sütunların, bu tek müşterilik `processed_df`'te de var olmasını sağlar. Eğer one-hot encoding sonucu bazı sütunlar oluşmamışsa (örneğin, müşteri belirli bir hizmet kategorisine sahip değilse), o sütunlar 0 değeriyle eklenir.
*   **`processed_df = processed_df.reindex(columns=model_columns, fill_value=0)`**: Bu çok kritik bir adımdır. `processed_df`'i, modelin eğitildiği `model_columns` listesindeki sütun sırasına ve adına göre yeniden düzenler. Eğer `processed_df`'te eksik bir sütun varsa `fill_value=0` ile o sütun 0 olarak eklenir. Eğer fazladan bir sütun varsa atılır. Bu, modele verilecek verinin tam olarak beklenen formatta olmasını garantiler.
*   **`cols_to_scale_for_this_customer = [...]`**: Bu müşterinin verisinde bulunan ve ölçeklenmesi gereken sayısal sütunları belirler (daha önce yüklenen `numerical_cols_to_scale` listesini kullanarak).
*   **`if cols_to_scale_for_this_customer: processed_df[cols_to_scale_for_this_customer] = scaler.transform(...)`**: Eğer ölçeklenecek sütun varsa, daha önce yüklenmiş olan `scaler` nesnesinin `transform` metodu kullanılarak bu sütunlar ölçeklenir. Burada `fit_transform` değil, sadece `transform` kullanılır, çünkü scaler zaten `prepare_artifacts.py`'de eğitilmiştir.
*   **`return processed_df`**: Ön işlemesi tamamlanmış, modelin tahmin yapmaya hazır olduğu tek müşterilik DataFrame'i döndürür.

```python
@app.route("/", methods=["GET"])
def home():
    return flask.render_template("index.html")
```

*   Bu, Flask'ta bir "route" (rota) tanımlar. Bir kullanıcı web tarayıcısında uygulamanın ana adresine (örneğin, `http://localhost:5000/`) gittiğinde bu fonksiyon çalışır.
*   **`@app.route("/", methods=["GET"])`**: `/` URL'sine (kök adres) gelen HTTP GET isteklerini bu `home()` fonksiyonuna yönlendirir.
*   **`def home(): ...`**: Bu fonksiyon, `flask.render_template("index.html")` çağrısı yapar. Bu, `templates` klasöründeki `index.html` dosyasını bulur, içeriğini işler (eğer içinde dinamik kısımlar varsa) ve sonucu kullanıcının tarayıcısına HTML sayfası olarak gönderir. Yani, ana sayfayı gösterir.

```python
@app.route("/predict", methods=["POST"])
def predict():
    prediction_text = None
    error_text = None
    customer_display_data = None

    if model is None:
        error_text = "Model veya gerekli dosyalar yüklenemedi. Lütfen sunucu kayıtlarını kontrol edin."
        return flask.render_template("index.html", error_text=error_text)

    try:
        customer_id_from_form = flask.request.form["customerID"]
        
        try:
            customer_id_lookup = int(customer_id_from_form)
        except ValueError:
            error_text = f"Geçersiz Müşteri ID formatı: '{customer_id_from_form}'. Lütfen sayısal bir ID girin."
            return flask.render_template("index.html", error_text=error_text)

        customer_raw_data = original_df[original_df["customerID"] == customer_id_lookup]

        if customer_raw_data.empty:
            error_text = f"'{customer_id_from_form}' ID'li müşteri bulunamadı."
        else:
            customer_raw_data_series = customer_raw_data.iloc[0].copy()
            customer_display_data = customer_raw_data_series.to_dict()
            customer_display_data["customerID_display"] = customer_id_from_form 

            preprocessed_customer_data = preprocess_single_customer(customer_raw_data_series, [])
            
            prediction = model.predict(preprocessed_customer_data)
            prediction_proba = model.predict_proba(preprocessed_customer_data)

            if prediction[0] == 1:
                prediction_text = f"'{customer_id_from_form}' ID'li müşteri %{prediction_proba[0][1]*100:.2f} olasılıkla hizmeti BIRAKACAK."
            else:
                prediction_text = f"'{customer_id_from_form}' ID'li müşteri %{prediction_proba[0][0]*100:.2f} olasılıkla hizmete DEVAM EDECEK."

    except Exception as e:
        error_text = f"Bir hata oluştu: {str(e)}"
        import traceback
        print(traceback.format_exc())

    return flask.render_template("index.html", 
                                 prediction_text=prediction_text, 
                                 error_text=error_text, 
                                 customer_data=customer_display_data)
```

*   Bu, uygulamanın tahmin yapma mantığını içeren en önemli rotadır.
*   **`@app.route("/predict", methods=["POST"])`**: `/predict` URL'sine gelen HTTP POST isteklerini bu `predict()` fonksiyonuna yönlendirir. `index.html`'deki form, "Tahmin Et" butonuna basıldığında bu adrese POST isteği gönderir.
*   **`prediction_text = None ...`**: Sonuçları `index.html`'e göndermek için kullanılacak değişkenler başlangıçta `None` olarak ayarlanır.
*   **`if model is None: ...`**: Eğer model yüklenmemişse, bir hata mesajı ayarlanır ve `index.html` bu hata mesajıyla birlikte tekrar kullanıcıya gösterilir.
*   **`try...except Exception as e:`**: Tahmin süreci boyunca oluşabilecek genel hataları yakalamak için bir `try` bloğu kullanılır.
*   **`customer_id_from_form = flask.request.form["customerID"]`**: Formdan gönderilen `customerID` değerini alır. `flask.request.form` bir sözlüktür ve formdaki alan adlarını anahtar olarak kullanır.
*   **`try: customer_id_lookup = int(customer_id_from_form) ... except ValueError:`**: Gelen müşteri ID'sinin sayısal bir değere dönüştürülmeye çalışıldığı yer. Eğer `ValueError` (Değer Hatası) oluşursa, ID'nin geçersiz olduğu anlaşılır ve kullanıcıya hata mesajı gösterilir.
*   **`customer_raw_data = original_df[original_df["customerID"] == customer_id_lookup]`**: Yüklenmiş olan `original_df` (orijinal veri seti) içinde, girilen `customer_id_lookup`'a eşit olan satırı (müşteriyi) bulur.
*   **`if customer_raw_data.empty: ...`**: Eğer `customer_raw_data` boşsa (yani o ID'ye sahip müşteri bulunamamışsa), bir hata mesajı ayarlanır.
*   **`else: ...`**: Eğer müşteri bulunmuşsa:
    *   **`customer_raw_data_series = customer_raw_data.iloc[0].copy()`**: Bulunan müşterinin verilerini (DataFrame'in ilk satırını) bir pandas Series olarak alır. `.copy()` ile orijinal verinin bir kopyası üzerinde çalışılır.
    *   **`customer_display_data = customer_raw_data_series.to_dict()`**: Müşterinin ham verilerini (ID hariç) `index.html`'de göstermek için bir sözlüğe dönüştürür.
    *   **`customer_display_data["customerID_display"] = customer_id_from_form`**: Kullanıcının girdiği orijinal ID'yi (string formatında olabilir) göstermek için saklar.
    *   **`preprocessed_customer_data = preprocess_single_customer(customer_raw_data_series, [])`**: Müşterinin ham verilerini, daha önce tanımlanan `preprocess_single_customer` fonksiyonuna göndererek modelin anlayacağı formata dönüştürür. İkinci argüman olan `[]` (boş liste) `original_df_columns_for_get_dummies` parametresine karşılık gelir; fonksiyon içinde bu sütunlar zaten `original_df`'ten yeniden türetildiği için burada boş gönderilmesi sorun teşkil etmez.
    *   **`prediction = model.predict(preprocessed_customer_data)`**: Ön işlemesi yapılmış müşteri verisini, yüklenmiş olan XGBoost modelinin `predict` metoduna verir. Bu, kesin bir tahmin (0 veya 1) döndürür.
    *   **`prediction_proba = model.predict_proba(preprocessed_customer_data)`**: Aynı veriyi modelin `predict_proba` metoduna verir. Bu, her sınıf için olasılıkları döndürür (örneğin, `[[0.25, 0.75]]` gibi, ilk eleman sınıf 0'a ait olma olasılığı, ikinci eleman sınıf 1'e ait olma olasılığı).
    *   **`if prediction[0] == 1: ... else: ...`**: Modelin tahminine (`prediction[0]`) göre kullanıcıya gösterilecek metni oluşturur. Eğer tahmin 1 ise (churn edecek), churn olasılığını (`prediction_proba[0][1]`) kullanarak bir mesaj oluşturur. Eğer tahmin 0 ise (churn etmeyecek), kalma olasılığını (`prediction_proba[0][0]`) kullanarak bir mesaj oluşturur. `%.2f` ile olasılık değeri yüzde olarak ve virgülden sonra iki basamakla formatlanır.
*   **`except Exception as e: ...`**: `try` bloğunda herhangi bir beklenmedik hata oluşursa, genel bir hata mesajı ayarlanır ve hatanın detayları sunucu konsoluna yazdırılır (`traceback.format_exc()`).
*   **`return flask.render_template("index.html", prediction_text=..., error_text=..., customer_data=...)`**: Son olarak, `index.html` şablonu tekrar render edilir, ancak bu sefer `prediction_text`, `error_text` ve `customer_data` değişkenleri de şablona gönderilir. `index.html` bu değişkenleri kullanarak tahmin sonucunu, hatayı veya müşteri bilgilerini sayfada uygun yerlerde gösterir.

```python
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
```

*   Bu standart bir Python bloğudur.
*   **`if __name__ == "__main__":`**: Bu kodun sadece `python app.py` komutuyla doğrudan çalıştırıldığında icra edilmesini sağlar. Eğer bu dosya başka bir Python dosyası tarafından `import` edilirse, bu blok çalışmaz.
*   **`app.run(host="0.0.0.0", port=5000, debug=True)`**: Flask geliştirme sunucusunu başlatır.
    *   **`host="0.0.0.0"`**: Sunucunun makinenin tüm ağ arayüzlerinde (sadece localhost değil) dinlemesini sağlar. Bu, uygulamanın aynı ağdaki diğer cihazlardan erişilebilir olmasını sağlar (güvenlik duvarı ayarları izin veriyorsa).
    *   **`port=5000`**: Sunucunun 5000 numaralı portu dinleyeceğini belirtir. Tarayıcıda `http://<makine_ipsi>:5000` adresine gidilerek uygulamaya erişilir.
    *   **`debug=True`**: Hata ayıklama modunu aktif hale getirir. Bu modda, kodda bir değişiklik yapıldığında sunucu genellikle otomatik olarak yeniden başlar ve tarayıcıda daha detaylı hata mesajları gösterilir. Geliştirme aşamasında çok kullanışlıdır ancak canlı (production) ortamlarda `False` olarak ayarlanmalıdır.

Bu detaylı açıklama ile `app.py` dosyasının her bir parçasının ne işe yaradığını ve projenin web arayüzünün nasıl çalıştığını anlamış olmalısınız.



## Telco Müşteri Kaybı (Churn) Tahmin Projesi: Genel Bakış ve İşleyiş

Bu bölümde, projenin genel amacını, kullanılan veri setini, modelin nasıl hazırlandığını ve web uygulamasının nasıl çalıştığını bütünsel bir bakış açısıyla özetleyeceğiz. Bu özet, kodların detaylı açıklamasını tamamlayıcı nitelikte olup, projenin sunumunda genel çerçeveyi çizmenize yardımcı olacaktır.

### Projenin Amacı ve Önemi

Bu proje, bir telekomünikasyon şirketinin (Telco) müşterilerinin şirketten ayrılma (churn etme) olasılığını tahmin etmeyi amaçlayan bir makine öğrenimi uygulamasıdır. Müşteri kaybı, abonelik tabanlı tüm işletmeler için kritik bir sorundur çünkü:

1. Mevcut bir müşteriyi elde tutmak, yeni bir müşteri kazanmaktan genellikle 5-25 kat daha az maliyetlidir.
2. Müşteri kaybı doğrudan gelir kaybına neden olur.
3. Yüksek müşteri kaybı oranları, şirketin hizmet kalitesi veya fiyatlandırma stratejisi gibi alanlarda sorunlar olabileceğini gösterir.

Bu proje, hangi müşterilerin ayrılma riski taşıdığını önceden tahmin ederek, şirketin bu müşterileri elde tutmak için proaktif adımlar atmasına olanak tanır. Örneğin, ayrılma riski yüksek müşterilere özel indirimler sunulabilir, onlarla iletişime geçilerek sorunları çözülebilir veya hizmet kalitesi iyileştirilebilir.

### Kullanılan Veri Seti

Proje, `WA_Fn-UseC_-Telco-Customer-Churn.csv` adlı bir veri dosyası üzerine kurulmuştur. Bu veri seti, bir telekom şirketinin müşterilerine ait çeşitli bilgileri içerir:

1. **Müşteri Demografik Bilgileri:** Cinsiyet, yaş grubu (SeniorCitizen), medeni durum, bakmakla yükümlü olduğu kişiler olup olmadığı.
2. **Abonelik Bilgileri:** Müşterinin ne kadar süredir şirketin abonesi olduğu (tenure), sözleşme türü (aylık, yıllık, iki yıllık).
3. **Hizmet Bilgileri:** Telefon hizmeti, internet hizmeti türü (DSL, Fiber optik), çoklu hat, online güvenlik, online yedekleme, cihaz koruma, teknik destek, TV ve film yayını abonelikleri.
4. **Hesap ve Ödeme Bilgileri:** Ödeme yöntemi, kağıtsız fatura kullanımı, aylık ödeme tutarı (MonthlyCharges), toplam ödeme tutarı (TotalCharges).
5. **Hedef Değişken:** Müşterinin şirketten ayrılıp ayrılmadığı bilgisi (Churn sütunu).

### Projenin Genel İşleyişi

Proje iki ana bileşenden oluşur:

1. **Model Hazırlama ve Eğitme (`prepare_artifacts.py`):** Bu betik, veri setini işler, makine öğrenimi modelini eğitir ve gerekli tüm dosyaları kaydeder.
2. **Web Uygulaması (`app.py`):** Bu betik, kullanıcıların eğitilmiş modeli kullanarak tahminler yapabilmesini sağlayan bir web arayüzü sunar.

#### Model Hazırlama Süreci

1. **Veri Yükleme:** İlk adım, CSV dosyasından veri setinin yüklenmesidir.
2. **Veri Ön İşleme:**
   - **Eksik Değerlerin Yönetimi:** `TotalCharges` sütunundaki eksik veya sayısal olmayan değerler, ortalama değerle doldurulur.
   - **Kategorik Verilerin Dönüştürülmesi:** Metin tabanlı kategorik veriler (örneğin, "Evet"/"Hayır" veya "DSL"/"Fiber optik") one-hot encoding yöntemiyle sayısal formata dönüştürülür.
   - **Hedef Değişkenin Hazırlanması:** `Churn` sütunu, modelin anlayacağı şekilde 0 ve 1 değerlerine dönüştürülür.
3. **Veri Setinin Bölünmesi:** Veri seti, eğitim (%80) ve test (%20) olmak üzere iki parçaya ayrılır.
4. **Sayısal Özelliklerin Ölçeklenmesi:** Sayısal özellikler (`tenure`, `MonthlyCharges`, `TotalCharges`), `StandardScaler` kullanılarak standartlaştırılır.
5. **Model Eğitimi:** XGBoost sınıflandırma algoritması, eğitim veri seti üzerinde eğitilir.
6. **Model ve Yardımcı Dosyaların Kaydedilmesi:** Eğitilmiş model, scaler, sütun listesi ve diğer gerekli bilgiler `model_artifacts` klasörüne kaydedilir.

#### Web Uygulamasının Çalışma Mantığı

1. **Başlangıç:** Uygulama başlatıldığında, kaydedilmiş model ve diğer yardımcı dosyalar yüklenir.
2. **Kullanıcı Etkileşimi:** Kullanıcı, web arayüzünden bir müşteri ID'si girer ve "Tahmin Et" butonuna tıklar.
3. **Müşteri Verisinin Bulunması:** Girilen ID'ye sahip müşteri, orijinal veri setinden bulunur.
4. **Veri Ön İşleme:** Bulunan müşterinin ham verileri, modelin anlayacağı formata dönüştürülür. Bu, model eğitilirken uygulanan aynı ön işleme adımlarını içerir.
5. **Tahmin Yapma:** Ön işlemesi tamamlanmış müşteri verisi, eğitilmiş modele verilir ve model bir tahmin üretir.
6. **Sonucun Gösterilmesi:** Modelin tahmini (müşterinin ayrılıp ayrılmayacağı) ve bu tahminin olasılığı, kullanıcıya anlaşılır bir şekilde gösterilir.

### Projenin Teknik Özellikleri

1. **Kullanılan Programlama Dili ve Kütüphaneler:**
   - **Python:** Tüm proje Python dilinde yazılmıştır.
   - **pandas:** Veri manipülasyonu ve analizi için kullanılır.
   - **scikit-learn:** Veri ön işleme (StandardScaler) ve model değerlendirme için kullanılır.
   - **XGBoost:** Sınıflandırma modeli için kullanılır.
   - **Flask:** Web uygulaması geliştirmek için kullanılır.
   - **joblib:** Model ve diğer Python nesnelerini kaydetmek ve yüklemek için kullanılır.

2. **Makine Öğrenimi Modeli:**
   - **XGBoost Classifier:** Yüksek performanslı bir gradient boosting algoritmasıdır. Karar ağaçlarını ardışık olarak birleştirerek, her yeni ağacın önceki ağaçların hatalarını düzeltmeye çalıştığı bir topluluk (ensemble) yöntemidir.
   - **Avantajları:** Genellikle yüksek tahmin doğruluğu, aşırı öğrenmeye (overfitting) karşı dayanıklılık ve hızlı eğitim süresi sunar.

3. **Web Uygulaması:**
   - **Flask Framework:** Hafif ve esnek bir Python web çatısıdır.
   - **HTML/CSS:** Kullanıcı arayüzü için kullanılır.
   - **Jinja2 Şablonları:** Flask'ın HTML şablonlarında dinamik içerik oluşturmak için kullandığı bir şablonlama motorudur.

### Projenin Kullanım Senaryoları

1. **Müşteri Hizmetleri:** Müşteri temsilcileri, riskli müşterileri belirleyip onlara özel ilgi gösterebilir.
2. **Pazarlama Kampanyaları:** Ayrılma riski yüksek müşterilere özel indirimler veya teklifler sunulabilir.
3. **Ürün Geliştirme:** Hangi hizmet özelliklerinin müşteri kaybıyla ilişkili olduğu analiz edilerek, ürün iyileştirmeleri yapılabilir.
4. **İş Stratejisi:** Genel müşteri kaybı eğilimleri izlenerek, şirketin uzun vadeli stratejileri şekillendirilebilir.

### Sunum İpuçları

Projeyi hocanıza sunarken şu noktalara dikkat etmeniz faydalı olacaktır:

1. **Problemi Net Tanımlayın:** Müşteri kaybının neden önemli olduğunu ve bu projenin nasıl çözüm sunduğunu açıklayın.
2. **Veri Setini Kısaca Tanıtın:** Hangi tür müşteri bilgilerinin kullanıldığını ve özellikle `Churn` etiketinin ne anlama geldiğini belirtin.
3. **Teknik Detayları Basitleştirin:** Veri ön işleme, model eğitimi ve tahmin süreçlerini teknik olmayan bir dille açıklayın.
4. **Kodları Adım Adım Açıklayın:** Bu dokümanda sunulan kod açıklamalarını kullanarak, `prepare_artifacts.py` ve `app.py` dosyalarının nasıl çalıştığını anlatın.
5. **Canlı Demo Yapın (Mümkünse):** Uygulamayı çalıştırıp birkaç farklı müşteri ID'si girerek nasıl tahmin yaptığını gösterin.
6. **Projenin Değerini Vurgulayın:** Bu tür bir tahmin modelinin bir telekom şirketine nasıl değer katabileceğini açıklayın.

Bu genel bakış, projenin teknik detaylarını ve pratik önemini bir araya getirerek, hocanıza yapacağınız sunumda projeyi bütünsel bir şekilde anlatmanıza yardımcı olacaktır.
