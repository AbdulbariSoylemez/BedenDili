import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Veriyi yükle
data_dict = pickle.load(open("veri.pickle", "rb"))
data = np.asarray(data_dict["data"])
labels = np.asarray(data_dict["labels"])

# Veriyi eğitim ve test setlerine ayır
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, shuffle=True, stratify=labels)

# Random Forest modelini oluştur ve eğit
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Modelin doğruluğunu hesapla
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print("Model {}% oranında doğru sonuç üretiyor.".format((score * 100)))

# Eğitilmiş modeli kaydet
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
