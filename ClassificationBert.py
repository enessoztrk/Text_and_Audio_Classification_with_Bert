#######################
# Text Classification #
#######################

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import accuracy_score
# !pip install simpletransformers

# Read Dataset
df = pd.read_excel("C:/Users/hp/PycharmProjects/Text and Audio Classification with Bert/data.xlsx")
"""""
    kategori	        metin
0	Bilim Ve Teknoloji	Fransa Ulusal Arkeolojik Araştırma Enstitüsü’n...
1	Bilim Ve Teknoloji	Baykar Teknoloji Lideri Selçuk Bayraktar, Kana...
2	Bilim Ve Teknoloji	Soyuz MS-18 uzay aracı, Kazakistan’ın Baykonur...
3	Bilim Ve Teknoloji	Siyasetten ekonomiye, spordan sağlığa, özel ha...
4	Bilim Ve Teknoloji	Öğrencilerin uzay teknolojileri ve roket bilim...
...	               ...	                                              ...
3056	          Spor	UEFA Avrupa Ligi'nde 2016-2017 sezonu grup kur...
3057	          Spor	Spor Toto Süper Lig'in 2. haftasında görev ala...
3058	          Spor	Geçen sezonun sona ermesinin ardından Trabzons...
3059	          Spor	TFF'den yapılan açıklamada, daha önce kurulda ...
3060	          Spor	2016 Rio Olimpiyat Oyunları'nda Türkiye'ye atl...
3061 rows × 2 columns
"""""

# Unique Categories
df["kategori"].unique()
# array(['Bilim Ve Teknoloji', 'Ekonomi', 'Sağlık', 'Siyaset', 'Spor'],dtype=object)

# Labeling
df['labels'] = pd.factorize(df.kategori)[0]
"""""
    kategori	        metin	                                            labels
0	Bilim Ve Teknoloji	Fransa Ulusal Arkeolojik Araştırma Enstitüsü’n...	0
1	Bilim Ve Teknoloji	Baykar Teknoloji Lideri Selçuk Bayraktar, Kana...	0
2	Bilim Ve Teknoloji	Soyuz MS-18 uzay aracı, Kazakistan’ın Baykonur...	0
3	Bilim Ve Teknoloji	Siyasetten ekonomiye, spordan sağlığa, özel ha...	0
4	Bilim Ve Teknoloji	Öğrencilerin uzay teknolojileri ve roket bilim...	0
...	               ...	                                              ...  ...
3056	          Spor  UEFA Avrupa Ligi'nde 2016-2017 sezonu grup kur...	4
3057	          Spor  Spor Toto Süper Lig'in 2. haftasında görev ala...	4
3058	          Spor  Geçen sezonun sona ermesinin ardından Trabzons...	4
3059	          Spor  TFF'den yapılan açıklamada, daha önce kurulda ...	4
3060	          Spor  2016 Rio Olimpiyat Oyunları'nda Türkiye'ye atl...	4
3061 rows × 3 columns
"""""

# test %20 - train %80
train, test = train_test_split(a, test_size=0.2, random_state=42)

# kategori dropped
train = train[["metin", "labels"]]
test = test[["metin", "labels"]]

# for bert text = string, label = int
train["metin"] = train["metin"].apply(lambda r: str(r))
train['labels'] = train['labels'].astype(int)

# Available models
model = ClassificationModel('bert', 'dbmdz/bert-base-turkish-uncased', num_labels=5, use_cuda=False,
                            args={'reprocess_input_data': True, 'overwrite_output_dir': True, 'num_train_epochs': 3,
                                  "train_batch_size": 64, "fp16": False, "output_dir": "bert_model"})

# Build models
model.train_model(train)

# Test data given to model data
result, model_outputs, wrong_predictions = model.eval_model(test)

# Predictions Vs Actuals
predictions = model_outputs.argmax(axis=1)
actuals = test.labels.values

predictions[:10]
# array([3, 0, 2, 2, 3, 0, 4, 4, 1, 2])

actuals[:10]
# array([3, 0, 2, 2, 3, 0, 4, 4, 1, 2])

# Predictor
# 97% success rate achieved
accuracy_score(actuals, predictions)

example = test.iloc[43]['metin']

print(example)
# İstanbul Teknik Üniversitesi (İTÜ) Nanoteknoloji Uygulama ve Araştırma Merkezi (İTÜnano)
# bünyesinde geliştirilen "NanoGeliştirilmiş Ölçeklenebilir Kuantum Güneş Pili Tasarımı Üretimi ve Karakterizasyonu"
# başlıklı projeyle nanoteknolojik malzemeden güneş pili üretilecek. Söz konusu eğitimin verildiği BTK Akademi,
# 2017'de Bilgi Teknolojileri ve İletişim Kurumu (BTK) İnsan Kaynakları ve Eğitim Dairesi Başkanlığı bünyesinde kuruldu.
# Cumhurbaşkanı Erdoğan'ın da katılımıyla 2020 yılı şubat ayında BTK ana yerleşkesinde halkın hizmetine açıldı.

tahmin = model.predict([tryy])

if tahmin[0] == 0:
    print("BİLİM VE TEKNOLOJİ")
elif tahmin[0] == 1:
    print("EKONOMİ")
elif tahmin[0] == 2:
    print("SAĞLIK")
elif tahmin[0] == 3:
    print("SİYASET")
else:
    print("SPOR")
# # BİLİM VE TEKNOLOJİ

#################################

# data2 import
dff = pd.read_excel("C:/Users/hp/PycharmProjects/Text and Audio Classification with Bert/data2.xlsx", names=["kategori", "metin"])

tryy = dff.iloc[3]["metin"]

example = dff.iloc[11]["metin"]

tahmin = model.predict([example])

if tahmin[0] == 0:
    print("BİLİM VE TEKNOLOJİ")
elif tahmin[0] == 1:
    print("EKONOMİ")
elif tahmin[0] == 2:
    print("SAĞLIK")
elif tahmin[0] == 3:
    print("SİYASET")
else:
    print("SPOR")


#######################################
# Classification from Audio Recording #
#######################################

import speech_recognition as sr

recognizer = sr.Recognizer()

''' recording the sound '''

with sr.AudioFile("C:/Users/hp/PycharmProjects/Text and Audio Classification with Bert/Sport.wav") as source:
    recorded_audio = recognizer.listen(source)
    print("Done recording")

''' Recorgnizing the Audio '''
try:
    print("Recognizing the text")
    text = recognizer.recognize_google(
        recorded_audio,
        language='tr-tr'
    )
    model.predict([tryy])

except Exception as ex:
    print(ex)
c = model.predict([text])

if c[0] == 0:
    print("BİLİM VE TEKNOLOJİ")
elif c[0] == 1:
    print("EKONOMİ")
elif c[0] == 2:
    print("SAĞLIK")
elif c[0] == 3:
    print("SİYASET")
else:
    print("SPOR")
# SPOR
