#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

     
df = pd.read_csv(r"C:\Users\gamze\OneDrive\Masaüstü\Datascienceproject\tweetsset.csv", encoding="utf-8")
#df.duplicated() # tekrarlayan veriyi gösterir
#df.drop_duplicates() #Tekrar eden satırları siler
# Veri setinde kayıp verilerin olup olmadığına bakıyorum ve düzeltilemeyecek kadar olan feature'leri siliyorum.Kayıp verimin olmadığını gördüm.
#print("Kayıp Veriler :{}".format(df.isnull().sum()))
# --- EKSIK VERI DOLDURMA (DATA IMPUTATION) ---
# Toplam kaç hücrede eksik değer (NaN ya da None) var?
#print("Total Null:",df.isnull().sum().sum()) #sonuç : 0
#df.shape #Satır ve sütun (öznitelik) sayısını görüntüleme
#df.info()
#df.isnull().sum()
#print("Grouped Null---\n",df.isnull().sum())
import string
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem.porter import PorterStemmer
import re
from nltk.stem import WordNetLemmatizer
from snowballstemmer import TurkishStemmer
import nltk
# Storing the sets of punctuation in variable punctation
punctation = string.punctuation
#punctuation ='''!()-[]{};':'"\,<>./?@#$%^&*_~'''
#Özel karakterleri temizleme
def ozelkarakter_temizleme (metin):
    return metin.translate(str.maketrans("","",punctation))

#Stopword temizleme
stopword = set(stopwords.words("turkish"))
def stopwords_temizleme (metin):
    return " ".join([kelime for kelime in str(metin).split() if kelime not in stopword])


#sık kullanılan kelimeleri temizleme
count = Counter()
for metin in df["Tweet"].values:
    for kelime in metin.split():
        count[kelime] += 1
count.most_common(10) # en sık tekrar eden 10 kelimeyi gösterir
frekans = set([i for (i,j) in count.most_common(15)])
nadir = 15
nadir_kelime = set([i for (i,j) in count.most_common()[:-nadir-1:-1]])
def frekans_sil(metin):
    return " ".join([kelime for kelime in str(metin).split() if kelime not in frekans])
#----------------Kelime Kökünü Alma
#lemma = WordNetLemmatizer("turkish")
#Lemmatizer

#def kelime_kök_alma (metin):
 #   return " ".join([lemma.lemmatize(kelime) for kelime in metin.split()])
 
    
snowBallStememr = TurkishStemmer()
def kelime_kök_alma(metin):
    wordlist = nltk.word_tokenize(metin)
    stemWords = [snowBallStememr.stemWord(kelime) for kelime in wordlist]
    return " ".join(stemWords)  
#---------
#Emojileri Silme

def emoji_silme (metin):
    emoji = re.compile("["
                               u"\U0001F600-\U0001F64F"  
                               u"\U0001F300-\U0001F5FF"                                 
                               u"\U0001F680-\U0001F6FF"  
                               u"\U0001F1E0-\U0001F1FF"  
                               u"\U00002500-\U00002BEF"                                 
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji.sub(r"",metin)


df["Tweet"] = df["Tweet"].str.lower()
df["ozel_karaktersiz"] = df["Tweet"].apply(lambda metin : ozelkarakter_temizleme(metin))
df["stop_word"] = df["ozel_karaktersiz"].apply(lambda metin : stopwords_temizleme(metin) )
df["sık_kullanılan"] = df["stop_word"].apply(lambda metin : frekans_sil(metin) )
df["kelime_kok"] = df["sık_kullanılan"].apply(lambda kelime : kelime_kök_alma(kelime))
df["emojisiz"] = df["kelime_kok"].apply(lambda metin : emoji_silme(metin))

pd.concat([pd.concat([df["User"], df["emojisiz"]], axis=1)]).to_csv('tweetsset_after_preprocessing.csv')
#df.shape
#print(df)