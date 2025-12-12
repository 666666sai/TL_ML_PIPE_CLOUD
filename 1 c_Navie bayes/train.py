import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import (accuracy_score)     
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
base = Path(__file__).resolve().parent

filepath=base/"sms+spam+collection_dataset/SMSSpamCollection.xlsx"
data=pd.read_excel(filepath, names=['label', 'message'])

data.dropna(inplace=True)

def clean_text(text):
    # ensure we have a string (avoids AttributeError when non-strings appear)
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]  # remove stopwords
    return " ".join(tokens)

data['clean_message'] = data['message'].apply(clean_text)
le = LabelEncoder()
data['label_num'] = le.fit_transform(data['label'])  # 'ham'->0, 'spam'->1 typically


X_train, X_test, y_train, y_test = train_test_split(
    data['clean_message'], data['label_num'], test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))






# -----------------------------

#  Feature Extraction
#  GaussianNB
# -----------------------------



tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

x_train_dense=X_train_tfidf.toarray()
X_test_dense=X_test_tfidf.toarray()
gnb=GaussianNB()
gnb.fit(x_train_dense,y_train)
y_prd_gnd=gnb.predict(X_test_dense)

print("Accuracy:", accuracy_score(y_test, y_prd_gnd))


#------------------------------------

#  above gaussian naive bayes model not suitable for long text data
# so better to go Multinominal NB or Bernovilli NB

#-----------------------------------
mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_tfidf, y_train)
mnb_pred = mnb.predict(X_test_tfidf)
print("MultinomialNB Accuracy:", accuracy_score(y_test, mnb_pred))