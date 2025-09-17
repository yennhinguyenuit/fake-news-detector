import re, os, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def clean_text(t):
    if not isinstance(t, str): return ""
    t = t.lower()
    t = re.sub(r'http\S+',' ', t)
    t = re.sub(r'[^a-zA-ZÀ-ỹ0-9\s]',' ', t)  # giữ Unicode tiếng Việt
    t = re.sub(r'\s+',' ', t).strip()
    return t

# đọc dữ liệu
df = pd.read_csv("data/data.csv", encoding="utf-8")
df = df[['text','label']].dropna().reset_index(drop=True)
df['text'] = df['text'].map(clean_text)
df['y'] = df['label'].map(lambda x: 1 if str(x).lower().strip()=='fake' else 0)

X = df['text'].values
y = df['y'].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# vector hóa & huấn luyện
vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
Xtrv = vec.fit_transform(Xtr)
Xtev = vec.transform(Xte)

clf = LogisticRegression(max_iter=300, class_weight='balanced')
clf.fit(Xtrv, ytr)

# đánh giá
ypred = clf.predict(Xtev)
print("Classification report:\n", classification_report(yte, ypred))
print("Confusion matrix:\n", confusion_matrix(yte, ypred))

# lưu mô hình
os.makedirs("models", exist_ok=True)
joblib.dump(vec, "models/tfidf.joblib")
joblib.dump(clf, "models/lr_fake.joblib")
print("✅ Saved models to models/")
