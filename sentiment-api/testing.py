import joblib

# ─── Load model and label encoder ───
MODEL_DIR = "C:\\Users\\Deepanshu\\Downloads\\sentiment-api\\model"
pipe = joblib.load(f"{MODEL_DIR}\\model.joblib")
le = joblib.load(f"{MODEL_DIR}\\label_encoder.joblib")

# ─── Cleaning function (same as training) ───
import re, string
import nltk
nltk.download("stopwords")
STOPWORDS = set(nltk.corpus.stopwords.words("english"))

def clean(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|#\w+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(tokens)

# ─── Batch of sentences to test ───
sentences = [
    "I love this product! Totally amazing experience.",
    "This app keeps crashing, really annoying!",
    "Had a wonderful dinner with friends tonight.",
    "The food was terrible and cold.",
    "Just got a promotion at work! Super happy!"
]

# ─── Clean sentences ───
clean_sentences = [clean(s) for s in sentences]

# ─── Predict ───
predictions = pipe.predict(clean_sentences)
pred_labels = le.inverse_transform(predictions)

# ─── Print results ───
for s, p in zip(sentences, pred_labels):
    print(f"Sentence: {s}\nPredicted sentiment: {p}\n")


pipe = joblib.load("model/model.joblib")
le = joblib.load("model/label_encoder.joblib")

sentences = [
    "I love this product! Totally amazing experience.",
    "This app keeps crashing, really annoying!",
    "Had a wonderful dinner with friends tonight.",
    "The food was terrible and cold.",
    "Just got a promotion at work! Super happy!"
]

# clean + predict as before
