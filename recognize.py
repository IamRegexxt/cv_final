# train_classifier.py
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# Load embeddings
with open('embedd.pkl', 'rb') as f:
    data = pickle.load(f)

embeddings = data['embeddings']
labels = data['labels']

# Encode string labels
le = LabelEncoder()
labels_num = le.fit_transform(labels)

# Train SVM classifier
model = SVC(C=1.0, kernel='linear', probability=True)
model.fit(embeddings, labels_num)

# Save classifier and label encoder
with open('classifier.pkl', 'wb') as f:
    pickle.dump((model, le), f)

print("Classifier trained and saved as classifier.pkl")
