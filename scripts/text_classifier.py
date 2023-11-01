from sklearn.metrics import accuracy_score, classification_report
import inspect

class TextClassifier:

    def __init__(self, model, vectorizer=None):
        self.model = model
        self.vectorizer = vectorizer

    def train(self, X_train, y_train, sample_weight=None):
        # Si un vectoriseur est fourni, utilisez-le pour transformer les données
        if self.vectorizer:
            X_train = self.vectorizer.fit_transform(X_train)
        
        # Vérifiez si le modèle actuel accepte l'argument 'sample_weight'
        if "sample_weight" in inspect.signature(self.model.fit).parameters:
            self.model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        # Si un vectoriseur est fourni, utilisez-le pour transformer les données
        if self.vectorizer:
            X_test = self.vectorizer.transform(X_test)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

    def predict(self, X):
        # Si un vectoriseur est fourni, utilisez-le pour transformer les données
        if self.vectorizer:
            X = self.vectorizer.transform(X)
        return self.model.predict(X)

