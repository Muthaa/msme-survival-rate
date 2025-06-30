from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(X_train, y_train, model_path: str = 'models/rf_model.pkl'):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")
