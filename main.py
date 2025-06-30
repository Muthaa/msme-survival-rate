from src.data_loader import load_data
from src.preprocessing import preprocess
from src.train import train_model
from src.evaluate import evaluate_model
import joblib

def main():
    df = load_data("data/msme_raw.csv")
    X_train, X_test, y_train, y_test = preprocess(df)
    train_model(X_train, y_train)
    model = joblib.load("models/rf_model.pkl")
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
