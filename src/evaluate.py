from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test, output_dir='outputs/'):
    y_pred = model.predict(X_test)
    
    report = classification_report(y_test, y_pred)
    with open(f"{output_dir}/classification_report.txt", "w") as f:
        f.write(report)
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()
    
    print("Evaluation complete. Results saved.")
