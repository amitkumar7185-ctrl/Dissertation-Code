from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
def prin_df_head(pivot_df):
    pivot_df.head()

def print_dataset_info(df):
    df.head()
    print(df.info())

def print_data_distribution(pivot_df):
    fault_counts = pivot_df['Fault'].value_counts()
    print(fault_counts)

    fault_percent = pivot_df['Fault'].value_counts(normalize=True) * 100
    print(fault_percent)

def print_confusion_matrix(y_test, y_pred):
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Fault', 'Fault'],
                yticklabels=['No Fault', 'Fault'],
                ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    return fig  


def print_classification(y_test, y_pred):
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))