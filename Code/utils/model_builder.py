import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from config import features

def get_scaler(X):
    # Normalize the data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return scaler, X_scaled
    
def split_data(X_scaled, y):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Train Random Forest model
def build_rf_model(X_train,y_train):    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model
# Predictions
def predict_model(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred