import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime
import json

class AnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.is_fitted = False
        self.training_data = []
    
    def extract_features(self, device_data):
        features = []
        current_time = datetime.utcnow()
        
        if device_data.get('last_location'):
            lat = device_data['last_location'].get('latitude', 0)
            lon = device_data['last_location'].get('longitude', 0)
            features.extend([lat, lon])
        
        if device_data.get('last_updated'):
            time_diff = (current_time - device_data['last_updated']).total_seconds() / 3600
            features.append(time_diff)
        
        hour = current_time.hour
        features.append(hour)
        
        while len(features) < 10:
            features.append(0)
        
        return features[:10]
    
    def add_training_data(self, device_data):
        features = self.extract_features(device_data)
        self.training_data.append(features)
        
        if len(self.training_data) >= 50:
            self.train_model()
    
    def train_model(self):
        if len(self.training_data) < 10:
            return
        
        X = np.array(self.training_data)
        self.model.fit(X)
        self.is_fitted = True
    
    def detect_anomaly(self, device_data):
        if not self.is_fitted or len(self.training_data) < 10:
            return False
        
        features = self.extract_features(device_data)
        prediction = self.model.predict([features])
        return prediction[0] == -1