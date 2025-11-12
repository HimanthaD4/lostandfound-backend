from datetime import datetime
import json

class UserModel:
    def __init__(self, db):
        self.db = db
        self.users = db.users
    
    def create_user(self, email, password):
        user = {
            'email': email,
            'password': password,
            'created_at': datetime.utcnow(),
            'devices': []
        }
        return self.users.insert_one(user)
    
    def find_user(self, email):
        return self.users.find_one({'email': email})
    
    def add_device(self, email, device_data):
        # Set initial timestamps
        device_data['created_at'] = datetime.utcnow()
        device_data['last_updated'] = datetime.utcnow()
        
        return self.users.update_one(
            {'email': email},
            {'$push': {'devices': device_data}}
        )
    
    def update_device_location(self, email, device_id, location_data):
        return self.users.update_one(
            {'email': email, 'devices.device_id': device_id},
            {'$set': {
                'devices.$.last_location': location_data,
                'devices.$.last_updated': datetime.utcnow()
            }}
        )
    
    def get_user_devices(self, email):
        user = self.users.find_one({'email': email})
        return user.get('devices', []) if user else []

class AlertModel:
    def __init__(self, db):
        self.db = db
        self.alerts = db.alerts
    
    def create_alert(self, alert_data):
        alert_data['created_at'] = datetime.utcnow()
        alert_data['resolved'] = False
        return self.alerts.insert_one(alert_data)
    
    def get_user_alerts(self, email):
        return list(self.alerts.find({'user_email': email}).sort('created_at', -1))