from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime, timezone, timedelta
import hashlib
import json
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import threading
import time
import logging
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.config["SECRET_KEY"] = "your-secret-key-here"

# Configure CORS properly for all origins during development
# CORS configuration for both development and production
cors_origins = [
    "http://localhost:3000", 
    "http://192.168.1.125:3000", 
    "http://127.0.0.1:3000", 
    "http://10.185.94.208:3000",
    "https://your-frontend-app.onrender.com"  # Add your future frontend URL
]

CORS(app, resources={
    r"/api/*": {
        "origins": cors_origins,
        "methods": ["GET", "POST", "PUT", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# MongoDB connection
mongodb_uri = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/')
client = MongoClient(mongodb_uri, connectTimeoutMS=30000, socketTimeoutMS=30000)
db = client['device_tracker']

# Create unique index for device_id across all users - MOVED AFTER db DEFINITION
try:
    db.users.create_index([("devices.device_id", 1)], unique=True, sparse=True)
    print("✅ Unique index created for device_id across all users")
except Exception as e:
    print(f"⚠️ Index may already exist: {e}")

# Custom JSON encoder to handle ObjectId and datetime
class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        return json.JSONEncoder.default(self, o)

app.json_encoder = JSONEncoder

class BehaviorAnalyzer:
    def __init__(self):
        self.user_profiles = {}
        self.learning_period = timedelta(minutes=7)  # 7 minutes = 7 days simulation
        self.learning_start_times = {}
    
    def initialize_user_learning(self, email):
        if email not in self.user_profiles:
            self.user_profiles[email] = {
                'learning_start': datetime.now(timezone.utc),
                'behavior_learned': False,
                'daily_patterns': {
                    'morning_locations': [],    # 6AM - 12PM
                    'afternoon_locations': [],  # 12PM - 6PM  
                    'evening_locations': [],    # 6PM - 12AM
                    'night_locations': []       # 12AM - 6AM
                },
                'device_movement_patterns': {},
                'typical_locations': {},
                'schedule_consistency': 0.0,
                'location_clusters': [],
                'time_based_rules': {},
                'learning_progress': 0
            }
            self.learning_start_times[email] = datetime.now(timezone.utc)
            print(f"🎯 Started behavior learning for {email}")
    
    def record_location_behavior(self, email, device_data, location_data):
        if email not in self.user_profiles:
            self.initialize_user_learning(email)
        
        current_time = datetime.now(timezone.utc)
        hour = current_time.hour
        device_id = device_data['device_id']
        device_type = device_data['device_type']
        
        # Record location based on time of day
        location_point = {
            'latitude': location_data['latitude'],
            'longitude': location_data['longitude'], 
            'timestamp': current_time,
            'device_type': device_type,
            'campus_section': self._get_campus_section(location_data),
            'hour': hour
        }
        
        # Categorize by time of day
        if 6 <= hour < 12:
            self.user_profiles[email]['daily_patterns']['morning_locations'].append(location_point)
        elif 12 <= hour < 18:
            self.user_profiles[email]['daily_patterns']['afternoon_locations'].append(location_point)
        elif 18 <= hour < 24:
            self.user_profiles[email]['daily_patterns']['evening_locations'].append(location_point)
        else:
            self.user_profiles[email]['daily_patterns']['night_locations'].append(location_point)
        
        # Update device movement patterns
        if device_id not in self.user_profiles[email]['device_movement_patterns']:
            self.user_profiles[email]['device_movement_patterns'][device_id] = {
                'device_type': device_type,
                'typical_locations': [],
                'movement_times': [],
                'stationary_periods': []
            }
        
        self.user_profiles[email]['device_movement_patterns'][device_id]['typical_locations'].append(location_point)
        self.user_profiles[email]['device_movement_patterns'][device_id]['movement_times'].append(current_time)
        
        # Update learning progress
        self._update_learning_progress(email)
        
        # Check if learning period is complete
        if self._is_learning_complete(email):
            self._finalize_behavior_learning(email)
    
    def _get_campus_section(self, location_data):
        lat = location_data['latitude']
        lng = location_data['longitude']
        
        # Simple campus section detection based on coordinates
        if 6.9265 <= lat <= 6.9275 and 79.8605 <= lng <= 79.8615:
            if lat > 6.9270:
                return 'library' if lng < 79.8610 else 'lab'
            else:
                return 'classroom' if lng < 79.8610 else 'admin'
        return 'outside_campus'
    
    def _update_learning_progress(self, email):
        if email not in self.learning_start_times:
            return
        
        elapsed = datetime.now(timezone.utc) - self.learning_start_times[email]
        total_minutes = self.learning_period.total_seconds() / 60
        progress_minutes = min(elapsed.total_seconds() / 60, total_minutes)
        
        progress_percent = (progress_minutes / total_minutes) * 100
        self.user_profiles[email]['learning_progress'] = progress_percent
        
        # Simulate day-by-day learning
        simulated_days = int(progress_minutes)  # 1 minute = 1 day
        print(f"📅 {email}: Simulated {simulated_days}/7 days learned - {progress_percent:.1f}% complete")
    
    def _is_learning_complete(self, email):
        return self.user_profiles[email]['learning_progress'] >= 100
    
    def _finalize_behavior_learning(self, email):
        profile = self.user_profiles[email]
        
        # Analyze patterns and create rules
        self._analyze_daily_patterns(email)
        self._create_time_based_rules(email)
        self._calculate_schedule_consistency(email)
        
        profile['behavior_learned'] = True
        print(f"✅ BEHAVIOR LEARNING COMPLETE for {email}")
        print(f"   - Daily patterns analyzed")
        print(f"   - Time-based rules created") 
        print(f"   - Schedule consistency: {profile['schedule_consistency']:.2f}")
    
    def _analyze_daily_patterns(self, email):
        profile = self.user_profiles[email]
        
        # Find most common locations for each time period
        for period, locations in profile['daily_patterns'].items():
            if locations:
                # Simple clustering - find centroid of locations
                lats = [loc['latitude'] for loc in locations]
                lngs = [loc['longitude'] for loc in locations]
                centroid_lat = sum(lats) / len(lats)
                centroid_lng = sum(lngs) / len(lngs)
                
                profile['typical_locations'][period] = {
                    'latitude': centroid_lat,
                    'longitude': centroid_lng,
                    'count': len(locations),
                    'campus_section': locations[0]['campus_section'] if locations else 'unknown'
                }
    
    def _create_time_based_rules(self, email):
        profile = self.user_profiles[email]
        rules = {}
        
        # Create rules based on device types and time patterns
        for device_id, device_pattern in profile['device_movement_patterns'].items():
            device_type = device_pattern['device_type']
            locations = device_pattern['typical_locations']
            
            if locations:
                # Find most common location for this device
                common_sections = {}
                for loc in locations:
                    section = loc.get('campus_section', 'unknown')
                    common_sections[section] = common_sections.get(section, 0) + 1
                
                if common_sections:
                    most_common_section = max(common_sections, key=common_sections.get)
                    
                    if device_type == 'laptop':
                        # Laptop should stay in library during day
                        rules[device_id] = {
                            'expected_locations': ['library', 'lab', 'classroom'],
                            'unexpected_times': {'night': 'Laptop should not move at night'},
                            'stationary_expected': True,
                            'typical_section': most_common_section
                        }
                    elif device_type == 'mobile':
                        # Mobile can move but should follow patterns
                        rules[device_id] = {
                            'expected_locations': ['library', 'lab', 'classroom', 'admin', 'outside_campus'],
                            'unexpected_times': {},
                            'stationary_expected': False,
                            'typical_section': most_common_section
                        }
        
        profile['time_based_rules'] = rules
    
    def _calculate_schedule_consistency(self, email):
        profile = self.user_profiles[email]
        
        # Calculate how consistent the user's schedule is
        total_locations = 0
        consistent_locations = 0
        
        for period, typical_loc in profile['typical_locations'].items():
            if typical_loc['count'] > 0:
                total_locations += 1
                # If user has significant locations in this period, consider it consistent
                if typical_loc['count'] >= 3:  # At least 3 data points
                    consistent_locations += 1
        
        profile['schedule_consistency'] = consistent_locations / total_locations if total_locations > 0 else 0
    
    def detect_behavior_anomaly(self, email, device_data, location_data):
        if email not in self.user_profiles or not self.user_profiles[email]['behavior_learned']:
            return None, "Behavior learning in progress"
        
        profile = self.user_profiles[email]
        device_id = device_data['device_id']
        device_type = device_data['device_type']
        current_time = datetime.now(timezone.utc)
        hour = current_time.hour
        current_section = self._get_campus_section(location_data)
        
        # Get rules for this device
        device_rules = profile['time_based_rules'].get(device_id, {})
        
        anomalies = []
        
        # Rule 1: Check if device is in unexpected location
        if device_rules and 'expected_locations' in device_rules:
            if current_section not in device_rules['expected_locations']:
                anomalies.append(f"Device in unexpected location: {current_section}")
        
        # Rule 2: Check for unusual movement times (especially for laptops)
        if device_type == 'laptop':
            if hour < 6 or hour > 22:  # Late night/early morning
                anomalies.append("Laptop moving during unusual hours")
        
        # Rule 3: Check if device separation is suspicious
        if self._suspicious_device_separation(email, device_data, location_data):
            anomalies.append("Suspicious device separation detected")
        
        # Rule 4: Check against learned daily patterns
        period_anomaly = self._check_daily_pattern_anomaly(email, device_data, location_data, current_time)
        if period_anomaly:
            anomalies.append(period_anomaly)
        
        return anomalies, "Behavior analysis complete"
    
    def _suspicious_device_separation(self, email, current_device_data, current_location):
        profile = self.user_profiles[email]
        current_device_type = current_device_data['device_type']
        
        # If this is a mobile device, check if laptop is too far away
        if current_device_type == 'mobile':
            laptop_locations = []
            mobile_locations = []
            
            # Get all device locations
            for device_id, device_pattern in profile['device_movement_patterns'].items():
                if device_pattern['device_type'] == 'laptop' and device_pattern['typical_locations']:
                    latest_laptop_loc = device_pattern['typical_locations'][-1]
                    laptop_locations.append(latest_laptop_loc)
                elif device_pattern['device_type'] == 'mobile' and device_pattern['typical_locations']:
                    latest_mobile_loc = device_pattern['typical_locations'][-1]
                    mobile_locations.append(latest_mobile_loc)
            
            # Check distance between mobile and laptop
            if laptop_locations and mobile_locations:
                laptop_loc = laptop_locations[-1]
                distance = self._calculate_distance(
                    current_location['latitude'], current_location['longitude'],
                    laptop_loc['latitude'], laptop_loc['longitude']
                )
                
                # If mobile is very far from laptop and it's during class hours
                current_hour = datetime.now(timezone.utc).hour
                if distance > 1000 and (9 <= current_hour <= 17):  # 1km during class hours
                    return True
        
        return False
    
    def _check_daily_pattern_anomaly(self, email, device_data, location_data, current_time):
        profile = self.user_profiles[email]
        hour = current_time.hour
        current_section = self._get_campus_section(location_data)
        
        # Determine time period
        if 6 <= hour < 12:
            period = 'morning_locations'
        elif 12 <= hour < 18:
            period = 'afternoon_locations' 
        elif 18 <= hour < 24:
            period = 'evening_locations'
        else:
            period = 'night_locations'
        
        # Check if current location matches typical pattern for this period
        if period in profile['typical_locations']:
            typical_loc = profile['typical_locations'][period]
            if typical_loc['count'] > 5:  # Only check if we have enough data
                distance = self._calculate_distance(
                    location_data['latitude'], location_data['longitude'],
                    typical_loc['latitude'], typical_loc['longitude']
                )
                
                # If significantly far from typical location
                if distance > 500:  # 500 meters
                    return f"Unusual location for {period.replace('_', ' ')}"
        
        return None
    
    def _calculate_distance(self, lat1, lng1, lat2, lng2):
        # Simple distance calculation (approximate)
        lat_diff = (lat2 - lat1) * 111000  # meters per degree latitude
        lng_diff = (lng2 - lng1) * 111000 * np.cos(np.radians(lat1))
        return np.sqrt(lat_diff**2 + lng_diff**2)
    
    def get_learning_progress(self, email):
        if email in self.user_profiles:
            return self.user_profiles[email]['learning_progress']
        return 0
    
    def get_behavior_summary(self, email):
        if email in self.user_profiles:
            profile = self.user_profiles[email]
            return {
                'learning_progress': profile['learning_progress'],
                'behavior_learned': profile['behavior_learned'],
                'schedule_consistency': profile['schedule_consistency'],
                'learned_patterns': len(profile['typical_locations']),
                'devices_analyzed': len(profile['device_movement_patterns'])
            }
        return None

class UserModel:
    def __init__(self, db):
        self.db = db
        self.users = db.users
    
    def create_user(self, email, password):
        user = {
            'email': email,
            'password': password,
            'created_at': datetime.now(timezone.utc),
            'devices': []
        }
        return self.users.insert_one(user)
    
    def find_user(self, email):
        return self.users.find_one({'email': email})
    
    def find_user_by_device_id(self, device_id):
        """Find which user owns a specific device"""
        return self.users.find_one({'devices.device_id': device_id})
    
    def add_device(self, email, device_data):
        try:
            # Check if device already exists in ANY user account
            existing_owner = self.find_user_by_device_id(device_data['device_id'])
            if existing_owner:
                return {
                    'modified_count': 0, 
                    'exists': True,
                    'owner_email': existing_owner['email'],
                    'error': f'Device already registered to {existing_owner["email"]}'
                }
            
            device_data['created_at'] = datetime.now(timezone.utc)
            device_data['last_updated'] = datetime.now(timezone.utc)
            device_data['owner_email'] = email  # Track the owner
            
            result = self.users.update_one(
                {'email': email},
                {'$push': {'devices': device_data}}
            )
            return {
                'modified_count': result.modified_count, 
                'exists': False,
                'owner_email': email
            }
        except Exception as e:
            if 'duplicate key error' in str(e).lower():
                # Device already exists in another account (unique index violation)
                existing_owner = self.find_user_by_device_id(device_data['device_id'])
                owner_email = existing_owner['email'] if existing_owner else 'another user'
                return {
                    'modified_count': 0,
                    'exists': True,
                    'owner_email': owner_email,
                    'error': f'Device already registered to {owner_email}'
                }
            raise e
    
    def update_device_location(self, email, device_id, location_data):
        # Verify the device belongs to this user before updating
        device = self.find_device_by_id(email, device_id)
        if not device:
            return {'modified_count': 0, 'error': 'Device not found or not owned by user'}
        
        return self.users.update_one(
            {'email': email, 'devices.device_id': device_id},
            {'$set': {
                'devices.$.last_location': location_data,
                'devices.$.last_updated': datetime.now(timezone.utc)
            }}
        )
    
    def get_user_devices(self, email):
        user = self.users.find_one({'email': email})
        return user.get('devices', []) if user else []
    
    def find_device_by_id(self, email, device_id):
        user = self.users.find_one({'email': email, 'devices.device_id': device_id})
        if user:
            for device in user.get('devices', []):
                if device['device_id'] == device_id:
                    return device
        return None
    
    def check_device_exists(self, email, device_id):
        device = self.find_device_by_id(email, device_id)
        return device is not None
    
    def check_device_exists_globally(self, device_id):
        """Check if device exists in ANY user account"""
        owner = self.find_user_by_device_id(device_id)
        return owner is not None
    
    def get_device_owner(self, device_id):
        """Get the email of the user who owns this device"""
        owner = self.find_user_by_device_id(device_id)
        return owner['email'] if owner else None
    
    def create_or_update_device(self, email, device_data, location_data):
        """Unified method to create or update device with location - with device uniqueness check"""
        device_id = device_data['device_id']
        
        # Check if device exists globally (in any account)
        existing_owner = self.find_user_by_device_id(device_id)
        if existing_owner:
            if existing_owner['email'] != email:
                # Device belongs to another user
                return {
                    'action': 'rejected', 
                    'device_id': device_id, 
                    'error': f'Device already registered to {existing_owner["email"]}',
                    'owner_email': existing_owner['email']
                }
            else:
                # Device exists and belongs to this user - update location
                result = self.update_device_location(email, device_id, location_data)
                return {
                    'action': 'updated', 
                    'device_id': device_id, 
                    'modified_count': result.modified_count,
                    'owner_email': email
                }
        else:
            # Create new device - this will be blocked by unique index if device exists
            try:
                device_data['last_location'] = location_data
                device_data['created_at'] = datetime.now(timezone.utc)
                device_data['last_updated'] = datetime.now(timezone.utc)
                device_data['owner_email'] = email
                
                result = self.add_device(email, device_data)
                if result['exists']:
                    return {
                        'action': 'rejected',
                        'device_id': device_id,
                        'error': result['error'],
                        'owner_email': result['owner_email']
                    }
                
                return {
                    'action': 'created', 
                    'device_id': device_id, 
                    'modified_count': result['modified_count'],
                    'owner_email': email
                }
            except Exception as e:
                if 'duplicate key error' in str(e).lower():
                    existing_owner = self.find_user_by_device_id(device_id)
                    owner_email = existing_owner['email'] if existing_owner else 'another user'
                    return {
                        'action': 'rejected',
                        'device_id': device_id,
                        'error': f'Device already registered to {owner_email}',
                        'owner_email': owner_email
                    }
                raise e

class AlertModel:
    def __init__(self, db):
        self.db = db
        self.alerts = db.alerts
    
    def create_alert(self, alert_data):
        alert_data['created_at'] = datetime.now(timezone.utc)
        alert_data['resolved'] = False
        return self.alerts.insert_one(alert_data)
    
    def get_user_alerts(self, email):
        alerts = list(self.alerts.find({'user_email': email}).sort('created_at', -1))
        # Convert ObjectId to string for JSON serialization
        for alert in alerts:
            if '_id' in alert:
                alert['_id'] = str(alert['_id'])
        return alerts

class AnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.is_fitted = False
        self.training_data = []
    
    def extract_features(self, device_data):
        features = []
        current_time = datetime.now(timezone.utc)
        
        if device_data.get('last_location'):
            lat = device_data['last_location'].get('latitude', 0)
            lon = device_data['last_location'].get('longitude', 0)
            features.extend([lat, lon])
        
        if device_data.get('last_updated'):
            if isinstance(device_data['last_updated'], datetime):
                time_diff = (current_time - device_data['last_updated']).total_seconds() / 3600
            else:
                time_diff = 0
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

# Initialize models - MOVED AFTER db DEFINITION
user_model = UserModel(db)
alert_model = AlertModel(db)
anomaly_detector = AnomalyDetector()
behavior_analyzer = BehaviorAnalyzer()

def get_client_ip():
    if request.environ.get('HTTP_X_FORWARDED_FOR'):
        return request.environ['HTTP_X_FORWARDED_FOR'].split(',')[0]
    else:
        return request.environ.get('HTTP_X_REAL_IP', request.remote_addr)

def detect_mobile_device(user_agent):
    mobile_indicators = ['mobile', 'android', 'iphone', 'ipad', 'tablet']
    user_agent_lower = user_agent.lower()
    return any(indicator in user_agent_lower for indicator in mobile_indicators)

def generate_device_id(email, user_agent, client_ip):
    """Generate consistent device ID across different login sessions"""
    # Use a combination that's stable for the same device
    device_string = f"{email}_{user_agent}"
    return hashlib.md5(device_string.encode()).hexdigest()

@app.route('/api/register', methods=['POST'])
@cross_origin()
def register():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400
        
        if user_model.find_user(email):
            return jsonify({'error': 'User already exists'}), 400
        
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        user_model.create_user(email, hashed_password)
        
        return jsonify({'message': 'User created successfully'}), 201
    except Exception as e:
        logging.error(f"Registration error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/login', methods=['POST'])
@cross_origin()
def login():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400
        
        user = user_model.find_user(email)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        if user['password'] != hashed_password:
            return jsonify({'error': 'Invalid password'}), 401
        
        client_ip = get_client_ip()
        user_agent = request.headers.get('User-Agent', '')
        
        # FIXED: Use consistent device ID generation
        device_id = generate_device_id(email, user_agent, client_ip)
        
        # Check if device exists globally
        existing_owner = user_model.find_user_by_device_id(device_id)
        device_owned_by_current_user = existing_owner and existing_owner['email'] == email
        
        # FIXED: Better device detection logic
        existing_device = user_model.find_device_by_id(email, device_id)
        
        if existing_device:
            # Device exists and belongs to this user
            device_info = {
                'needs_setup': False,
                'device_id': device_id,
                'device_type': existing_device.get('device_type', 'unknown'),
                'device_name': existing_device.get('device_name', 'Existing Device'),
                'is_mobile': detect_mobile_device(user_agent),
                'owned_by_current_user': True
            }
            print(f"✅ Device found in user account: {device_id}")
        elif existing_owner:
            # Device exists but belongs to another user
            device_info = {
                'needs_setup': True,
                'device_id': device_id,
                'device_type': 'unknown',
                'device_name': 'New Device',
                'is_mobile': detect_mobile_device(user_agent),
                'owned_by_current_user': False,
                'current_owner': existing_owner['email'],
                'error': f'This device is already registered to {existing_owner["email"]}'
            }
            print(f"⚠️ Device owned by another user: {device_id}")
        else:
            # New device - needs setup
            is_mobile = detect_mobile_device(user_agent)
            device_type = 'mobile' if is_mobile else 'laptop/desktop'
            
            device_info = {
                'needs_setup': True,
                'device_id': device_id,
                'ip_address': client_ip,
                'device_type': device_type,
                'user_agent': user_agent,
                'is_mobile': is_mobile,
                'owned_by_current_user': False
            }
            print(f"🆕 New device detected: {device_id}")
        
        return jsonify({
            'message': 'Login successful', 
            'email': email,
            'device_info': device_info
        }), 200
    except Exception as e:
        logging.error(f"Login error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/add_device', methods=['POST'])
@cross_origin()
def add_device():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        email = data.get('email')
        device_data = data.get('device_data')
        
        if not device_data:
            return jsonify({'error': 'No device data provided'}), 400
        
        # Check if device already exists globally
        existing_owner = user_model.find_user_by_device_id(device_data.get('device_id'))
        if existing_owner:
            if existing_owner['email'] == email:
                return jsonify({'error': 'Device already exists in your account'}), 400
            else:
                return jsonify({
                    'error': f'Device already registered to {existing_owner["email"]}',
                    'owner_email': existing_owner['email']
                }), 400
        
        result = user_model.add_device(email, device_data)
        
        if result['exists']:
            return jsonify({
                'error': result['error'],
                'owner_email': result['owner_email']
            }), 400
        
        if result['modified_count'] == 0:
            return jsonify({'error': 'Failed to add device'}), 500
        
        # Initialize behavior learning for this user
        behavior_analyzer.initialize_user_learning(email)
        
        # Create welcome alert
        alert_data = {
            'user_email': email,
            'type': 'device_added',
            'message': f"New device '{device_data.get('device_name', 'Unknown')}' was added to your account",
            'device_id': device_data.get('device_id')
        }
        alert_model.create_alert(alert_data)
        
        return jsonify({'message': 'Device added successfully'}), 200
    except Exception as e:
        logging.error(f"Add device error: {str(e)}")
        if 'duplicate key error' in str(e).lower():
            existing_owner = user_model.find_user_by_device_id(device_data.get('device_id'))
            owner_email = existing_owner['email'] if existing_owner else 'another user'
            return jsonify({
                'error': f'Device already registered to {owner_email}',
                'owner_email': owner_email
            }), 400
        return jsonify({'error': str(e)}), 500

@app.route('/api/update_device_location', methods=['POST'])
@cross_origin()
def update_device_location():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        email = data.get('email')
        device_id = data.get('device_id')
        location = data.get('location')
        
        if not all([email, device_id, location]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # First, check if device exists and belongs to this user
        device = user_model.find_device_by_id(email, device_id)
        
        if not device:
            # Check if device exists globally
            existing_owner = user_model.find_user_by_device_id(device_id)
            if existing_owner:
                return jsonify({
                    'error': f'Device not found in your account. It is registered to {existing_owner["email"]}', 
                    'code': 'DEVICE_OWNED_BY_OTHER_USER',
                    'owner_email': existing_owner['email'],
                    'device_id': device_id
                }), 403
            else:
                return jsonify({
                    'error': 'Device not found', 
                    'code': 'DEVICE_NOT_FOUND',
                    'device_id': device_id
                }), 404
        else:
            # Update existing device location
            result = user_model.update_device_location(email, device_id, location)
            
            if result.modified_count == 0:
                return jsonify({'error': 'Device not found or no changes made'}), 404
            
            # BEHAVIOR LEARNING: Record this location for behavior analysis
            behavior_analyzer.record_location_behavior(email, device, location)
            
            # Check for behavior anomalies
            anomalies, analysis_message = behavior_analyzer.detect_behavior_anomaly(email, device, location)
            
            if anomalies:
                for anomaly in anomalies:
                    alert_data = {
                        'user_email': email,
                        'type': 'suspicious_behavior',
                        'message': f"🚨 BEHAVIOR ALERT: {anomaly}",
                        'device_id': device_id,
                        'severity': 'high'
                    }
                    alert_model.create_alert(alert_data)
            
            # Check for ML anomalies
            if device:
                anomaly_detector.add_training_data(device)
                if anomaly_detector.detect_anomaly(device):
                    alert_data = {
                        'user_email': email,
                        'type': 'suspicious_activity',
                        'message': f"Unusual location activity detected for device '{device.get('device_name', 'Unknown')}'",
                        'device_id': device_id
                    }
                    alert_model.create_alert(alert_data)
            
            return jsonify({
                'message': 'Location updated successfully',
                'behavior_analysis': analysis_message,
                'anomalies_detected': len(anomalies) if anomalies else 0
            }), 200
    except Exception as e:
        logging.error(f"Update location error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/create_or_update_device', methods=['POST'])
@cross_origin()
def create_or_update_device():
    """Unified endpoint for device creation and location update"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        email = data.get('email')
        device_data = data.get('device_data')
        location_data = data.get('location_data')
        
        if not all([email, device_data, location_data]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Use the unified method
        result = user_model.create_or_update_device(email, device_data, location_data)
        
        if result['action'] == 'created':
            # Initialize behavior learning for this user
            behavior_analyzer.initialize_user_learning(email)
            
            # Record initial location for behavior analysis
            device_for_analysis = {
                'device_id': device_data['device_id'],
                'device_type': device_data.get('device_type', 'unknown'),
                'device_name': device_data.get('device_name', 'Unknown')
            }
            behavior_analyzer.record_location_behavior(email, device_for_analysis, location_data)
            
            # Create device added alert
            alert_data = {
                'user_email': email,
                'type': 'device_added',
                'message': f"New device '{device_data.get('device_name', 'Unknown')}' was added to your account",
                'device_id': device_data.get('device_id')
            }
            alert_model.create_alert(alert_data)
            
            return jsonify({
                'message': 'Device created successfully',
                'action': 'created',
                'device_id': result['device_id']
            }), 201
        elif result['action'] == 'updated':
            # Record location for behavior analysis
            device = user_model.find_device_by_id(email, device_data['device_id'])
            if device:
                behavior_analyzer.record_location_behavior(email, device, location_data)
            
            return jsonify({
                'message': 'Device location updated successfully',
                'action': 'updated',
                'device_id': result['device_id']
            }), 200
        else:  # rejected
            return jsonify({
                'error': result['error'],
                'owner_email': result['owner_email'],
                'device_id': result['device_id']
            }), 400
            
    except Exception as e:
        logging.error(f"Create/update device error: {str(e)}")
        if 'duplicate key error' in str(e).lower():
            existing_owner = user_model.find_user_by_device_id(device_data.get('device_id'))
            owner_email = existing_owner['email'] if existing_owner else 'another user'
            return jsonify({
                'error': f'Device already registered to {owner_email}',
                'owner_email': owner_email
            }), 400
        return jsonify({'error': str(e)}), 500

@app.route('/api/devices/<email>', methods=['GET'])
@cross_origin()
def get_devices(email):
    try:
        devices = user_model.get_user_devices(email)
        
        # Convert datetime objects to strings
        for device in devices:
            if 'last_updated' in device and isinstance(device['last_updated'], datetime):
                device['last_updated'] = device['last_updated'].isoformat()
            if 'created_at' in device and isinstance(device['created_at'], datetime):
                device['created_at'] = device['created_at'].isoformat()
        
        return jsonify(devices), 200
    except Exception as e:
        logging.error(f"Get devices error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/check_device/<email>/<device_id>', methods=['GET'])
@cross_origin()
def check_device(email, device_id):
    try:
        exists = user_model.check_device_exists(email, device_id)
        return jsonify({'exists': exists}), 200
    except Exception as e:
        logging.error(f"Check device error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/check_device_global/<device_id>', methods=['GET'])
@cross_origin()
def check_device_global(device_id):
    """Check if device exists in ANY user account and return owner info"""
    try:
        exists = user_model.check_device_exists_globally(device_id)
        owner_email = user_model.get_device_owner(device_id) if exists else None
        return jsonify({
            'exists': exists,
            'owner_email': owner_email
        }), 200
    except Exception as e:
        logging.error(f"Check device global error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/<email>', methods=['GET'])
@cross_origin()
def get_alerts(email):
    try:
        alerts = alert_model.get_user_alerts(email)
        return jsonify(alerts), 200
    except Exception as e:
        logging.error(f"Get alerts error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/behavior/progress/<email>', methods=['GET'])
@cross_origin()
def get_behavior_progress(email):
    """Get behavior learning progress for a user"""
    try:
        progress = behavior_analyzer.get_learning_progress(email)
        summary = behavior_analyzer.get_behavior_summary(email)
        
        return jsonify({
            'learning_progress': progress,
            'behavior_summary': summary
        }), 200
    except Exception as e:
        logging.error(f"Get behavior progress error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/behavior/simulate_complete/<email>', methods=['POST'])
@cross_origin()
def simulate_learning_complete(email):
    """Simulate complete learning for demo purposes"""
    try:
        if email in behavior_analyzer.user_profiles:
            behavior_analyzer.user_profiles[email]['learning_progress'] = 100
            behavior_analyzer._finalize_behavior_learning(email)
            
            return jsonify({
                'message': 'Learning simulated as complete',
                'behavior_learned': True
            }), 200
        else:
            return jsonify({'error': 'User not found in behavior analyzer'}), 404
    except Exception as e:
        logging.error(f"Simulate learning complete error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
@cross_origin()
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now(timezone.utc).isoformat()}), 200

@app.route('/api/test', methods=['GET'])
@cross_origin()
def test_connection():
    return jsonify({'message': 'Backend is working!', 'timestamp': datetime.now(timezone.utc).isoformat()}), 200

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Backend URL: http://localhost:5000")
    print("API Health Check: http://localhost:5000/api/health")
    print("API Test: http://localhost:5000/api/test")
    print("✅ Device uniqueness enforcement: ENABLED")
    print("🎯 Behavior Learning Engine: ENABLED")
    print("🔧 FIXED: Consistent device ID generation")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)