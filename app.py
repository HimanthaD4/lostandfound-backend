from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os

# Configure logging
import logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.config["SECRET_KEY"] = "your-secret-key-here"

# Configure CORS properly for all origins during development
cors_origins = [
    "http://localhost:3000", 
    "http://192.168.1.125:3000", 
    "http://127.0.0.1:3000", 
    "http://10.185.94.208:3000",
    "https://lostandfound-client-nu.vercel.app"
]

CORS(app, resources={
    r"/api/*": {
        "origins": cors_origins,
        "methods": ["GET", "POST", "PUT", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

# MongoDB connection
mongodb_uri = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/')
client = MongoClient(mongodb_uri, connectTimeoutMS=30000, socketTimeoutMS=30000)
db = client['device_tracker']

# Create unique index for device_id across all users
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
            device_data['owner_email'] = email
            
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
        
        # CRITICAL FIX: Ensure location data is specific to this device
        location_data['device_id'] = device_id
        location_data['timestamp'] = datetime.now(timezone.utc)
        
        result = self.users.update_one(
            {'email': email, 'devices.device_id': device_id},
            {'$set': {
                'devices.$.last_location': location_data,
                'devices.$.last_updated': datetime.now(timezone.utc)
            }}
        )
        
        return result
    
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
        """Unified method to create or update device with location"""
        device_id = device_data['device_id']
        
        # Check if device exists globally (in any account)
        existing_owner = self.find_user_by_device_id(device_id)
        if existing_owner:
            if existing_owner['email'] != email:
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
            # Create new device
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

# Initialize models
user_model = UserModel(db)

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
    device_string = f"{email}_{user_agent}"
    return hashlib.md5(device_string.encode()).hexdigest()

@app.route('/')
@cross_origin()
def home():
    return jsonify({
        'message': 'Device Tracker Backend is running!',
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat()
    }), 200

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
        
        device_id = generate_device_id(email, user_agent, client_ip)
        
        existing_owner = user_model.find_user_by_device_id(device_id)
        device_owned_by_current_user = existing_owner and existing_owner['email'] == email
        
        existing_device = user_model.find_device_by_id(email, device_id)
        
        if existing_device:
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
        
        device = user_model.find_device_by_id(email, device_id)
        
        if not device:
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
            # CRITICAL FIX: Ensure location data is specific to this device
            location['device_id'] = device_id
            location['timestamp'] = datetime.now(timezone.utc).isoformat()
            
            result = user_model.update_device_location(email, device_id, location)
            
            if result.modified_count == 0:
                return jsonify({'error': 'Device not found or no changes made'}), 404
            
            return jsonify({
                'message': 'Location updated successfully',
                'device_id': device_id
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
        
        # CRITICAL FIX: Ensure location data is specific to this device
        location_data['device_id'] = device_data['device_id']
        location_data['timestamp'] = datetime.now(timezone.utc).isoformat()
        
        result = user_model.create_or_update_device(email, device_data, location_data)
        
        if result['action'] == 'created':
            return jsonify({
                'message': 'Device created successfully',
                'action': 'created',
                'device_id': result['device_id']
            }), 201
        elif result['action'] == 'updated':
            return jsonify({
                'message': 'Device location updated successfully',
                'action': 'updated',
                'device_id': result['device_id']
            }), 200
        else:
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

@app.route('/api/health', methods=['GET'])
@cross_origin()
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now(timezone.utc).isoformat()}), 200

@app.route('/api/test', methods=['GET'])
@cross_origin()
def test_connection():
    return jsonify({'message': 'Backend is working!', 'timestamp': datetime.now(timezone.utc).isoformat()}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Flask server on port {port}...")
    print(f"Backend URL: http://0.0.0.0:{port}")
    print(f"API Health Check: http://0.0.0.0:{port}/api/health")
    print("✅ Device uniqueness enforcement: ENABLED")
    print("📍 Independent device location tracking: ENABLED")
    
    app.run(debug=False, host='0.0.0.0', port=port)