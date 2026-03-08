import random
import bcrypt
from datetime import datetime, timedelta

def generate_otp():
    return str(random.randint(100000, 999999))

def hash_value(value):
    return bcrypt.hashpw(value.encode(), bcrypt.gensalt()).decode()

def verify_hash(value, hashed):
    return bcrypt.checkpw(value.encode(), hashed.encode())

def otp_expiry():
    return datetime.utcnow() + timedelta(minutes=5)