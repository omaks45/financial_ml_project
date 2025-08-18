# test_imports.py
try:
    import mysql.connector
    print("mysql.connector imported successfully")
except ImportError as e:
    print(f"mysql.connector failed: {e}")

try:
    import dotenv
    print("dotenv imported successfully") 
except ImportError as e:
    print(f"dotenv failed: {e}")