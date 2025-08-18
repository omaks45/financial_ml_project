import os
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DatabaseConfig:
    def __init__(self):
        self.host = os.getenv('DB_HOST', 'localhost')
        self.user = os.getenv('DB_USER', 'root')
        self.password = os.getenv('DB_PASSWORD', 'your_password')
        self.database = os.getenv('DB_NAME', 'ml_financial_analysis')
    
    def get_connection(self):
        """Get MySQL database connection"""
        try:
            connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                autocommit=True
            )
            if connection.is_connected():
                print(f"Successfully connected to {self.database} database")
                return connection
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            return None
    
    def test_connection(self):
        """Test database connection"""
        conn = self.get_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DATABASE();")
            result = cursor.fetchone()
            print(f"Connected to database: {result[0]}")
            
            # Show available tables
            cursor.execute("SHOW TABLES;")
            tables = cursor.fetchall()
            print("Available tables:")
            for table in tables:
                print(f"  - {table[0]}")
            
            conn.close()
            return True
        return False