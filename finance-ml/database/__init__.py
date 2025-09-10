# Add these imports to database/__init__.py
from .ml_database_schema import MLDatabaseSchema, run_day2_tasks
from .enhanced_operations import EnhancedDatabaseOperations, create_enhanced_ml_database

__all__ = [
    'MLDatabaseSchema', 
    'run_day2_tasks',
    'EnhancedDatabaseOperations', 
    'create_enhanced_ml_database'
]