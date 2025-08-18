from config.database import DatabaseConfig
from database.operations import DatabaseOperations

def test_database_setup():
    """Test database connection and operations"""
    print("=== Testing Database Setup ===")
    
    # Test connection
    db_config = DatabaseConfig()
    if db_config.test_connection():
        print("✓ Database connection successful")
    else:
        print("✗ Database connection failed")
        return
    
    # Test operations
    db_ops = DatabaseOperations()
    
    # Test getting company list
    companies = db_ops.get_company_list()
    print(f"✓ Found {len(companies)} companies in database")
    if companies:
        print("Sample companies:")
        for i, comp in enumerate(companies[:5]):
            print(f"  {comp['id']}: {comp['name']}")
    
    # Test inserting sample data
    test_company = "TCS"
    test_pros = ["Good ROE track record", "Strong dividend payout"]
    test_cons = ["High stock valuation", "Slow sales growth"]
    
    if db_ops.insert_pros_cons(test_company, test_pros, test_cons):
        print(f"✓ Successfully inserted test data for {test_company}")
    else:
        print(f"✗ Failed to insert test data for {test_company}")

if __name__ == "__main__":
    test_database_setup()