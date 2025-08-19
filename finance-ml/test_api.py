#!/usr/bin/env python3
"""
Test script for API connectivity - Day 1 Task 4
Tests the API client with sample companies and validates data extraction
"""

import sys
import os
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.api_client import create_api_client
import logging

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_api_connectivity():
    """
    Test API connectivity and data fetching functionality
    """
    print("=" * 60)
    print("FINANCIAL ML PROJECT - DAY 1 API CONNECTIVITY TEST")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize API client
    try:
        api = create_api_client()
        print("✓ API client initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize API client: {e}")
        return False
    
    # Test 1: Basic connectivity test
    print("\n" + "-" * 40)
    print("TEST 1: Basic Connectivity Test")
    print("-" * 40)
    
    connectivity_result = api.test_connectivity('TCS')
    if not connectivity_result:
        print("✗ Basic connectivity test failed. Exiting...")
        return False
    
    # Test 2: Fetch single company data
    print("\n" + "-" * 40)
    print("TEST 2: Single Company Data Fetch")
    print("-" * 40)
    
    test_company = 'HDFCBANK'
    print(f"Fetching data for: {test_company}")
    
    try:
        company_data = api.fetch_company_data(test_company)
        if company_data:
            print(f"✓ Successfully fetched data for {test_company}")
            print(f"  Data structure keys: {list(company_data.keys())}")
            
            # Display sample financial metrics
            if 'financial_metrics' in company_data:
                print("  Sample financial metrics:")
                for metric, value in company_data['financial_metrics'].items():
                    if value is not None:
                        print(f"    {metric}: {value}")
        else:
            print(f"✗ Failed to fetch data for {test_company}")
            return False
    except Exception as e:
        print(f"✗ Exception while fetching {test_company}: {e}")
        return False
    
    # Test 3: Multiple companies batch fetch
    print("\n" + "-" * 40)
    print("TEST 3: Multiple Companies Batch Fetch")
    print("-" * 40)
    
    sample_companies = ['TCS', 'DMART', 'INFY', 'SBILIFE', 'WIPRO']
    print(f"Testing batch fetch with companies: {sample_companies}")
    
    try:
        batch_results = api.fetch_multiple_companies(sample_companies, batch_size=2)
        
        successful_fetches = 0
        failed_fetches = 0
        
        print("\nBatch fetch results:")
        for company_id, data in batch_results.items():
            if data:
                print(f"  ✓ {company_id}: SUCCESS")
                successful_fetches += 1
            else:
                print(f"  ✗ {company_id}: FAILED")
                failed_fetches += 1
        
        print(f"\nBatch Summary:")
        print(f"  Total companies: {len(sample_companies)}")
        print(f"  Successful: {successful_fetches}")
        print(f"  Failed: {failed_fetches}")
        print(f"  Success rate: {(successful_fetches/len(sample_companies)*100):.1f}%")
        
        if successful_fetches == 0:
            print("✗ All batch fetches failed")
            return False
        
    except Exception as e:
        print(f"✗ Exception during batch fetch: {e}")
        return False
    
    # Test 4: Data structure validation
    print("\n" + "-" * 40)
    print("TEST 4: Data Structure Validation")
    print("-" * 40)
    
    try:
        # Use successful data from previous test
        sample_data = None
        for company_id, data in batch_results.items():
            if data:
                sample_data = data
                print(f"Validating data structure using {company_id} data")
                break
        
        if sample_data:
            required_keys = ['company_id', 'company_info', 'financial_metrics', 'raw_data']
            
            print("Checking required data structure keys:")
            all_keys_present = True
            for key in required_keys:
                if key in sample_data:
                    print(f"  ✓ {key}: Present")
                else:
                    print(f"  ✗ {key}: Missing")
                    all_keys_present = False
            
            if all_keys_present:
                print("✓ Data structure validation passed")
            else:
                print("✗ Data structure validation failed")
                return False
        else:
            print("✗ No valid data available for structure validation")
            return False
            
    except Exception as e:
        print(f"✗ Exception during data structure validation: {e}")
        return False
    
    # Test 5: Save sample data
    print("\n" + "-" * 40)
    print("TEST 5: Data Export Test")
    print("-" * 40)
    
    try:
        # Save batch results to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"sample_data_{timestamp}.json"
        
        api.save_data_to_file(batch_results, filename)
        
        # Verify file was created
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"✓ Sample data exported to {filename}")
            print(f"  File size: {file_size} bytes")
            
            # Clean up test file
            os.remove(filename)
            print(f"  Test file cleaned up")
        else:
            print(f"✗ Failed to create export file {filename}")
            return False
            
    except Exception as e:
        print(f"✗ Exception during data export test: {e}")
        return False
    
    # Final summary
    print("\n" + "=" * 60)
    print("API CONNECTIVITY TEST RESULTS")
    print("=" * 60)
    print("✓ All tests passed successfully!")
    print("✓ API client is ready for Day 2 database integration")
    print("✓ Financial data extraction is working correctly")
    print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return True

def display_api_info():
    """
    Display API configuration information
    """
    api = create_api_client()
    print("\n" + "-" * 40)
    print("API CONFIGURATION INFO")
    print("-" * 40)
    print(f"Base URL: {api.base_url}")
    print(f"API Key: {'*' * (len(api.api_key) - 4) + api.api_key[-4:]}")
    print(f"Rate limit: {api.min_request_interval} seconds between requests")
    print("Available sample companies:", api.get_sample_companies()[:10], "...")

def main():
    """
    Main function to run all tests
    """
    try:
        display_api_info()
        
        success = test_api_connectivity()
        
        if success:
            print("\nDay 1 Task 3 & 4 completed successfully!")
            print("Ready to proceed to Day 2: Database Integration")
            return 0
        else:
            print("\nTests failed. Please check the API configuration and network connectivity.")
            return 1
            
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n\nUnexpected error during testing: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)