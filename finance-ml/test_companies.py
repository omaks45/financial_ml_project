#!/usr/bin/env python3
"""
debug_companies.py
Quick debugging script to test company loading
"""

import os
import sys
import logging
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def debug_company_loading():
    """Debug the company loading process"""
    
    print("=" * 60)
    print("DEBUGGING COMPANY LOADING PROCESS")
    print("=" * 60)
    
    # 1. Check data directory
    data_dir = "data"
    print(f"\n1. CHECKING DATA DIRECTORY: {data_dir}")
    
    if os.path.exists(data_dir):
        print(f"✓ Directory exists")
        files = os.listdir(data_dir)
        if files:
            print(f"✓ Files found: {len(files)}")
            for file in files:
                file_path = os.path.join(data_dir, file)
                file_size = os.path.getsize(file_path)
                print(f"  - {file} ({file_size} bytes)")
        else:
            print("✗ Directory is empty")
    else:
        print(f"✗ Directory does not exist")
    
    # 2. Look for Excel files
    print(f"\n2. LOOKING FOR EXCEL FILES")
    excel_files = []
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith(('.xlsx', '.xls')):
                excel_files.append(file)
    
    if excel_files:
        print(f"✓ Found {len(excel_files)} Excel files:")
        for file in excel_files:
            print(f"  - {file}")
        
        # 3. Try to read the first Excel file
        print(f"\n3. READING FIRST EXCEL FILE: {excel_files[0]}")
        try:
            file_path = os.path.join(data_dir, excel_files[0])
            df = pd.read_excel(file_path)
            
            print(f"✓ Successfully loaded Excel file")
            print(f"✓ Shape: {df.shape}")
            print(f"✓ Columns: {list(df.columns)}")
            
            print(f"\n4. SAMPLE DATA:")
            print(df.head())
            
            # 5. Try to extract companies
            print(f"\n5. EXTRACTING COMPANY IDs:")
            potential_columns = ['Symbol', 'company_id', 'Company ID', 'ID', 'SYMBOL', 'ticker', 'Ticker']
            
            found_companies = False
            for col in potential_columns:
                if col in df.columns:
                    companies = df[col].dropna().tolist()
                    print(f"✓ Found companies in column '{col}': {len(companies)}")
                    print(f"✓ First 10: {companies[:10]}")
                    found_companies = True
                    break
            
            if not found_companies:
                print("✗ No suitable company ID column found")
                print("Available columns:", list(df.columns))
                
                # Try first column
                first_col = df.columns[0]
                companies = df[first_col].dropna().tolist()
                print(f"Trying first column '{first_col}': {companies[:10]}")
        
        except Exception as e:
            print(f"✗ Error reading Excel file: {e}")
    
    else:
        print("✗ No Excel files found")
        print("\n3. CREATING SAMPLE EXCEL FILE")
        
        # Create sample data
        sample_data = {
            'Symbol': ['TCS', 'HDFCBANK', 'DMART', 'INFY', 'RELIANCE'] + 
                     [f'COMPANY{i}' for i in range(6, 101)]  # 100 companies total
        }
        
        try:
            os.makedirs(data_dir, exist_ok=True)
            df = pd.DataFrame(sample_data)
            sample_path = os.path.join(data_dir, 'Nifty100Companies.xlsx')
            df.to_excel(sample_path, index=False)
            
            print(f"✓ Created sample file: {sample_path}")
            print(f"✓ Contains {len(sample_data['Symbol'])} companies")
            
        except Exception as e:
            print(f"✗ Error creating sample file: {e}")
    
    # 6. Test the actual CompanyDataLoader
    print(f"\n6. TESTING CompanyDataLoader:")
    try:
        # Import your fixed loader
        sys.path.append('.')
        from data.company_loader import CompanyDataLoader
        
        loader = CompanyDataLoader()
        companies = loader.load_companies_from_excel()
        
        print(f"✓ Loader returned {len(companies)} companies")
        print(f"✓ Companies: {companies[:10]}{'...' if len(companies) > 10 else ''}")
        
    except Exception as e:
        print(f"✗ Error with CompanyDataLoader: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("DEBUGGING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    debug_company_loading()