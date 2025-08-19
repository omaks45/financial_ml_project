"""
Database Models for Financial ML Analysis
Defines SQLAlchemy models for storing financial data and ML results
"""

from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime, Boolean, 
    JSON, ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import json

Base = declarative_base()

class Company(Base):
    """
    Model for storing basic company information
    """
    __tablename__ = 'companies'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    company_id = Column(String(20), unique=True, nullable=False, index=True)  # TCS, HDFCBANK, etc.
    company_name = Column(String(200), nullable=True)
    sector = Column(String(100), nullable=True)
    industry = Column(String(100), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    financial_data = relationship("FinancialData", back_populates="company", cascade="all, delete-orphan")
    ml_results = relationship("MLResult", back_populates="company", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Company(id='{self.company_id}', name='{self.company_name}')>"

class FinancialData(Base):
    """
    Model for storing raw financial data from API
    """
    __tablename__ = 'financial_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    company_id = Column(String(20), ForeignKey('companies.company_id'), nullable=False)
    
    # Financial Metrics (the key ones used for ML analysis)
    roe = Column(Float, nullable=True)  # Return on Equity
    roce = Column(Float, nullable=True)  # Return on Capital Employed
    debt_to_equity = Column(Float, nullable=True)
    current_ratio = Column(Float, nullable=True)
    book_value = Column(Float, nullable=True)
    eps = Column(Float, nullable=True)  # Earnings Per Share
    pe_ratio = Column(Float, nullable=True)
    dividend_yield = Column(Float, nullable=True)
    sales_growth = Column(Float, nullable=True)
    profit_growth = Column(Float, nullable=True)
    
    # Additional financial metrics
    revenue = Column(Float, nullable=True)
    net_income = Column(Float, nullable=True)
    total_assets = Column(Float, nullable=True)
    total_debt = Column(Float, nullable=True)
    market_cap = Column(Float, nullable=True)
    
    # Store complete raw API response as JSON
    raw_api_data = Column(JSON, nullable=True)
    
    # Data quality indicators
    data_completeness_score = Column(Float, nullable=True)  # 0-100% how much data we have
    last_updated_source = Column(DateTime, nullable=True)  # When source data was last updated
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    company = relationship("Company", back_populates="financial_data")
    
    # Indexes for better query performance
    __table_args__ = (
        Index('idx_company_created', 'company_id', 'created_at'),
        UniqueConstraint('company_id', name='uq_financial_data_company'),
    )
    
    def __repr__(self):
        return f"<FinancialData(company='{self.company_id}', roe={self.roe})>"
    
    def to_dict(self):
        """Convert financial data to dictionary for ML processing"""
        return {
            'roe': self.roe,
            'roce': self.roce,
            'debt_to_equity': self.debt_to_equity,
            'current_ratio': self.current_ratio,
            'book_value': self.book_value,
            'eps': self.eps,
            'pe_ratio': self.pe_ratio,
            'dividend_yield': self.dividend_yield,
            'sales_growth': self.sales_growth,
            'profit_growth': self.profit_growth,
            'revenue': self.revenue,
            'net_income': self.net_income,
            'total_assets': self.total_assets,
            'total_debt': self.total_debt,
            'market_cap': self.market_cap
        }

class MLResult(Base):
    """
    Model for storing ML analysis results - matches the existing 'ml' table
    This is the main table that the web frontend reads from
    """
    __tablename__ = 'ml'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    company_id = Column(String(20), ForeignKey('companies.company_id'), nullable=False, index=True)
    
    # ML Analysis Results
    analysis_title = Column(String(500), nullable=True)  # Dynamic title based on company analysis
    
    # Pros and Cons (JSON arrays to store up to 3 each)
    pros = Column(JSON, nullable=True)  # List of up to 3 pros
    cons = Column(JSON, nullable=True)  # List of up to 3 cons
    
    # ML Scores and Metrics
    overall_score = Column(Float, nullable=True)  # Overall financial health score (0-100)
    growth_score = Column(Float, nullable=True)
    profitability_score = Column(Float, nullable=True)
    financial_stability_score = Column(Float, nullable=True)
    
    # Analysis metadata
    analysis_version = Column(String(10), default="1.0")  # Track ML model version
    confidence_score = Column(Float, nullable=True)  # How confident we are in this analysis
    
    # Processing status
    status = Column(String(20), default='processing')  # processing, completed, failed
    error_message = Column(Text, nullable=True)  # Store any error messages
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    company = relationship("Company", back_populates="ml_results")
    
    # Ensure only one active ML result per company
    __table_args__ = (
        Index('idx_ml_company_status', 'company_id', 'status'),
        UniqueConstraint('company_id', name='uq_ml_result_company'),
    )
    
    def __repr__(self):
        return f"<MLResult(company='{self.company_id}', score={self.overall_score}, status='{self.status}')>"
    
    def get_pros_list(self):
        """Get pros as Python list"""
        if self.pros and isinstance(self.pros, str):
            try:
                return json.loads(self.pros)
            except:
                return []
        return self.pros or []
    
    def get_cons_list(self):
        """Get cons as Python list"""
        if self.cons and isinstance(self.cons, str):
            try:
                return json.loads(self.cons)
            except:
                return []
        return self.cons or []
    
    def set_pros_list(self, pros_list):
        """Set pros from Python list"""
        self.pros = pros_list[:3] if pros_list else []  # Limit to 3
    
    def set_cons_list(self, cons_list):
        """Set cons from Python list"""
        self.cons = cons_list[:3] if cons_list else []  # Limit to 3

class ProcessingLog(Base):
    """
    Model for tracking processing status and logs
    Useful for monitoring the one-by-one processing workflow
    """
    __tablename__ = 'processing_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    company_id = Column(String(20), nullable=False, index=True)
    
    # Processing stages
    stage = Column(String(50), nullable=False)  # 'api_fetch', 'ml_analysis', 'database_save'
    status = Column(String(20), nullable=False)  # 'started', 'completed', 'failed'
    
    # Details
    message = Column(Text, nullable=True)
    error_details = Column(Text, nullable=True)
    processing_time = Column(Float, nullable=True)  # Time taken in seconds
    
    # Batch information (for tracking batch processing)
    batch_id = Column(String(50), nullable=True)
    batch_position = Column(Integer, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Indexes for monitoring queries
    __table_args__ = (
        Index('idx_log_company_stage', 'company_id', 'stage', 'created_at'),
        Index('idx_log_batch', 'batch_id', 'batch_position'),
        Index('idx_log_status', 'status', 'created_at'),
    )
    
    def __repr__(self):
        return f"<ProcessingLog(company='{self.company_id}', stage='{self.stage}', status='{self.status}')>"

# Utility function to create all tables
def create_tables(engine):
    """
    Create all database tables
    
    Args:
        engine: SQLAlchemy engine instance
    """
    Base.metadata.create_all(engine)
    print("All database tables created successfully!")

# Utility function to drop all tables (for development)
def drop_tables(engine):
    """
    Drop all database tables (use carefully!)
    
    Args:
        engine: SQLAlchemy engine instance
    """
    Base.metadata.drop_all(engine)
    print("All database tables dropped!")

if __name__ == "__main__":
    # Example of how to use these models
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    # Create an in-memory SQLite database for testing
    engine = create_engine('sqlite:///test_financial_ml.db', echo=True)
    
    # Create tables
    create_tables(engine)
    
    # Create session
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Example: Create a company
    company = Company(
        company_id='TCS',
        company_name='Tata Consultancy Services',
        sector='Information Technology',
        industry='Software Services'
    )
    
    # Example: Add financial data
    financial_data = FinancialData(
        company_id='TCS',
        roe=25.4,
        roce=30.2,
        debt_to_equity=0.1,
        sales_growth=15.3,
        profit_growth=18.2
    )
    
    # Example: Add ML result
    ml_result = MLResult(
        company_id='TCS',
        analysis_title='TCS shows strong financial performance with excellent ROE',
        pros=['Strong ROE of 25.4%', 'Low debt-to-equity ratio', 'Good profit growth'],
        cons=['Could improve dividend yield'],
        overall_score=85.5,
        status='completed'
    )
    
    # Add to session and commit
    session.add_all([company, financial_data, ml_result])
    session.commit()
    
    print("Example data added successfully!")
    session.close()