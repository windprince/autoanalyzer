import sys
import os
import pandas as pd
import numpy as np
import pytest
from io import StringIO

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DataProfiler import DataProfiler

class TestDataProfiler:
    
    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create a sample CSV file for testing"""
        csv_content = """
        id,name,age,salary,department,hire_date
        1,John Doe,32,75000,Engineering,2020-01-15
        2,Jane Smith,28,82000,Marketing,2019-03-20
        3,Bob Johnson,45,120000,Executive,2015-11-05
        4,Alice Brown,36,95000,Engineering,2018-07-30
        5,Charlie Wilson,29,67000,Marketing,2021-02-10
        6,Diana Miller,41,105000,Executive,2017-09-12
        7,Edward Davis,33,78000,Engineering,2019-05-08
        8,Fiona Garcia,27,65000,Marketing,2021-01-25
        9,George Thompson,52,135000,Executive,2010-08-01
        10,Hannah Martinez,30,72000,Engineering,2020-04-15
        """
        
        file_path = tmp_path / "sample.csv"
        with open(file_path, 'w') as f:
            f.write(csv_content.strip())
        
        return str(file_path)
    
    def test_initialization(self):
        """Test initializing the DataProfiler"""
        profiler = DataProfiler()
        assert profiler.data is None
        assert profiler.profile == {}
    
    def test_load_data(self, sample_csv):
        """Test loading data from CSV file"""
        profiler = DataProfiler()
        data = profiler.load_data(sample_csv, "csv")
        
        # Check that data was loaded correctly
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 10
        assert list(data.columns) == ['id', 'name', 'age', 'salary', 'department', 'hire_date']
    
    def test_column_type_detection(self, sample_csv):
        """Test detecting column types"""
        profiler = DataProfiler()
        profiler.load_data(sample_csv, "csv")
        
        # Set profile with column types
        profiler.profile = {"basic_stats": profiler._get_basic_stats()}
        column_types = profiler._detect_column_types()
        
        # Check column type detection
        assert column_types['id'] in ['numeric', 'categorical']
        assert column_types['name'] in ['categorical', 'text', 'object']
        assert column_types['age'] == 'numeric'
        assert column_types['salary'] == 'numeric'
        assert column_types['department'] == 'categorical'
        assert column_types['hire_date'] in ['datetime', 'categorical', 'object'] 
    
    def test_is_categorical(self):
        """Test categorical detection"""
        profiler = DataProfiler()
        
        # Test with clearly categorical data
        categorical_series = pd.Series(['A', 'B', 'C', 'A', 'B', 'A', 'C', 'B', 'A', 'B'])
        assert profiler._is_categorical(categorical_series) == True
        
        # Test with numeric data that looks categorical
        numeric_categorical = pd.Series([1, 2, 3, 1, 2, 1, 3, 2, 1, 2])
        assert profiler._is_categorical(numeric_categorical) == True
        
        # Test with data that is not categorical
        non_categorical = pd.Series(np.random.rand(100))
        assert profiler._is_categorical(non_categorical) == False
    
    def test_is_datetime(self):
        """Test datetime detection"""
        profiler = DataProfiler()
        
        # Test with date strings
        date_series = pd.Series(['2021-01-01', '2021-01-02', '2021-01-03'])
        assert profiler._is_datetime(date_series) == True
        
        # Test with non-date strings
        non_date_series = pd.Series(['abc', 'def', 'ghi'])
        assert profiler._is_datetime(non_date_series) == False
        
    def test_full_profile(self, sample_csv):
        """Test generating a full profile"""
        profiler = DataProfiler()
        profiler.load_data(sample_csv, "csv")
        profile = profiler.generate_profile()
        
        # Check that profile contains all the expected sections
        assert "basic_stats" in profile
        assert "column_types" in profile
        assert "missing_data" in profile
        assert "distribution_info" in profile
        assert "potential_relationships" in profile
        assert "data_quality_issues" in profile
        
        # Check basic stats
        assert profile["basic_stats"]["row_count"] == 10
        assert profile["basic_stats"]["column_count"] == 6