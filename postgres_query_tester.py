#!/usr/bin/env python3
"""
PostgreSQL Full-Text Search Query Tester

This script tests the synthetic training dataset queries against a PostgreSQL database
to validate tsquery syntax and measure result counts.
"""

import os
import json
import logging
import time
from typing import List, Dict
import psycopg2
from psycopg2 import sql, Error
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'query_test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class PostgreSQLQueryTester:
    def __init__(self):
        """Initialize database connection using environment variables."""
        self.connection = None
        self.cursor = None
        
        # Get connection parameters from environment
        self.db_config = {
            'host': os.getenv('POSTGRES_HOST'),
            'port': os.getenv('POSTGRES_PORT', 5432),
            'database': os.getenv('POSTGRES_DB'),
            'user': os.getenv('POSTGRES_USER'),
            'password': os.getenv('POSTGRES_PASSWORD')
        }
        
        # Validate required environment variables
        required_vars = ['POSTGRES_HOST', 'POSTGRES_DB', 'POSTGRES_USER', 'POSTGRES_PASSWORD']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    def connect(self):
        """Establish database connection."""
        try:
            self.connection = psycopg2.connect(**self.db_config)
            self.cursor = self.connection.cursor()
            logger.info(f"Successfully connected to PostgreSQL database: {self.db_config['database']}")
            
            # Test the connection and check if document table exists
            self.cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'document'
                );
            """)
            
            table_exists = self.cursor.fetchone()[0]
            if not table_exists:
                logger.warning("Table 'document' does not exist in the database")
            else:
                # Check if search_vector column exists
                self.cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.columns 
                        WHERE table_schema = 'public' 
                        AND table_name = 'document'
                        AND column_name = 'search_vector'
                    );
                """)
                
                search_vector_exists = self.cursor.fetchone()[0]
                if not search_vector_exists:
                    logger.warning("Column 'search_vector' does not exist in the 'document' table")
                else:
                    logger.info("Database schema validation successful")
                    
        except Error as e:
            logger.error(f"Error connecting to PostgreSQL database: {e}")
            raise
    
    def disconnect(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        logger.info("Database connection closed")
    
    def test_single_query(self, input_query, tsquery_expression):
        """
        Test a single tsquery expression and return results.
        
        Args:
            input_query (str): The original natural language query
            tsquery_expression (str): The PostgreSQL tsquery expression to test
            
        Returns:
            dict: Results containing success status, row count, or error message
        """
        try:
            # Set a reasonable timeout (30 seconds)
            self.cursor.execute("SET statement_timeout = '30s';")
            
            # Construct the SQL query with LIMIT for efficiency
            sql_query = """
                SELECT title FROM document 
                WHERE search_vector @@ plainto_tsquery('english', %s)
                LIMIT 1000
            """
            
            # Execute the query
            start_time = time.time()
            self.cursor.execute(sql_query, (tsquery_expression,))
            
            # Fetch results
            results = self.cursor.fetchall()
            execution_time = time.time() - start_time
            
            # For large result sets, get total count separately (with limit)
            count_sql = """
                SELECT COUNT(*) FROM (
                    SELECT 1 FROM document 
                    WHERE search_vector @@ plainto_tsquery('english', %s)
                    LIMIT 10000
                ) AS limited_count
            """
            
            self.cursor.execute(count_sql, (tsquery_expression,))
            limited_count = self.cursor.fetchone()[0]
            
            return {
                'success': True,
                'row_count': len(results),
                'total_count_limited': limited_count,
                'execution_time': execution_time,
                'error': None
            }
            
        except Error as e:
            # Reset timeout on error
            try:
                self.cursor.execute("SET statement_timeout = 0;")
            except:
                pass
            return {
                'success': False,
                'row_count': 0,
                'total_count_limited': 0,
                'execution_time': 0,
                'error': str(e)
            }
    
    def load_jsonl_dataset(self, jsonl_file: str) -> List[Dict]:
        """
        Load test dataset from JSONL file with prompt/completion format.
        
        Args:
            jsonl_file (str): Path to JSONL file
            
        Returns:
            list: List of test cases with extracted queries and completions
        """
        try:
            test_cases = []
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        data = json.loads(line)
                        
                        # Extract the natural language query from the prompt
                        if 'prompt' in data and 'completion' in data:
                            prompt = data['prompt']
                            completion = data['completion']
                            
                            # Extract the actual medical query from the prompt
                            # Looking for the pattern after "Convert this medical query to a PostgreSQL tsquery expression:\n"
                            query_start = prompt.find("Convert this medical query to a PostgreSQL tsquery expression:\n")
                            if query_start != -1:
                                query_start += len("Convert this medical query to a PostgreSQL tsquery expression:\n")
                                natural_query = prompt[query_start:].strip()
                            else:
                                # Fallback: use the whole prompt if pattern not found
                                natural_query = prompt.strip()
                            
                            test_cases.append({
                                'input': natural_query,
                                'expected_output': completion.strip(),
                                'raw_prompt': prompt,
                                'raw_completion': completion
                            })
                            
                        else:
                            logger.warning(f"Line {line_num}: Missing 'prompt' or 'completion' field")
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"Line {line_num}: Invalid JSON - {e}")
                        continue
            
            logger.info(f"Loaded {len(test_cases)} test cases from {jsonl_file}")
            return test_cases
            
        except FileNotFoundError:
            logger.error(f"JSONL file {jsonl_file} not found")
            raise
        except Exception as e:
            logger.error(f"Error loading JSONL file {jsonl_file}: {e}")
            raise
    
    def load_test_dataset(self, filename='postgres_training_data.json'):
        """
        Load the test dataset from JSON file.
        
        Args:
            filename (str): Path to the JSON dataset file
            
        Returns:
            list: List of test cases
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            logger.info(f"Loaded {len(dataset)} test queries from {filename}")
            return dataset
        except FileNotFoundError:
            logger.error(f"Dataset file {filename} not found")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON file {filename}: {e}")
            raise
    
    def run_all_tests(self, dataset_filename=None, max_queries=None, use_jsonl=False, compare_mode=False):
        """
        Run all test queries from the dataset and log results.
        
        Args:
            dataset_filename (str): Path to the dataset file
            max_queries (int): Limit number of queries to test (None = all)
            use_jsonl (bool): Whether to load JSONL format with prompt/completion
            compare_mode (bool): Whether to compare expected vs actual results
        """
        # Load test dataset
        if use_jsonl:
            if not dataset_filename:
                dataset_filename = './data/train.jsonl'
            dataset = self.load_jsonl_dataset(dataset_filename)
        else:
            if not dataset_filename:
                dataset_filename = 'postgres_training_data.json'
            dataset = self.load_test_dataset(dataset_filename)
        
        # Limit queries if specified
        if max_queries:
            dataset = dataset[:max_queries]
            logger.info(f"Limited to first {max_queries} queries for testing")
        
        # Initialize counters
        total_queries = len(dataset)
        successful_queries = 0
        failed_queries = 0
        total_results = 0
        exact_matches = 0
        
        logger.info(f"Starting test run with {total_queries} queries")
        if compare_mode:
            logger.info("Running in comparison mode - testing both expected and actual outputs")
        logger.info("-" * 80)
        
        # Test each query
        for i, test_case in enumerate(dataset, 1):
            if use_jsonl:
                input_query = test_case['input']
                expected_output = test_case['expected_output']
            else:
                input_query = test_case['input']
                expected_output = test_case['output']
            
            logger.info(f"Query {i}/{total_queries}:")
            logger.info(f"  Input: {input_query}")
            
            if compare_mode:
                logger.info(f"  Expected: {expected_output}")
            
            # Test the expected/provided query
            logger.info(f"  Testing: {expected_output}")
            result = self.test_single_query(input_query, expected_output)
            
            if result['success']:
                successful_queries += 1
                total_results += result['row_count']
                logger.info(f"  ✓ SUCCESS: Found {result['row_count']} rows (limited), "
                          f"total limited to 10k: {result['total_count_limited']}, "
                          f"time: {result['execution_time']:.3f}s")
                
                # In comparison mode, also test if we can generate the same query
                if compare_mode and hasattr(self, 'model_generator'):
                    try:
                        generated_output = self.model_generator.generate_query(input_query)
                        logger.info(f"  Generated: {generated_output}")
                        
                        # Check if generated matches expected
                        if generated_output.strip() == expected_output.strip():
                            exact_matches += 1
                            logger.info("  ✓ EXACT MATCH with expected output!")
                        else:
                            logger.info("  ⚠ Different from expected output")
                            
                        # Test the generated query too
                        gen_result = self.test_single_query(input_query, generated_output)
                        if gen_result['success']:
                            logger.info(f"  Generated query also works: {gen_result['row_count']} rows")
                        else:
                            logger.info(f"  Generated query failed: {gen_result['error']}")
                            
                    except Exception as e:
                        logger.info(f"  Model generation failed: {e}")
                        
            else:
                failed_queries += 1
                logger.error(f"  ✗ ERROR: {result['error']}")
            
            logger.info("-" * 80)
        
        # Log summary statistics
        logger.info("TEST RUN SUMMARY:")
        logger.info(f"  Total queries tested: {total_queries}")
        logger.info(f"  Successful queries: {successful_queries}")
        logger.info(f"  Failed queries: {failed_queries}")
        logger.info(f"  Success rate: {(successful_queries/total_queries)*100:.1f}%")
        logger.info(f"  Total results found: {total_results}")
        logger.info(f"  Average results per successful query: {total_results/successful_queries:.1f}" if successful_queries > 0 else "  Average results per successful query: 0")
        
        if compare_mode and exact_matches > 0:
            logger.info(f"  Exact matches (generated = expected): {exact_matches}/{successful_queries} ({(exact_matches/successful_queries)*100:.1f}%)")
    
    def set_model_generator(self, generator):
        """Set a model generator for comparison mode."""
        self.model_generator = generator
    
    def test_database_setup(self):
        """
        Test basic database functionality and full-text search setup.
        """
        logger.info("Testing database setup...")
        
        try:
            # Test basic query
            self.cursor.execute("SELECT version();")
            version = self.cursor.fetchone()[0]
            logger.info(f"PostgreSQL version: {version}")
            
            # Test document table
            self.cursor.execute("SELECT COUNT(*) FROM document;")
            doc_count = self.cursor.fetchone()[0]
            logger.info(f"Total documents in table: {doc_count}")
            
            # Test search_vector functionality with a simple query (LIMIT for efficiency)
            logger.info("Testing search_vector functionality...")
            self.cursor.execute("""
                SELECT COUNT(*) FROM (
                    SELECT 1 FROM document 
                    WHERE search_vector @@ plainto_tsquery('english', 'heart')
                    LIMIT 1000
                ) AS limited_results;
            """)
            test_results = self.cursor.fetchone()[0]
            logger.info(f"Test search for 'heart' found: {test_results} documents (limited to 1000)")
            
            # Test tsquery parsing
            self.cursor.execute("SELECT plainto_tsquery('english', 'heart & failure');")
            parsed_query = self.cursor.fetchone()[0]
            logger.info(f"Sample tsquery parsing result: {parsed_query}")
            
            # Test if indexes exist on search_vector
            self.cursor.execute("""
                SELECT indexname FROM pg_indexes 
                WHERE tablename = 'document' 
                AND indexdef LIKE '%search_vector%';
            """)
            indexes = self.cursor.fetchall()
            if indexes:
                logger.info(f"Found {len(indexes)} index(es) on search_vector column")
            else:
                logger.warning("No indexes found on search_vector column - queries may be slow")
                
        except Error as e:
            logger.error(f"Database setup test failed: {e}")
            raise

def main():
    """Main function to run the query tester."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test PostgreSQL tsquery expressions')
    parser.add_argument('--data', type=str, help='Path to dataset file (JSON or JSONL)')
    parser.add_argument('--jsonl', action='store_true', help='Use JSONL format with prompt/completion structure')
    parser.add_argument('--quick', action='store_true', help='Run only first 5 queries for quick testing')
    parser.add_argument('--max-queries', type=int, help='Maximum number of queries to test')
    parser.add_argument('--skip-setup', action='store_true', help='Skip database setup tests')
    parser.add_argument('--compare', action='store_true', help='Compare mode: test expected vs generated outputs')
    parser.add_argument('--model-path', type=str, help='Path to fused model for comparison mode')
    
    args = parser.parse_args()
    
    tester = PostgreSQLQueryTester()
    
    try:
        # Connect to database
        tester.connect()
        
        # Test database setup (unless skipped)
        if not args.skip_setup:
            tester.test_database_setup()
        
        # Set up model generator for comparison mode
        if args.compare:
            try:
                if args.model_path:
                    # Use custom model path
                    from simple_fused_inference import SimpleFusedInference
                    generator = SimpleFusedInference(args.model_path)
                    tester.set_model_generator(generator)
                    logger.info(f"Loaded model for comparison from {args.model_path}")
                else:
                    # Try default fused model path
                    try:
                        from simple_fused_inference import SimpleFusedInference
                        generator = SimpleFusedInference("./models/fused_model")
                        tester.set_model_generator(generator)
                        logger.info("Loaded default fused model for comparison")
                    except:
                        logger.warning("Could not load fused model for comparison mode")
                        logger.info("Continuing without comparison - will only test provided queries")
                        args.compare = False
            except ImportError:
                logger.warning("simple_fused_inference.py not available for comparison mode")
                args.compare = False
        
        # Determine dataset file and format
        dataset_file = args.data
        use_jsonl = args.jsonl
        
        # Auto-detect format if not specified
        if dataset_file and not use_jsonl:
            if dataset_file.endswith('.jsonl'):
                use_jsonl = True
                logger.info("Auto-detected JSONL format from file extension")
        
        # Determine max queries
        max_queries = None
        if args.quick:
            max_queries = 5
        elif args.max_queries:
            max_queries = args.max_queries
        
        # Run test queries
        tester.run_all_tests(
            dataset_filename=dataset_file,
            max_queries=max_queries,
            use_jsonl=use_jsonl,
            compare_mode=args.compare
        )
        
    except Exception as e:
        logger.error(f"Test run failed: {e}")
        raise
    finally:
        # Always disconnect
        tester.disconnect()
    
    # Usage examples
    if args.quick or max_queries == 5:
        print("\n💡 Usage Examples:")
        print("# Test with your training data (JSONL format):")
        print("python postgres_query_tester.py --data ./data/train.jsonl --jsonl")
        print("\n# Test with comparison mode (expected vs generated):")
        print("python postgres_query_tester.py --data ./data/valid.jsonl --jsonl --compare")
        print("\n# Test specific model:")
        print("python postgres_query_tester.py --data ./data/test.jsonl --jsonl --compare --model-path ./models/fused_model")
        print("\n# Quick test with original JSON format:")
        print("python postgres_query_tester.py --data postgres_training_data.json --quick")

if __name__ == "__main__":
    main()
