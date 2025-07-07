#!/usr/bin/env python3
"""
Abstract Summarizer

This script connects to PostgreSQL database and retrieves documents from 2025
to create summaries for fine-tuning. Only processes abstracts between 100-500 characters.
"""

import os
import json
import logging
from typing import List, Dict, Optional
import psycopg2
from psycopg2 import sql, Error
from dotenv import load_dotenv
from datetime import datetime, date
import ollama

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'abstract_summarizer_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class AbstractSummarizer:
    """
    Retrieves abstracts from PostgreSQL database and generates summaries for fine-tuning.
    """
    
    def __init__(self, model_name: str = "mistral-small3.2"):
        """Initialize database connection and AI model using environment variables."""
        self.connection = None
        self.cursor = None
        self.model_name = model_name
        
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
        
        # Test Ollama connection
        self._test_ollama_connection()
    
    def _test_ollama_connection(self):
        """Test connection to Ollama and verify model availability."""
        try:
            # Test if Ollama is running
            models = ollama.list()
            logger.info(f"Ollama connection successful. Available models: {len(models['models'])}")
            
            # Check if our model is available - handle different response formats
            model_names = []
            for model in models['models']:
                # Try different possible keys for model name
                if 'name' in model:
                    model_names.append(model['name'])
                elif 'model' in model:
                    model_names.append(model['model'])
                elif isinstance(model, str):
                    model_names.append(model)
                else:
                    # Try to get the first string value
                    for key, value in model.items():
                        if isinstance(value, str):
                            model_names.append(value)
                            break
            
            logger.info(f"Found model names: {model_names[:5]}...")  # Show first 5 for brevity
            
            if self.model_name not in model_names:
                logger.warning(f"Model '{self.model_name}' not found in available models")
                logger.info(f"Attempting to pull model '{self.model_name}'...")
                try:
                    ollama.pull(self.model_name)
                    logger.info(f"Successfully pulled model '{self.model_name}'")
                except Exception as pull_error:
                    logger.warning(f"Could not pull model: {pull_error}")
                    logger.info(f"Will attempt to use model '{self.model_name}' anyway")
            else:
                logger.info(f"Model '{self.model_name}' is available")
                
        except Exception as e:
            logger.warning(f"Error checking Ollama models: {e}")
            logger.info(f"Will attempt to use model '{self.model_name}' anyway")
    
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
                raise ValueError("Table 'document' does not exist in the database")
            
            logger.info("Database connection and table validation successful")
                    
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
    
    def get_abstracts_for_summary(self, limit: int = 100, from_date: Optional[date] = None, to_date: Optional[date] = None) -> List[Dict[str, str]]:
        """
        Retrieve documents within a specified date range with abstracts between 100-500 characters.

        Args:
            limit (int): Maximum number of documents to retrieve (default: 100)
            from_date (Optional[date]): Start date for the range (default: None, uses 2025-01-01)
            to_date (Optional[date]): End date for the range (default: None, uses current date)

        Returns:
            list: List of dictionaries containing title, abstract, and publication_date
        """
        try:
            # Set default date range if not provided
            if from_date is None:
                from_date = date(2025, 1, 1)
            if to_date is None:
                to_date = date.today()

            # Query to get documents within the specified date range with abstracts in the right length range
            query = """
                SELECT title, abstract, publication_date
                FROM document
                WHERE publication_date >= %s
                AND publication_date <= %s
                AND LENGTH(abstract) BETWEEN 100 AND 500
                AND abstract IS NOT NULL
                AND abstract != ''
                ORDER BY publication_date DESC
                LIMIT %s
            """

            logger.info(f"Executing query to retrieve {limit} documents from {from_date} to {to_date} with abstracts 100-500 chars...")
            self.cursor.execute(query, (from_date, to_date, limit))
            
            # Fetch results
            results = self.cursor.fetchall()
            
            # Convert to list of dictionaries
            documents = []
            for row in results:
                title, abstract, pub_date = row
                documents.append({
                    'title': title if title else '',
                    'abstract': abstract if abstract else '',
                    'publication_date': pub_date.strftime('%Y-%m-%d') if pub_date else ''
                })
            
            logger.info(f"Successfully retrieved {len(documents)} documents from {from_date} to {to_date} for summarization")
            return documents
            
        except Error as e:
            logger.error(f"Error retrieving documents: {e}")
            raise
    
    def generate_summary(self, title: str, abstract: str, output_format: str = "phi3") -> Optional[Dict[str, str]]:
        """
        Generate a summary for the given title and abstract.

        Args:
            title (str): Document title
            abstract (str): Document abstract
            output_format (str): Output format - "phi3" for MLX format or "prompt_completion" for legacy format

        Returns:
            dict: Summary in specified format or None if failed
        """
        try:
            # Create the text to summarize (title + abstract)
            text_to_summarize = f"{title}\n\n{abstract}"
            
            # Create prompt for the AI model
            prompt = f"Summarize the following text in 3 sentences or less, capturing the essential message. Be direct and concise without filler phrases like 'This study shows' or 'The text discusses'. Focus on the key findings and facts:\n\n<text>\n{text_to_summarize}\n</text>"
            
            # Call Ollama API
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            )
            
            # Extract the response content
            summary = response['message']['content'].strip()

            if output_format == "phi3":
                # Format for phi3 MLX training
                # Extract the original text from the prompt for the user message
                text_to_summarize = f"{title}\n\n{abstract}"
                phi3_text = f"<|user|>\nSummarize the following text in 3 sentences or less, capturing the essential message: <text>{text_to_summarize}</text> <|end|>\n<|assistant|> \n{summary} <|end|>"
                return {
                    'text': phi3_text
                }
            else:
                # Legacy prompt/completion format
                return {
                    'prompt': prompt,
                    'completion': summary
                }
                
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return None
    
    def get_document_stats(self) -> Dict[str, int]:
        """
        Get statistics about documents suitable for summarization.
        
        Returns:
            dict: Statistics including total count and suitable documents count
        """
        try:
            stats = {}
            
            # Total document count
            self.cursor.execute("SELECT COUNT(*) FROM document;")
            stats['total_documents'] = self.cursor.fetchone()[0]
            
            # 2025 documents suitable for summarization
            self.cursor.execute("""
                SELECT COUNT(*) FROM document 
                WHERE EXTRACT(YEAR FROM publication_date) = 2025
                AND publication_date < CURRENT_DATE
                AND LENGTH(abstract) BETWEEN 100 AND 500
                AND abstract IS NOT NULL
                AND abstract != '';
            """)
            stats['suitable_documents'] = self.cursor.fetchone()[0]
            
            # Total 2025 documents
            self.cursor.execute("""
                SELECT COUNT(*) FROM document 
                WHERE EXTRACT(YEAR FROM publication_date) = 2025
                AND publication_date < CURRENT_DATE;
            """)
            stats['total_2025_docs'] = self.cursor.fetchone()[0]
            
            # Date range
            self.cursor.execute("""
                SELECT MIN(publication_date), MAX(publication_date) 
                FROM document 
                WHERE publication_date IS NOT NULL;
            """)
            min_date, max_date = self.cursor.fetchone()
            stats['date_range'] = {
                'min': min_date.strftime('%Y-%m-%d') if min_date else None,
                'max': max_date.strftime('%Y-%m-%d') if max_date else None
            }
            
            return stats
            
        except Error as e:
            logger.error(f"Error getting document statistics: {e}")
            raise
    
    def generate_summary_jsonl(self, documents: List[Dict[str, str]], output_filename: Optional[str] = None, output_format: str = "phi3") -> str:
        """
        Generate summaries for all documents and save to JSONL file.

        Args:
            documents (list): List of document dictionaries
            output_filename (str, optional): Output JSONL filename (default: auto-generated)
            output_format (str): Output format - "phi3" for MLX format or "prompt_completion" for legacy format

        Returns:
            str: Path to the generated JSONL file
        """
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"summaries_{timestamp}.jsonl"
        
        total_summaries = 0
        processed_docs = 0
        
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                for i, doc in enumerate(documents, 1):
                    logger.info(f"Processing document {i}/{len(documents)}: {doc['title'][:60]}...")
                    
                    # Generate summary for this document
                    summary_data = self.generate_summary(doc['title'], doc['abstract'], output_format)
                    
                    if summary_data:
                        # Write summary to JSONL file
                        json_line = json.dumps(summary_data, ensure_ascii=False)
                        f.write(json_line + '\n')
                        total_summaries += 1
                        processed_docs += 1
                        logger.info(f"  Generated summary")
                    else:
                        logger.warning(f"  Failed to generate summary for document {i}")
                    
                    # Progress update every 10 documents
                    if i % 10 == 0:
                        logger.info(f"Progress: {i}/{len(documents)} documents processed, {total_summaries} summaries generated")
            
            logger.info(f"Summary generation complete!")
            logger.info(f"  Processed documents: {processed_docs}/{len(documents)}")
            logger.info(f"  Total summaries: {total_summaries}")
            logger.info(f"  Output file: {output_filename}")
            
            return output_filename
            
        except Exception as e:
            logger.error(f"Error generating summary JSONL file: {e}")
            raise

def main():
    """Main function to retrieve abstracts and generate summaries."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate summaries from 2025 abstracts for fine-tuning')
    parser.add_argument('--limit', type=int, default=100, help='Number of documents to retrieve (default: 100)')
    parser.add_argument('--output-jsonl', type=str, help='Output JSONL file for summaries')
    parser.add_argument('--stats-only', action='store_true', help='Show only database statistics')
    parser.add_argument('--model', type=str, default='mistral-small3.2', help='Ollama model to use (default: mistral-small3.2)')
    parser.add_argument('--from-date', type=str, help='Start date for document retrieval (YYYY-MM-DD format, default: 2025-01-01)')
    parser.add_argument('--to-date', type=str, help='End date for document retrieval (YYYY-MM-DD format, default: today)')
    parser.add_argument('--format', type=str, choices=['phi3', 'prompt_completion'], default='phi3',
                       help='Output format: phi3 for MLX training or prompt_completion for legacy format (default: phi3)')
    
    args = parser.parse_args()

    # Parse date arguments
    from_date = None
    to_date = None

    if args.from_date:
        try:
            from_date = datetime.strptime(args.from_date, '%Y-%m-%d').date()
        except ValueError:
            logger.error(f"Invalid from-date format: {args.from_date}. Use YYYY-MM-DD format.")
            return

    if args.to_date:
        try:
            to_date = datetime.strptime(args.to_date, '%Y-%m-%d').date()
        except ValueError:
            logger.error(f"Invalid to-date format: {args.to_date}. Use YYYY-MM-DD format.")
            return

    # Validate date range
    if from_date and to_date and from_date > to_date:
        logger.error("from-date cannot be later than to-date")
        return

    # Log the date range being used
    if from_date or to_date:
        effective_from = from_date if from_date else date(2025, 1, 1)
        effective_to = to_date if to_date else date.today()
        logger.info(f"Using date range: {effective_from} to {effective_to}")

    summarizer = AbstractSummarizer(model_name=args.model)
    
    try:
        # Connect to database
        summarizer.connect()
        
        # Get and display statistics
        stats = summarizer.get_document_stats()
        logger.info("Database Statistics:")
        logger.info(f"  Total documents: {stats['total_documents']:,}")
        logger.info(f"  Total 2025 documents (valid dates): {stats['total_2025_docs']:,}")
        logger.info(f"  Suitable for summarization (100-500 chars): {stats['suitable_documents']:,}")
        logger.info(f"  Date range: {stats['date_range']['min']} to {stats['date_range']['max']}")
        
        if args.stats_only:
            return
        
        if stats['suitable_documents'] == 0:
            logger.warning("No suitable documents found for summarization!")
            return
        
        # Retrieve abstracts
        documents = summarizer.get_abstracts_for_summary(limit=args.limit, from_date=from_date, to_date=to_date)
        
        if not documents:
            logger.warning("No documents retrieved!")
            return
        
        # Display sample information
        logger.info(f"\nSample of retrieved documents:")
        for i, doc in enumerate(documents[:3], 1):
            logger.info(f"  {i}. {doc['publication_date']} - {doc['title'][:60]}... (abstract: {len(doc['abstract'])} chars)")
        
        # Generate summaries
        logger.info(f"\nStarting summary generation for {len(documents)} documents...")
        logger.info(f"Using model: {args.model}")
        
        jsonl_file = summarizer.generate_summary_jsonl(documents, args.output_jsonl, args.format)
        logger.info(f"\nSummaries saved to: {jsonl_file}")
        logger.info(f"Format: {args.format}")
        if args.format == "phi3":
            logger.info("Ready for phi3 MLX training!")
        else:
            logger.info("Ready for fine-tuning (legacy format)!")
        
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        raise
    finally:
        # Always disconnect
        summarizer.disconnect()

if __name__ == "__main__":
    main()
