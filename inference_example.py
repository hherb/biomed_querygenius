#!/usr/bin/env python3
"""
Inference example for the fine-tuned Phi-3-mini model
Generates PostgreSQL tsquery expressions from natural language medical queries
"""

import json
import mlx.core as mx
from mlx_lm import load, generate
from pathlib import Path

class PostgresQueryGenerator:
    """Inference class for generating PostgreSQL queries."""
    
    def __init__(self, model_path: str = "microsoft/Phi-3-mini-4k-instruct", adapter_path: str = "./lora_adapters"):
        """
        Initialize the query generator.
        
        Args:
            model_path: Path to the base model
            adapter_path: Path to the LoRA adapters directory
        """
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the fine-tuned model and tokenizer."""
        print(f"Loading model from {self.model_path}")
        
        if Path(self.adapter_path).exists():
            print(f"Loading LoRA adapters from {self.adapter_path}")
            # Load model with LoRA adapters
            self.model, self.tokenizer = load(
                self.model_path,
                adapter_path=self.adapter_path
            )
        else:
            print("No LoRA adapters found, loading base model only")
            # Load base model only
            self.model, self.tokenizer = load(self.model_path)
        
        print("Model loaded successfully!")
    
    def _create_prompt(self, query: str) -> str:
        """Create a structured prompt for the model."""
        return f"""<|system|>
You are a medical database search expert. Convert natural language medical queries into PostgreSQL full-text search expressions using tsquery syntax.

Rules:
- Use & for AND operations
- Use | for OR operations  
- Use quotes for exact phrases
- Favor broader terms to avoid false negatives
- Don't include subset terms (e.g., use 'thoracotomy' not 'emergency thoracotomy')
- Let PostgreSQL stemming handle plurals

<|user|>
Convert this medical query to a PostgreSQL tsquery expression:
{query}

<|assistant|>
"""
    
    def generate_query(self, input_text: str, max_tokens: int = 100, temperature: float = 0.1) -> str:
        """
        Generate a PostgreSQL tsquery expression from natural language.
        
        Args:
            input_text: Natural language medical query
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            
        Returns:
            Generated tsquery expression
        """
        prompt = self._create_prompt(input_text)
        
        # Generate response (MLX-LM returns only the generated part)
        try:
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens
            )
        except Exception as e:
            print(f"Generation failed: {e}")
            return f"Error: {e}"
        
        # MLX-LM returns just the generated part, no need to remove prompt
        generated_text = response.strip()
        
        # Clean up the response (remove special tokens and extra text)
        # Remove <|end|> and other special tokens
        generated_text = generated_text.replace('<|end|>', '').replace('<|endoftext|>', '')
        
        # If it starts with "tsquery", remove that prefix
        if generated_text.startswith('tsquery '):
            generated_text = generated_text[8:]  # Remove "tsquery "
        
        # Take only the first line if there are multiple lines
        lines = generated_text.split('\n')
        if lines:
            result = lines[0].strip()
        else:
            result = generated_text.strip()
        
        return result
    
    def batch_generate(self, queries: list, **kwargs) -> list:
        """Generate tsquery expressions for multiple queries."""
        results = []
        for query in queries:
            result = self.generate_query(query, **kwargs)
            results.append(result)
        return results

def main():
    """Example usage of the PostgreSQL query generator."""
    
    # Initialize generator
    generator = PostgresQueryGenerator()
    
    # Test queries
    test_queries = [
        "What is the sensitivity of troponin for myocardial infarction?",
        "What are the contraindications for lumbar puncture?",
        "What is the treatment for septic shock?",
        "What are the risk factors for pulmonary embolism?",
        "What is the Glasgow Coma Scale scoring system?",
        "What are the signs of compartment syndrome?",
        "What is the accuracy of ultrasound for gallbladder stones?",
        "What are the criteria for massive transfusion protocol?",
    ]
    
    print("Testing PostgreSQL Query Generation")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}:")
        print(f"Input:  {query}")
        
        # Generate tsquery
        generated_query = generator.generate_query(query)
        print(f"Output: {generated_query}")
        
        # Validate basic syntax (simple check)
        if any(char in generated_query for char in ['&', '|']):
            print("✓ Contains logical operators")
        else:
            print("⚠ No logical operators detected")
        
        print("-" * 60)
    
    # Interactive mode
    print("\nInteractive Mode (type 'quit' to exit):")
    while True:
        user_input = input("\nEnter medical query: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if user_input:
            result = generator.generate_query(user_input)
            print(f"Generated tsquery: {result}")

if __name__ == "__main__":
    main()
