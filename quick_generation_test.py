#!/usr/bin/env python3
"""
Quick test script to check if the trained model generates good outputs
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append('.')

try:
    from phi3_mlx_training import Phi3QueryGenerator
    
    def test_generation():
        """Test the trained model generation."""
        print("🧪 Testing Fine-tuned Phi-3-mini Model")
        print("=" * 60)
        
        # Initialize generator
        try:
            generator = Phi3QueryGenerator(
                model_path="microsoft/Phi-3-mini-4k-instruct",
                adapter_path="./lora_adapters"
            )
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return
        
        # Test queries from your original dataset
        test_queries = [
            "What is the sensitivity of troponin for myocardial infarction?",
            "What are the contraindications for lumbar puncture?", 
            "What is the treatment for septic shock?",
            "What are the risk factors for pulmonary embolism?",
            "What is the Glasgow Coma Scale scoring system?",
            "What are the signs of compartment syndrome?",
            "What is the accuracy of ultrasound for gallbladder stones?",
            "What are the side effects of amiodarone?",
        ]
        
        # Expected patterns to look for
        success_count = 0
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔬 Test {i}/{len(test_queries)}:")
            print(f"Input:  {query}")
            
            try:
                # Generate with the model
                result = generator.generate_query(query, max_tokens=50, temperature=0.1)
                print(f"Output: {result}")
                
                # Basic quality checks
                checks = []
                
                # Check 1: Contains logical operators
                if any(op in result for op in ['&', '|']):
                    checks.append("✅ Has logical operators")
                else:
                    checks.append("❌ Missing logical operators")
                
                # Check 2: Not too long (should be concise)
                if len(result.split()) <= 15:
                    checks.append("✅ Reasonable length")
                else:
                    checks.append("⚠️ Might be too verbose")
                
                # Check 3: Contains medical terms from input
                input_words = set(query.lower().replace('?', '').split())
                output_words = set(result.lower().split())
                if input_words & output_words:
                    checks.append("✅ Contains relevant terms")
                else:
                    checks.append("⚠️ May not contain input terms")
                
                # Check 4: Looks like tsquery syntax
                if not any(char in result for char in ['<', '>', 'SELECT', 'FROM']):
                    checks.append("✅ Looks like tsquery (not SQL)")
                else:
                    checks.append("❌ Looks like SQL, not tsquery")
                
                # Print checks
                for check in checks:
                    print(f"  {check}")
                
                # Count as success if has operators and reasonable length
                if '&' in result or '|' in result:
                    success_count += 1
                    
            except Exception as e:
                print(f"❌ Generation failed: {e}")
            
            print("-" * 60)
        
        # Summary
        print(f"\n📊 Results Summary:")
        print(f"Successful generations: {success_count}/{len(test_queries)} ({success_count/len(test_queries)*100:.1f}%)")
        
        if success_count >= len(test_queries) * 0.8:
            print("🎉 Model appears to be working well!")
        elif success_count >= len(test_queries) * 0.5:
            print("🤔 Model shows promise but may need more training")
        else:
            print("😞 Model may need significant improvement")
            
        print(f"\n💡 Next steps:")
        if success_count >= len(test_queries) * 0.7:
            print("- Test with your PostgreSQL database using the query tester")
            print("- Try more diverse medical queries")
            print("- Consider the model ready for use")
        else:
            print("- Consider training for more iterations (1500-2000)")
            print("- Add more diverse examples to training data")
            print("- Check if validation loss can be reduced")
    
    if __name__ == "__main__":
        test_generation()
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the correct directory")
    print("Try: python quick_test.py")
