#!/usr/bin/env python3
"""
Format Comparison Example

This script demonstrates the difference between phi3 MLX format and 
legacy prompt/completion format for training data generation.
"""

import json

def show_qa_formats():
    """Show Q&A format examples."""
    print("Q&A Format Comparison")
    print("=" * 50)
    
    # Sample Q&A data
    question = "What are the side effects of aspirin?"
    answer = "Common side effects include stomach irritation, bleeding risk, and allergic reactions."
    
    # Phi3 MLX format
    phi3_format = {
        "text": f"<|user|>\n{question} <|end|>\n<|assistant|> \n{answer} <|end|>"
    }
    
    # Legacy format
    legacy_format = {
        "prompt": question,
        "completion": answer
    }
    
    print("\n📱 Phi3 MLX Format (for MLX training):")
    print(json.dumps(phi3_format, indent=2))
    
    print("\n📄 Legacy Format (prompt/completion):")
    print(json.dumps(legacy_format, indent=2))

def show_summary_formats():
    """Show summary format examples."""
    print("\n\nSummary Format Comparison")
    print("=" * 50)
    
    # Sample data
    title = "Efficacy of Aspirin in Cardiovascular Disease Prevention"
    abstract = "This study examined the effectiveness of low-dose aspirin in preventing cardiovascular events in high-risk patients over a 5-year period."
    summary = "Low-dose aspirin significantly reduces cardiovascular events in high-risk patients over 5 years."
    
    # Phi3 MLX format
    text_to_summarize = f"{title}\n\n{abstract}"
    phi3_format = {
        "text": f"<|user|>\nSummarize the following text in 3 sentences or less, capturing the essential message: <text>{text_to_summarize}</text> <|end|>\n<|assistant|> \n{summary} <|end|>"
    }
    
    # Legacy format
    prompt = f"Summarize the following text in 3 sentences or less, capturing the essential message. Be direct and concise without filler phrases like 'This study shows' or 'The text discusses'. Focus on the key findings and facts:\n\n<text>\n{text_to_summarize}\n</text>"
    legacy_format = {
        "prompt": prompt,
        "completion": summary
    }
    
    print("\n📱 Phi3 MLX Format (for MLX training):")
    print(json.dumps(phi3_format, indent=2))
    
    print("\n📄 Legacy Format (prompt/completion):")
    print(json.dumps(legacy_format, indent=2))

def show_usage_examples():
    """Show command line usage examples."""
    print("\n\nCommand Line Usage Examples")
    print("=" * 50)
    
    print("\n🔧 Generate Q&A pairs:")
    print("# Phi3 format (default)")
    print("python abstract_qa.py --limit 50 --output-jsonl qa_phi3.jsonl")
    print("\n# Legacy format")
    print("python abstract_qa.py --format prompt_completion --limit 50 --output-jsonl qa_legacy.jsonl")
    
    print("\n📝 Generate summaries:")
    print("# Phi3 format (default)")
    print("python abstract_summarizer.py --limit 50 --output-jsonl summaries_phi3.jsonl")
    print("\n# Legacy format")
    print("python abstract_summarizer.py --format prompt_completion --limit 50 --output-jsonl summaries_legacy.jsonl")
    
    print("\n🔄 Convert existing data:")
    print("# Convert legacy format to phi3 format")
    print("python scripts/convert2phi3mlxformat.py legacy_data.jsonl phi3_data.jsonl")
    print("\n# Preview conversion")
    print("python scripts/convert2phi3mlxformat.py legacy_data.jsonl --preview")

def main():
    """Main function."""
    print("🔄 Training Data Format Comparison")
    print("This example shows the difference between phi3 MLX format and legacy format")
    print()
    
    show_qa_formats()
    show_summary_formats()
    show_usage_examples()
    
    print("\n\n💡 Key Differences:")
    print("- Phi3 format uses a single 'text' field with chat template markers")
    print("- Legacy format uses separate 'prompt' and 'completion' fields")
    print("- Phi3 format is required for MLX training with phi3 models")
    print("- Legacy format can be converted to phi3 format using the conversion script")
    
    print("\n✅ Both formats are now supported by the data generation scripts!")

if __name__ == "__main__":
    main()
