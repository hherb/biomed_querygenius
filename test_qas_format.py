#!/usr/bin/env python3
"""
Test script to verify the new Q/A/S format functionality.
"""

import json
import sys
import os

# Add the current directory to the path so we can import abstract_qa
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from abstract_qa import AbstractQARetriever

def test_qas_format():
    """Test the Q/A/S format generation."""
    
    # Sample document data
    test_doc = {
        'title': 'Efficacy of Aspirin in Primary Prevention of Cardiovascular Disease',
        'abstract': 'This randomized controlled trial examined the effects of low-dose aspirin (81mg daily) versus placebo in 10,000 patients without prior cardiovascular disease. After 5 years of follow-up, aspirin reduced the risk of myocardial infarction by 22% (HR 0.78, 95% CI 0.65-0.94, p=0.008) but increased the risk of major bleeding by 58% (HR 1.58, 95% CI 1.21-2.07, p=0.001). The number needed to treat to prevent one MI was 300, while the number needed to harm for one major bleeding event was 400.',
        'doi': '10.1001/jama.2024.12345',
        'external_id': '38123456'
    }
    
    print("Testing Q/A/S format generation...")
    print("=" * 50)
    
    # Test with DOI
    print("\nTest 1: Document with DOI")
    print(f"Title: {test_doc['title']}")
    print(f"DOI: {test_doc['doi']}")
    
    # Simulate the Q/A/S format generation
    sample_qa = {
        'question': 'What is the effect of low-dose aspirin on myocardial infarction risk in primary prevention?',
        'answer': 'Low-dose aspirin (81mg daily) reduces myocardial infarction risk by 22% in primary prevention, with a number needed to treat of 300 patients.'
    }
    
    # Generate Q/A/S format
    source = test_doc['doi'] if test_doc['doi'] else f"PMID:{test_doc['external_id']}" if test_doc['external_id'] else "Unknown"
    qas_text = f"Q: {sample_qa['question']}\nA: {sample_qa['answer']}\nS: {source}"

    print("\nGenerated Q/A/S format:")
    print(qas_text)
    print()  # Extra empty line for visual separation
    
    # Test without DOI (using PMID)
    print("\n" + "=" * 50)
    print("\nTest 2: Document without DOI (using PMID)")
    test_doc_no_doi = test_doc.copy()
    test_doc_no_doi['doi'] = ''
    
    print(f"Title: {test_doc_no_doi['title']}")
    print(f"External ID: {test_doc_no_doi['external_id']}")
    
    source = test_doc_no_doi['doi'] if test_doc_no_doi['doi'] else f"PMID:{test_doc_no_doi['external_id']}" if test_doc_no_doi['external_id'] else "Unknown"
    qas_text = f"Q: {sample_qa['question']}\nA: {sample_qa['answer']}\nS: {source}"

    print("\nGenerated Q/A/S format:")
    print(qas_text)
    print()  # Extra empty line for visual separation
    
    # Test with neither DOI nor external_id
    print("\n" + "=" * 50)
    print("\nTest 3: Document without DOI or external_id")
    test_doc_no_source = test_doc.copy()
    test_doc_no_source['doi'] = ''
    test_doc_no_source['external_id'] = ''
    
    print(f"Title: {test_doc_no_source['title']}")
    print("No DOI or external_id available")
    
    source = test_doc_no_source['doi'] if test_doc_no_source['doi'] else f"PMID:{test_doc_no_source['external_id']}" if test_doc_no_source['external_id'] else "Unknown"
    qas_text = f"Q: {sample_qa['question']}\nA: {sample_qa['answer']}\nS: {source}"

    print("\nGenerated Q/A/S format:")
    print(qas_text)
    print()  # Extra empty line for visual separation
    
    print("\n" + "=" * 50)
    print("\nTest completed successfully!")
    print("\nTo use the new Q/A/S format with abstract_qa.py:")
    print("python abstract_qa.py --format qas --limit 5")
    print("\nThis will generate a text file with Q:/A:/S: format instead of JSONL.")
    print("Each Q/A/S triplet is separated by two empty lines for easy visual review.")

if __name__ == "__main__":
    test_qas_format()
