#!/usr/bin/env python3
"""
Debug script for testing raw response saving functionality.
"""

import os
import tempfile
import time
from pathlib import Path
import json
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src import batch
from tests.utils.pdf_utils import create_pdf


class ExampleModel(BaseModel):
    """Example structured response model."""
    summary: str
    key_points: list[str]





def test_raw_responses_file_processing():
    """Test raw response saving with file processing."""
    print("\n🔍 Testing raw response saving - File processing mode")
    
    raw_dir = Path("./raw_responses/file_processing")
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Saving raw responses to: {raw_dir}")
    
    # Create a mock PDF file for testing using the pdf_utils
    content = "This is a test document about artificial intelligence. AI has revolutionized many industries."
    pdf_data = create_pdf([content])
    
    mock_file_path = Path("./test_document.pdf")
    mock_file_path.write_bytes(pdf_data)
    
    try:
        # Create batch with file processing and raw response saving
        job = batch(
            files=[str(mock_file_path)],
            prompt="Summarize this document",
            model="claude-3-5-sonnet-20241022",
            raw_results_dir=str(raw_dir),
            enable_citations=True,
            verbose=True
        )
            
        print(f"📊 Batch ID: {job._batch_id}")
        print(f"🔄 Waiting for batch to complete...")
        
        # Wait for completion
        while not job.is_complete():
            time.sleep(5)
            print("⏳ Still processing...")
        
        print("✅ Batch completed!")
        
        # Get results
        results = job.results()
        print(f"📝 Results: {len(results)} file processing responses")
        
        # Check raw response files
        raw_files = list(raw_dir.glob("*.json"))
        print(f"📄 Raw response files: {len(raw_files)}")
        
        for i, raw_file in enumerate(sorted(raw_files)):
            if raw_file.name.endswith('.json'):
                print(f"   {i+1}. {raw_file.name}")
        
        # Show file processing results
        print("\n🎯 File Processing Results:")
        for i, result_entry in enumerate(results):
            result = result_entry["result"]
            print(f"   {i+1}. {result[:100]}...")

        job.stats(print_stats=True)
            
    except Exception as e:
        print(f"⚠️  File processing test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up mock file
        if mock_file_path.exists():
            mock_file_path.unlink()


def main():
    """Run all debug tests."""
    print("🚀 Starting raw response saving debug tests")
    
    # Check if API key is available
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY not found in environment")
        print("   Please set your API key in .env file")
        return
    
    try:
        # Test different modes
        # test_raw_responses_text_mode()
        # test_raw_responses_structured_mode()
        test_raw_responses_file_processing()
        
        print("\n✅ All raw response saving tests completed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()