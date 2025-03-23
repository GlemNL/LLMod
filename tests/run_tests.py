
#!/usr/bin/env python3
"""
Script to run the moderation tests with the local Ollama LLM
"""
import asyncio
import pytest
import sys
import os
from pathlib import Path

def main():
    """Main function to run the tests"""
    # Ensure the current directory is in the path
    sys.path.insert(0, os.getcwd())
    
    # Add project root to the path if running from tests directory
    if Path.cwd().name == "tests/":
        sys.path.insert(0, str(Path.cwd().parent))
    
    # Run the tests
    pytest_args = [
        "tests/moderation/test_moderation.py",       # The test file
        "-v",                       # Verbose output
        "--asyncio-mode=auto",      # Auto-detect asyncio mode
    ]
    
    exit_code = pytest.main(pytest_args)
    return exit_code

if __name__ == "__main__":
    sys.exit(main())