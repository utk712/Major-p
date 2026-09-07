import sys
import os
sys.path.append(os.getcwd())

from app import gemini_support

# Test cases for unrestricted AI
test_questions = [
    "What is the capital of France?",
    "How does photosynthesis work?",
    "Tell me a joke.",
    "What are the benefits of exercise?",
    "Explain quantum physics in simple terms.",
    "How to cook spaghetti?",
    "What is machine learning?",
    "Tell me about the history of the internet."
]

personal_details = {"name": "TestUser"}

print("Testing InsureBot's unrestricted AI responses:\n")

for question in test_questions:
    print(f"Question: {question}")
    response = gemini_support(question, personal_details)
    print(f"Response: {response}\n")
    print("-" * 50)
