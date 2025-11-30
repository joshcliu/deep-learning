"""
Examples of how to use the dataset loaders.

Run this script to test that datasets load correctly.
"""

from src.data import MMLUDataset, TriviaQADataset, GSM8KDataset


def example_mmlu():
    """Example: Loading and using MMLU dataset."""
    print("\n" + "="*60)
    print("MMLU DATASET EXAMPLE")
    print("="*60)
    
    # Load all STEM subjects
    dataset = MMLUDataset(split="validation", category="stem")
    print(f"\nLoaded {len(dataset)} STEM examples")
    
    # Get statistics
    stats = dataset.get_statistics()
    print(f"\nStatistics:")
    print(f"  - Subjects: {stats['num_subjects']}")
    print(f"  - Examples per category: {stats['examples_per_category']}")
    
    # Look at first example
    example = dataset[0]
    print(f"\nFirst example:")
    print(f"  Question: {example.question}")
    print(f"  Choices: {example.choices}")
    print(f"  Answer: {example.answer} ({example.get_correct_answer()})")
    print(f"  Subject: {example.metadata['subject']}")
    
    # Format as prompt
    print(f"\nFormatted prompt (multiple choice):")
    print(example.format_prompt(style="multiple_choice"))
    
    # Format with a specific answer for correctness checking
    print(f"\nChecking correctness:")
    correct_text = example.format_with_answer(example.get_correct_answer())
    print(f"  Correct: {correct_text}")
    
    wrong_text = example.format_with_answer(example.choices[1])
    print(f"  Wrong: {wrong_text}")
    
    # Sample 5 random examples
    print(f"\nRandom sample of 5:")
    sample = dataset.sample(5, seed=42)
    for i, ex in enumerate(sample, 1):
        print(f"  {i}. {ex.metadata['subject']}: {ex.question[:50]}...")


def example_triviaqa():
    """Example: Loading and using TriviaQA dataset."""
    print("\n" + "="*60)
    print("TRIVIAQA DATASET EXAMPLE")
    print("="*60)
    
    # Load validation set (limit to 100 for speed)
    dataset = TriviaQADataset(split="validation", max_examples=100)
    print(f"\nLoaded {len(dataset)} examples")
    
    # Get statistics
    stats = dataset.get_statistics()
    print(f"\nStatistics:")
    print(f"  - Average gold answers: {stats['avg_gold_answers']:.2f}")
    print(f"  - Max gold answers: {stats['max_gold_answers']}")
    
    # Look at first example
    example = dataset[0]
    print(f"\nFirst example:")
    print(f"  Question: {example.question}")
    print(f"  Gold answers: {example.metadata['gold_answers']}")
    
    # Test correctness checking
    print(f"\nTesting answer correctness:")
    
    # Exact match
    gold = example.metadata['gold_answers'][0]
    print(f"  Gold answer: '{gold}'")
    print(f"  Exact match: {dataset.is_correct(example, gold, fuzzy_match=False)}")
    
    # Fuzzy match (case insensitive, substring)
    variant = gold.upper()
    print(f"  Uppercase variant: '{variant}'")
    print(f"  Fuzzy match: {dataset.is_correct(example, variant, fuzzy_match=True)}")
    
    # Wrong answer
    wrong = "This is definitely wrong"
    print(f"  Wrong answer: '{wrong}'")
    print(f"  Match: {dataset.is_correct(example, wrong, fuzzy_match=True)}")


def example_gsm8k():
    """Example: Loading and using GSM8K dataset."""
    print("\n" + "="*60)
    print("GSM8K DATASET EXAMPLE")
    print("="*60)
    
    # Load test set (limit to 50 for speed)
    dataset = GSM8KDataset(split="test", max_examples=50)
    print(f"\nLoaded {len(dataset)} examples")
    
    # Get statistics
    stats = dataset.get_statistics()
    print(f"\nStatistics:")
    print(f"  - Average answer: {stats['avg_answer']:.2f}")
    print(f"  - Max answer: {stats['max_answer']}")
    print(f"  - Negative answers: {stats['negative_answers']}")
    print(f"  - Decimal answers: {stats['decimal_answers']}")
    
    # Look at first example
    example = dataset[0]
    print(f"\nFirst example:")
    print(f"  Question: {example.question}")
    print(f"  Gold answer: {example.metadata['gold_answer']}")
    print(f"  Solution steps:\n{example.metadata['solution'][:200]}...")
    
    # Test correctness checking with different answer formats
    print(f"\nTesting answer extraction and correctness:")
    
    gold = example.metadata['gold_answer']
    
    # Format 1: Just the number
    test1 = str(gold)
    print(f"  '{test1}' -> {dataset.is_correct(example, test1)}")
    
    # Format 2: With "The answer is" preamble
    test2 = f"The answer is {gold}"
    print(f"  '{test2}' -> {dataset.is_correct(example, test2)}")
    
    # Format 3: Full solution with #### format
    test3 = f"Let me solve this step by step.\nStep 1: ...\n#### {gold}"
    print(f"  'Solution with #### {gold}' -> {dataset.is_correct(example, test3)}")
    
    # Format 4: Wrong answer
    test4 = f"The answer is {gold + 100}"
    print(f"  'Wrong answer {gold + 100}' -> {dataset.is_correct(example, test4)}")


def example_unified_workflow():
    """Example: Unified workflow for any dataset."""
    print("\n" + "="*60)
    print("UNIFIED WORKFLOW EXAMPLE")
    print("="*60)
    
    # Load dataset
    dataset = MMLUDataset(split="validation", subjects=["anatomy"])
    print(f"\nLoaded {len(dataset)} examples from anatomy")
    
    # Simulate getting predictions from a model
    # In practice, you'd run your LLM here
    print("\nSimulating model predictions...")
    
    predictions = []
    confidences = []
    labels = []
    
    for i, example in enumerate(dataset[:10]):  # Just first 10
        # Format question for model
        prompt = example.format_prompt(style="multiple_choice")
        
        # Simulate model response
        # In reality: model_output = your_model.generate(prompt)
        import random
        random.seed(i)
        
        predicted_idx = random.randint(0, 3)
        confidence = random.uniform(0.5, 1.0)
        
        # Check if correct
        is_correct = (predicted_idx == example.answer)
        
        predictions.append(predicted_idx)
        confidences.append(confidence)
        labels.append(1 if is_correct else 0)
        
        print(f"  Example {i+1}: pred={predicted_idx}, "
              f"gold={example.answer}, conf={confidence:.2f}, "
              f"correct={is_correct}")
    
    # Now you can use these for calibration metrics
    print(f"\nReady for calibration analysis:")
    print(f"  Predictions: {predictions}")
    print(f"  Confidences: {confidences}")
    print(f"  Labels: {labels}")
    print(f"  Accuracy: {sum(labels) / len(labels):.2%}")


if __name__ == "__main__":
    # Run all examples
    example_mmlu()
    example_triviaqa()
    example_gsm8k()
    example_unified_workflow()
    
    print("\n" + "="*60)
    print("All dataset loaders working correctly!")
    print("="*60)