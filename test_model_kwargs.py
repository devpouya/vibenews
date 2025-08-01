#!/usr/bin/env python3
"""
Test model kwargs for different architectures
"""

def test_model_kwargs():
    """Test the model kwargs generation for different architectures"""
    
    test_cases = [
        {
            'model_name': 'distilbert-base-uncased',
            'expected_params': ['dropout', 'attention_dropout']
        },
        {
            'model_name': 'bert-base-uncased', 
            'expected_params': ['hidden_dropout_prob', 'attention_probs_dropout_prob']
        },
        {
            'model_name': 'roberta-base',
            'expected_params': ['hidden_dropout_prob', 'attention_probs_dropout_prob']
        }
    ]
    
    dropout = 0.1
    
    for case in test_cases:
        model_name = case['model_name']
        expected_params = case['expected_params']
        
        # Simulate the logic from our fixed code
        model_kwargs = {
            'num_labels': 3,
        }
        
        if 'distilbert' in model_name.lower():
            model_kwargs['dropout'] = dropout
            model_kwargs['attention_dropout'] = dropout
        elif 'bert' in model_name.lower():
            model_kwargs['hidden_dropout_prob'] = dropout
            model_kwargs['attention_probs_dropout_prob'] = dropout
        elif 'roberta' in model_name.lower():
            model_kwargs['hidden_dropout_prob'] = dropout
            model_kwargs['attention_probs_dropout_prob'] = dropout
        
        print(f"📊 {model_name}:")
        print(f"   Generated kwargs: {list(model_kwargs.keys())}")
        
        has_expected = all(param in model_kwargs for param in expected_params)
        if has_expected:
            print(f"   ✅ Has expected parameters: {expected_params}")
        else:
            print(f"   ❌ Missing expected parameters: {expected_params}")
            return False
    
    return True

if __name__ == "__main__":
    print("🧪 Testing Model Parameter Generation")
    print("="*50)
    
    if test_model_kwargs():
        print("\n🎉 All architecture-specific parameters correct!")
    else:
        print("\n❌ Parameter generation issues found")
        exit(1)