#!/usr/bin/env python3
"""
Test the experiment pipeline with a quick run
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

def test_experiment_pipeline():
    """Test experiment pipeline components"""
    print("🧪 Testing experiment pipeline...")
    
    try:
        # Test configuration loading
        from ml.experiment_config import ExperimentManager, ExperimentConfig
        
        manager = ExperimentManager()
        configs = manager.list_configs()
        print(f"✅ Found {len(configs)} experiment configs: {configs}")
        
        # Load a config
        if configs:
            config = manager.load_config(configs[0])
            print(f"✅ Loaded config: {config.experiment_name}")
            
            # Validate config
            errors = config.validate()
            if errors:
                print(f"❌ Config validation errors: {errors}")
                return False
            else:
                print("✅ Config validation passed")
        
        # Test model factory
        from ml.model_factory import ModelFactory, ModelConfigs
        
        print("✅ Available architectures:", ModelFactory.get_available_architectures())
        
        # Test creating a model config
        model_config = ModelConfigs.bert_base()
        print(f"✅ Created model config: {model_config.architecture}")
        
        # Test experiment tracker (without actual training)
        from ml.experiment_tracker import ExperimentTracker
        
        if configs:
            tracker = ExperimentTracker(config, log_dir="test_runs")
            
            # Test logging some fake metrics
            tracker.log_metrics({
                "test/accuracy": 0.85,
                "test/f1_macro": 0.82,
                "test/loss": 0.45
            }, step=1)
            
            tracker.close()
            print("✅ Experiment tracker test passed")
        
        print("\n🎉 All pipeline tests passed!")
        print("\n📋 Ready to run experiments:")
        print("  python run_experiment.py experiments/configs/bert_baseline.yaml")
        print("  python run_experiment.py experiments/configs/distilbert_fast.yaml")
        print("  python run_experiment.py experiments/configs/roberta_strong.yaml")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure dependencies are installed:")
        print("  pip install tensorboard matplotlib seaborn")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_experiment_pipeline()
    sys.exit(0 if success else 1)