"""
Test script to validate the new Task Token architecture.
This script tests:
1. Model initialization
2. Forward pass
3. Feature extraction with task tokens
4. Output shapes
5. Frozen weights verification
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import yaml
from dflip_models.qwen_vision import DFLIPProfiler, create_profiler


def test_model_initialization():
    """Test if model initializes correctly."""
    print("=" * 60)
    print("TEST 1: Model Initialization")
    print("=" * 60)
    
    try:
        model = DFLIPProfiler(
            model_name="Qwen/Qwen2.5-VL-7B-Instruct",
            num_generators=10,
            extract_layers=[6, 12, 18],
            device='cpu',  # Use CPU for testing
            cache_dir='./models_cache'
        )
        print("✓ Model initialized successfully")
        return model
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        raise


def test_frozen_weights(model):
    """Test if Qwen VIT weights are frozen."""
    print("\n" + "=" * 60)
    print("TEST 2: Frozen Weights Verification")
    print("=" * 60)
    
    # Check base model is frozen
    frozen_params = sum(1 for p in model.base_model.parameters() if not p.requires_grad)
    total_base_params = sum(1 for p in model.base_model.parameters())
    
    print(f"Frozen parameters in base model: {frozen_params}/{total_base_params}")
    
    if frozen_params == total_base_params:
        print("✓ All Qwen VIT parameters are frozen")
    else:
        print("✗ Some Qwen VIT parameters are trainable (unexpected)")
    
    # Check feature fusion module is trainable
    trainable_fusion = sum(1 for p in model.feature_fusion.parameters() if p.requires_grad)
    total_fusion = sum(1 for p in model.feature_fusion.parameters())
    
    print(f"Trainable parameters in feature fusion: {trainable_fusion}/{total_fusion}")
    
    if trainable_fusion == total_fusion:
        print("✓ All Feature Fusion parameters are trainable")
    else:
        print("✗ Some Feature Fusion parameters are frozen (unexpected)")
    
    # Check task tokens are trainable
    task_token_params = [model.binary_task_token, model.multiclass_task_token, model.localization_task_token]
    trainable_tokens = sum(1 for p in task_token_params if p.requires_grad)
    
    print(f"Trainable task tokens: {trainable_tokens}/{len(task_token_params)}")
    
    if trainable_tokens == len(task_token_params):
        print("✓ All task tokens are trainable")
    else:
        print("✗ Some task tokens are frozen (unexpected)")
    
    # Check heads are trainable
    trainable_heads = (
        sum(1 for p in model.detection_head.parameters() if p.requires_grad) +
        sum(1 for p in model.identification_head.parameters() if p.requires_grad) +
        sum(1 for p in model.localization_head.parameters() if p.requires_grad)
    )
    total_heads = (
        sum(1 for p in model.detection_head.parameters()) +
        sum(1 for p in model.identification_head.parameters()) +
        sum(1 for p in model.localization_head.parameters())
    )
    
    print(f"Trainable parameters in task heads: {trainable_heads}/{total_heads}")
    
    if trainable_heads == total_heads:
        print("✓ All task head parameters are trainable")
    else:
        print("✗ Some task head parameters are frozen (unexpected)")


def test_forward_pass(model):
    """Test forward pass with dummy data."""
    print("\n" + "=" * 60)
    print("TEST 3: Forward Pass")
    print("=" * 60)
    
    # Create dummy input
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 448, 448, dtype=torch.bfloat16)
    
    print(f"Input shape: {dummy_images.shape}")
    
    try:
        with torch.no_grad():
            outputs = model(dummy_images, return_features=True)
        print("✓ Forward pass successful")
        return outputs
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_output_shapes(outputs, batch_size=2, num_generators=10):
    """Test if output shapes are correct."""
    print("\n" + "=" * 60)
    print("TEST 4: Output Shape Validation")
    print("=" * 60)
    
    expected_shapes = {
        'detection_logits': (batch_size, 2),
        'identification_logits': (batch_size, num_generators),
        'localization_mask': (batch_size, 1, 448, 448),
        'binary_features': (batch_size, None),  # Variable size
        'multiclass_features': (batch_size, None),  # Variable size
        'spatial_features': (batch_size, None, None, None)  # Variable size
    }
    
    all_correct = True
    for key, expected_shape in expected_shapes.items():
        if key not in outputs:
            print(f"✗ Missing output: {key}")
            all_correct = False
            continue
        
        actual_shape = tuple(outputs[key].shape)
        
        # Check matching dimensions
        if key in ['binary_features', 'multiclass_features', 'spatial_features']:
            # Just check batch size for variable-size outputs
            if actual_shape[0] == batch_size:
                print(f"✓ {key}: {actual_shape}")
            else:
                print(f"✗ {key}: Expected batch_size={batch_size}, got {actual_shape}")
                all_correct = False
        else:
            if actual_shape == expected_shape:
                print(f"✓ {key}: {actual_shape}")
            else:
                print(f"✗ {key}: Expected {expected_shape}, got {actual_shape}")
                all_correct = False
    
    if all_correct:
        print("\n✓ All output shapes are correct")
    else:
        print("\n✗ Some output shapes are incorrect")
    
    return all_correct


def test_prediction_mode(model):
    """Test prediction mode."""
    print("\n" + "=" * 60)
    print("TEST 5: Prediction Mode")
    print("=" * 60)
    
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 448, 448, dtype=torch.bfloat16)
    
    try:
        predictions = model.predict(dummy_images, threshold=0.5)
        
        print("Prediction outputs:")
        for key, value in predictions.items():
            print(f"  {key}: {value.shape}")
        
        print("✓ Prediction mode successful")
        return True
    except Exception as e:
        print(f"✗ Prediction mode failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_save_load(model):
    """Test save and load weights."""
    print("\n" + "=" * 60)
    print("TEST 6: Save and Load Weights")
    print("=" * 60)
    
    save_path = "./test_weights"
    os.makedirs(save_path, exist_ok=True)
    
    try:
        # Save weights
        model.save_weights(save_path)
        print("✓ Weights saved successfully")
        
        # Create a new model
        new_model = DFLIPProfiler(
            model_name="Qwen/Qwen2.5-VL-7B-Instruct",
            num_generators=10,
            extract_layers=[6, 12, 18],
            device='cpu',
            cache_dir='./models_cache'
        )
        
        # Load weights
        new_model.load_weights(save_path)
        print("✓ Weights loaded successfully")
        
        # Clean up
        import shutil
        shutil.rmtree(save_path)
        
        return True
    except Exception as e:
        print(f"✗ Save/load failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_factory():
    """Test creating model from config."""
    print("\n" + "=" * 60)
    print("TEST 7: Config Factory")
    print("=" * 60)
    
    # Load config
    config_path = "../configs/dflip_config.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override device for testing
        config['hardware'] = {'device': 'cpu'}
        
        model = create_profiler(config)
        print("✓ Model created from config successfully")
        return True
    except Exception as e:
        print(f"✗ Config factory failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("DFLIP PROFILER ARCHITECTURE VALIDATION")
    print("=" * 60)
    
    try:
        # Test 1: Initialization
        model = test_model_initialization()
        
        # Test 2: Frozen weights
        test_frozen_weights(model)
        
        # Test 3: Forward pass
        outputs = test_forward_pass(model)
        
        # Test 4: Output shapes
        test_output_shapes(outputs)
        
        # Test 5: Prediction mode
        test_prediction_mode(model)
        
        # Test 6: Save/Load
        test_save_load(model)
        
        # Test 7: Config factory
        test_config_factory()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"TESTS FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
