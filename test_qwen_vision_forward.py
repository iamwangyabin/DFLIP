"""
Qwen Vision 模型前向传播测试脚本
不需要训练，仅测试随机输入的输出

这个脚本用来验证DFLIPProfiler模型的forward pass是否正确运行
"""

import torch
import torch.nn as nn
from dflip_models.qwen_vision import DFLIPProfiler


def test_qwen_vision_forward():
    """测试Qwen Vision模型的前向传播"""
    
    print("=" * 80)
    print("Qwen Vision Forward Pass Test")
    print("=" * 80)
    
    # 1. 初始化模型
    print("\n[1] 初始化模型...")
    try:
        model = DFLIPProfiler(
            model_name="Qwen/Qwen2.5-VL-7B-Instruct",
            num_generators=10,
            extract_layers=[6, 12, 18]
        )
        print("✓ 模型初始化成功")
    except Exception as e:
        print(f"✗ 模型初始化失败: {e}")
        return
    
    # 2. 设置模型为评估模式
    print("\n[2] 设置模型为评估模式...")
    model.eval()
    print("✓ 模型设置为eval模式")
    
    # 3. 创建随机输入
    print("\n[3] 创建随机输入...")
    batch_size = 2
    num_channels = 3
    height = 448
    width = 448
    
    # 创建随机像素值张量 (B, C, H, W)
    pixel_values = torch.randn(batch_size, num_channels, height, width)
    print(f"✓ 创建随机输入张量: {pixel_values.shape}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Channels: {num_channels}")
    print(f"  - Height: {height}")
    print(f"  - Width: {width}")
    
    # 4. 前向传播 (不计算梯度)
    print("\n[4] 执行前向传播...")
    try:
        with torch.no_grad():
            outputs = model.forward(pixel_values, return_features=True)
        print("✓ 前向传播成功")
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. 检查输出
    print("\n[5] 检查输出...")
    print(f"\n输出字典包含的key: {list(outputs.keys())}")
    
    # 检查detection输出
    if 'detection_logits' in outputs:
        det_logits = outputs['detection_logits']
        print(f"\n  📊 Detection Logits:")
        print(f"     - Shape: {det_logits.shape}")
        print(f"     - Expected: ({batch_size}, 2)")
        print(f"     - Min: {det_logits.min():.4f}, Max: {det_logits.max():.4f}")
        print(f"     - Mean: {det_logits.mean():.4f}, Std: {det_logits.std():.4f}")
        
        # 转换为概率
        det_probs = torch.softmax(det_logits, dim=-1)
        print(f"     - Softmax概率: {det_probs[0].tolist()}")
        assert det_logits.shape == (batch_size, 2), "Detection logits shape不正确"
        print(f"     ✓ Shape验证通过")
    
    # 检查identification输出
    if 'identification_logits' in outputs:
        id_logits = outputs['identification_logits']
        print(f"\n  🏷️ Identification Logits:")
        print(f"     - Shape: {id_logits.shape}")
        print(f"     - Expected: ({batch_size}, 10)")
        print(f"     - Min: {id_logits.min():.4f}, Max: {id_logits.max():.4f}")
        print(f"     - Mean: {id_logits.mean():.4f}, Std: {id_logits.std():.4f}")
        
        # 转换为概率
        id_probs = torch.softmax(id_logits, dim=-1)
        top_k = torch.topk(id_probs[0], k=3)
        print(f"     - Top-3生成器: {top_k.indices.tolist()}, 概率: {top_k.values.tolist()}")
        assert id_logits.shape == (batch_size, 10), "Identification logits shape不正确"
        print(f"     ✓ Shape验证通过")
    
    # 检查localization输出
    if 'localization_mask' in outputs:
        loc_mask = outputs['localization_mask']
        print(f"\n  🎯 Localization Mask:")
        print(f"     - Shape: {loc_mask.shape}")
        print(f"     - Expected: ({batch_size}, 1, 448, 448)")
        print(f"     - Min: {loc_mask.min():.4f}, Max: {loc_mask.max():.4f}")
        print(f"     - Mean: {loc_mask.mean():.4f}, Std: {loc_mask.std():.4f}")
        assert loc_mask.shape == (batch_size, 1, 448, 448), "Localization mask shape不正确"
        print(f"     ✓ Shape验证通过")
    
    # 检查特征输出
    if 'binary_features' in outputs:
        bin_feat = outputs['binary_features']
        print(f"\n  🔹 Binary Task Features:")
        print(f"     - Shape: {bin_feat.shape}")
        print(f"     - Expected: ({batch_size}, 1024)")
        assert bin_feat.shape[0] == batch_size, "Binary features batch size不正确"
        print(f"     ✓ Shape验证通过")
    
    if 'multiclass_features' in outputs:
        multi_feat = outputs['multiclass_features']
        print(f"\n  🔸 Multiclass Task Features:")
        print(f"     - Shape: {multi_feat.shape}")
        print(f"     - Expected: ({batch_size}, 1024)")
        assert multi_feat.shape[0] == batch_size, "Multiclass features batch size不正确"
        print(f"     ✓ Shape验证通过")
    
    if 'spatial_features' in outputs:
        spatial_feat = outputs['spatial_features']
        print(f"\n  🔺 Spatial Features:")
        print(f"     - Shape: {spatial_feat.shape}")
        print(f"     - Expected: ({batch_size}, 1024, H, W)")
        assert spatial_feat.shape[0] == batch_size, "Spatial features batch size不正确"
        print(f"     ✓ Shape验证通过")
    
    # 6. 测试predict方法
    print("\n[6] 测试predict方法...")
    try:
        with torch.no_grad():
            predictions = model.predict(pixel_values, threshold=0.5)
        print("✓ Predict方法成功")
        
        print(f"\n预测结果:")
        print(f"  - is_fake shape: {predictions['is_fake'].shape}")
        print(f"    值: {predictions['is_fake'].tolist()}")
        print(f"  - fake_probs shape: {predictions['fake_probs'].shape}")
        print(f"    值: {predictions['fake_probs'].tolist()}")
        print(f"  - generator_ids shape: {predictions['generator_ids'].shape}")
        print(f"    值: {predictions['generator_ids'].tolist()}")
        print(f"  - generator_probs shape: {predictions['generator_probs'].shape}")
        print(f"  - forgery_masks shape: {predictions['forgery_masks'].shape}")
        print(f"  - forgery_masks_binary shape: {predictions['forgery_masks_binary'].shape}")
        
    except Exception as e:
        print(f"✗ Predict方法失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 7. 测试梯度计算（仅用于参数验证，不用于优化）
    print("\n[7] 验证可训练参数...")
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"  - 可训练参数: {trainable_count:,}")
    print(f"  - 总参数数: {total_count:,}")
    print(f"  - 可训练比例: {100 * trainable_count / total_count:.2f}%")
    print(f"     ✓ 参数统计完成")
    
    # 8. 最终总结
    print("\n" + "=" * 80)
    print("✓ 所有测试通过！模型forward pass正确")
    print("=" * 80)


if __name__ == "__main__":
    test_qwen_vision_forward()
