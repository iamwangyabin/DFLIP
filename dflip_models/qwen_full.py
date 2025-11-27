"""
Stage 2: The Interpreter
Full Qwen VLM with frozen Stage 1 weights for prompt prediction.
"""

import torch
import torch.nn as nn
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from typing import Dict, Optional, List

from dflip_dataset import format_stage2_conversation


class DFLIPInterpreter(nn.Module):
    """
    Stage 2: The Interpreter
    
    Full Vision-Language Model that:
    1. Takes Stage 1 profiling results as context
    2. Predicts the generation prompt
    3. Provides natural language explanations
    
    Uses complete Qwen2.5-VL with LoRA on LLM layers.
    Freezes Stage 1 vision LoRA weights.
    
    Args:
        model_name: Hugging Face model name
        stage1_checkpoint: Path to Stage 1 LoRA checkpoint
        lora_config: LoRA configuration for LLM
        freeze_vision: Whether to freeze vision encoder
        device: Target device
        cache_dir: Model cache directory
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        stage1_checkpoint: Optional[str] = None,
        lora_config: Optional[Dict] = None,
        freeze_vision: bool = True,
        cache_dir: Optional[str] = None
    ):
        super().__init__()
        
        self.model_name = model_name
        self.freeze_vision = freeze_vision
        
        # Load base Qwen VL model
        print(f"Loading {model_name}...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype="auto",
            device_map="auto"
        )
        
        # Load Stage 1 LoRA weights if provided
        if stage1_checkpoint:
            self._load_stage1_weights(stage1_checkpoint)
        
        # Apply LoRA to LLM layers
        if lora_config is not None:
            self.model = self._apply_llm_lora(lora_config)
        
        # Freeze vision encoder if requested
        if freeze_vision:
            self._freeze_vision_encoder()
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        print("Initialized DFLIP Interpreter")
    
    def _load_stage1_weights(self, checkpoint_path: str):
        """Load Stage 1 LoRA weights into vision encoder."""
        print(f"Loading Stage 1 weights from {checkpoint_path}...")
        
        # Load Stage 1 vision LoRA
        self.model = PeftModel.from_pretrained(
            self.model,
            checkpoint_path,
            is_trainable=False  # Freeze Stage 1 weights
        )
        
        print("Loaded and froze Stage 1 vision weights")
    
    def _apply_llm_lora(self, lora_config: Dict):
        """Apply LoRA adapters to language model layers."""
        config = LoraConfig(
            r=lora_config.get('r', 16),
            lora_alpha=lora_config.get('lora_alpha', 32),
            target_modules=lora_config.get('target_modules', [
                'q_proj', 'v_proj', 'k_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj'
            ]),
            lora_dropout=lora_config.get('lora_dropout', 0.1),
            bias=lora_config.get('bias', 'none'),
            task_type=TaskType.CAUSAL_LM
        )
        
        print("Applying LoRA to language model...")
        
        # If model already has PEFT adapters (from Stage 1), add new adapter
        if hasattr(self.model, 'add_adapter'):
            self.model.add_adapter("llm_lora", config)
            self.model.set_adapter("llm_lora")
        else:
            model = get_peft_model(self.model, config)
        
        if hasattr(self.model, 'print_trainable_parameters'):
            self.model.print_trainable_parameters()
        
        return self.model
    
    def _freeze_vision_encoder(self):
        """Freeze all vision encoder parameters."""
        for name, param in self.model.named_parameters():
            if 'visual' in name or 'vision' in name:
                param.requires_grad = False
        
        print("Frozen vision encoder parameters")
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Forward pass for training.
        
        Args:
            pixel_values: (B, C, H, W) images
            input_ids: (B, L) tokenized conversation
            attention_mask: (B, L) attention mask
            labels: (B, L) labels for language modeling loss
        
        Returns:
            Dictionary with loss and logits
        """
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return {
            'loss': outputs.loss,
            'logits': outputs.logits
        }
    
    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        profiling_results: Optional[Dict] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> List[str]:
        """
        Generate prompt predictions and explanations.
        
        Args:
            pixel_values: (B, C, H, W) images
            profiling_results: Optional Stage 1 results to include in context
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
        
        Returns:
            List of generated text responses
        """
        self.eval()
        
        batch_size = pixel_values.shape[0]
        
        # Create conversation prompt
        conversations = []
        for i in range(batch_size):
            if profiling_results:
                # Use Stage 1 results
                is_fake = profiling_results['is_fake'][i].item()
                generator_id = profiling_results['generator_ids'][i].item()
                # TODO: Map generator_id to name
                generator_name = f"Generator_{generator_id}"
                
                messages = format_stage2_conversation(
                    is_fake=bool(is_fake),
                    generator=generator_name if is_fake else None,
                    include_assistant=False  # Don't include answer for generation
                )
            else:
                # Default prompt without Stage 1 context
                messages = format_stage2_conversation(
                    is_fake=False,
                    include_assistant=False
                )
            
            conversations.append(messages)
        
        # Format with processor
        texts = [
            self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            for conv in conversations
        ]
        
        # Tokenize
        inputs = self.processor(
            images=[pixel_values[i] for i in range(batch_size)],
            text=texts,
            return_tensors='pt',
            padding=True
        )
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id
        )
        
        # Decode
        generated_texts = self.processor.batch_decode(
            outputs[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_texts
    
    def save_lora_weights(self, save_path: str):
        """Save LLM LoRA weights."""
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(save_path)
        
        print(f"Saved Stage 2 weights to {save_path}")
    
    def load_lora_weights(self, load_path: str):
        """Load LLM LoRA weights."""
        self.model = PeftModel.from_pretrained(
            self.model,
            load_path,
            is_trainable=True
        )
        
        print(f"Loaded Stage 2 weights from {load_path}")


class DFLIPFullPipeline(nn.Module):
    """
    Complete DFLIP pipeline combining Stage 1 and Stage 2.
    
    Args:
        profiler: DFLIPProfiler instance (Stage 1)
        interpreter: DFLIPInterpreter instance (Stage 2)
    """
    
    def __init__(self, profiler, interpreter):
        super().__init__()
        
        self.profiler = profiler
        self.interpreter = interpreter
        
        # Set profiler to eval mode (frozen)
        self.profiler.eval()
        for param in self.profiler.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def predict(
        self,
        pixel_values: torch.Tensor,
        return_stage1: bool = True,
        **generation_kwargs
    ) -> Dict:
        """
        End-to-end prediction pipeline.
        
        Args:
            pixel_values: (B, C, H, W) images
            return_stage1: Whether to return Stage 1 results
            **generation_kwargs: Arguments for Stage 2 generation
        
        Returns:
            Dictionary with Stage 1 and Stage 2 results
        """
        # Stage 1: Profiling
        stage1_outputs = self.profiler.predict(pixel_values)
        
        # Stage 2: Interpretation
        generated_texts = self.interpreter.generate(
            pixel_values=pixel_values,
            profiling_results=stage1_outputs,
            **generation_kwargs
        )
        
        results = {
            'generated_prompts': generated_texts
        }
        
        if return_stage1:
            results.update({
                'is_fake': stage1_outputs['is_fake'],
                'fake_probs': stage1_outputs['fake_probs'],
                'generator_ids': stage1_outputs['generator_ids'],
                'forgery_masks': stage1_outputs['forgery_masks']
            })
        
        return results


def create_interpreter(config: Dict, stage1_checkpoint: Optional[str] = None) -> DFLIPInterpreter:
    """
    Factory function to create DFLIPInterpreter from config.
    
    Args:
        config: Configuration dictionary
        stage1_checkpoint: Path to Stage 1 checkpoint
    
    Returns:
        Initialized DFLIPInterpreter model
    """
    model_config = config.get('model', {})
    stage2_config = config.get('stage2_training', {})
    
    if stage1_checkpoint is None:
        stage1_checkpoint = stage2_config.get('stage1_checkpoint')
    
    interpreter = DFLIPInterpreter(
        model_name=model_config.get('base_model', 'Qwen/Qwen3-VL-2B-Instruct'),
        stage1_checkpoint=stage1_checkpoint,
        lora_config=model_config.get('stage2_lora'),
        freeze_vision=stage2_config.get('freeze_vision', True),
        cache_dir=model_config.get('cache_dir')
    )
    
    return interpreter
