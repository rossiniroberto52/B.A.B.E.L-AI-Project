# Example test file structure

import pytest
import torch
from model import GPT, GPTConfig


class TestGPTConfig:
    """Test GPTConfig dataclass"""
    
    def test_default_config(self):
        """Test that default config is created correctly"""
        config = GPTConfig()
        assert config.block_size == 256
        assert config.vocab_size == 65
        assert config.n_layer == 6
        assert config.n_head == 6
        assert config.n_embd == 384
    
    def test_custom_config(self):
        """Test that custom config can be created"""
        config = GPTConfig(
            block_size=512,
            vocab_size=128,
            n_layer=12,
            n_head=8,
            n_embd=512
        )
        assert config.block_size == 512
        assert config.vocab_size == 128
        assert config.n_layer == 12


class TestGPTModel:
    """Test GPT model"""
    
    @pytest.fixture
    def config(self):
        """Create a small config for testing"""
        return GPTConfig(
            block_size=64,
            vocab_size=65,
            n_layer=2,
            n_head=2,
            n_embd=64,
            dropout=0.0
        )
    
    @pytest.fixture
    def model(self, config):
        """Create a model instance"""
        return GPT(config)
    
    def test_model_creation(self, model, config):
        """Test that model is created successfully"""
        assert model.config == config
        assert model.lm_head is not None
    
    def test_forward_pass_without_targets(self, model, config):
        """Test forward pass without targets (inference mode)"""
        batch_size = 2
        seq_len = 32
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        logits, loss = model(idx)
        
        assert logits.shape == (batch_size, 1, config.vocab_size)
        assert loss is None
    
    def test_forward_pass_with_targets(self, model, config):
        """Test forward pass with targets (training mode)"""
        batch_size = 2
        seq_len = 32
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        logits, loss = model(idx, targets)
        
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert loss is not None
        assert loss.item() >= 0
    
    def test_generate(self, model, config):
        """Test text generation"""
        batch_size = 1
        idx = torch.randint(0, config.vocab_size, (batch_size, 10))
        
        generated = model.generate(idx, max_new_tokens=20)
        
        assert generated.shape == (batch_size, 30)
    
    def test_num_parameters(self, model):
        """Test parameter counting"""
        total_params = model.num_parameters()
        assert total_params > 0
        
        params_exclude_emb = model.num_parameters(exclude_embedding=True)
        assert params_exclude_emb < total_params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
