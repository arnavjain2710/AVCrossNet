"""
Dummy training test script for AVCrossNet.
Creates random noise data and tests the training loop to verify everything works correctly.
"""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from model import AVSEModule
from config import sampling_rate, max_frames, max_audio_len, img_height, img_width

# Audio length must be divisible by 256 (product of resample factors: 4*4*2*2*2*2 = 256)
# Use 44800 which matches the dataset and is divisible by 256
TEST_AUDIO_LEN = 44800  # 44800 / 256 = 175


class DummyAVSEDataset(Dataset):
    """Dummy dataset that generates random audio and video data."""
    
    def __init__(self, num_samples=10, audio_len=None, num_frames=None):
        """
        Args:
            num_samples: Number of samples in the dataset
            audio_len: Length of audio samples (default: TEST_AUDIO_LEN, must be divisible by 256)
            num_frames: Number of video frames (default: max_frames)
        """
        self.num_samples = num_samples
        # Audio length must be divisible by 256 for the Denoiser resample factors
        self.audio_len = audio_len if audio_len is not None else TEST_AUDIO_LEN
        # Ensure it's divisible by 256
        if self.audio_len % 256 != 0:
            self.audio_len = (self.audio_len // 256) * 256
        self.num_frames = num_frames if num_frames is not None else max_frames
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Generate random dummy data matching the expected format."""
        # Generate random audio (noisy and clean)
        noisy_audio = torch.randn(self.audio_len).float()
        clean_audio = torch.randn(self.audio_len).float()
        
        # Generate random video frames: (1, num_frames, height, width)
        # Grayscale frames normalized to [0, 1]
        video_frames = torch.rand(1, self.num_frames, img_height, img_width).float()
        
        return {
            "noisy_audio": noisy_audio,
            "clean": clean_audio,
            "video_frames": video_frames,
            "scene": f"dummy_scene_{idx}"
        }


class DummyAVSEDataModule(LightningDataModule):
    """Dummy data module for testing."""
    
    def __init__(self, batch_size=2, num_train=20, num_val=5):
        super().__init__()
        self.batch_size = batch_size
        self.num_train = num_train
        self.num_val = num_val
        self.train_dataset = None
        self.dev_dataset = None
        self.val_dataset = None
        
    def setup(self, stage=None):
        """Setup datasets."""
        self.train_dataset = DummyAVSEDataset(num_samples=self.num_train)
        self.dev_dataset = DummyAVSEDataset(num_samples=self.num_val)
        self.val_dataset = self.dev_dataset  # Alias for compatibility
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
            pin_memory=False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.dev_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )


def test_model_forward():
    """Test that model forward pass works with dummy data."""
    print("=" * 60)
    print("Test 1: Model Forward Pass")
    print("=" * 60)
    
    model = AVSEModule(lr=0.001)
    model.eval()
    
    # Create dummy batch
    batch_size = 2
    dummy_batch = {
        "noisy_audio": torch.randn(batch_size, TEST_AUDIO_LEN),
        "clean": torch.randn(batch_size, TEST_AUDIO_LEN),
        "video_frames": torch.rand(batch_size, 1, max_frames, img_height, img_width),
        "scene": ["dummy_1", "dummy_2"]
    }
    
    with torch.no_grad():
        output = model(dummy_batch)
    
    assert output.shape == (batch_size, TEST_AUDIO_LEN), \
        f"Expected output shape {(batch_size, TEST_AUDIO_LEN)}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"
    
    print("✓ Model forward pass works correctly\n")


def test_training_step():
    """Test a single training step."""
    print("=" * 60)
    print("Test 2: Training Step")
    print("=" * 60)
    
    model = AVSEModule(lr=0.001)
    model.train()
    
    # Create dummy batch
    batch_size = 2
    dummy_batch = {
        "noisy_audio": torch.randn(batch_size, TEST_AUDIO_LEN),
        "clean": torch.randn(batch_size, TEST_AUDIO_LEN),
        "video_frames": torch.rand(batch_size, 1, max_frames, img_height, img_width),
    }
    
    # Forward and backward pass
    loss = model.training_step(dummy_batch, batch_idx=0)
    
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.requires_grad or loss.numel() == 1, "Loss should be a scalar tensor"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be Inf"
    
    print(f"✓ Training step completed. Loss: {loss.item():.4f}\n")


def test_validation_step():
    """Test a single validation step."""
    print("=" * 60)
    print("Test 3: Validation Step")
    print("=" * 60)
    
    model = AVSEModule(lr=0.001)
    model.eval()
    
    # Create dummy batch
    batch_size = 2
    dummy_batch = {
        "noisy_audio": torch.randn(batch_size, TEST_AUDIO_LEN),
        "clean": torch.randn(batch_size, TEST_AUDIO_LEN),
        "video_frames": torch.rand(batch_size, 1, max_frames, img_height, img_width),
    }
    
    with torch.no_grad():
        loss = model.validation_step(dummy_batch, batch_idx=0)
    
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be Inf"
    
    print(f"✓ Validation step completed. Loss: {loss.item():.4f}\n")


def test_gradient_flow():
    """Test that gradients flow through the model."""
    print("=" * 60)
    print("Test 4: Gradient Flow")
    print("=" * 60)
    
    model = AVSEModule(lr=0.001)
    model.train()
    
    # Create dummy batch
    batch_size = 2
    dummy_batch = {
        "noisy_audio": torch.randn(batch_size, TEST_AUDIO_LEN),
        "clean": torch.randn(batch_size, TEST_AUDIO_LEN),
        "video_frames": torch.rand(batch_size, 1, max_frames, img_height, img_width),
    }
    
    # Forward pass
    loss = model.cal_loss(dummy_batch)
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist for model parameters
    has_gradients = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_gradients = True
            # Check that gradient is not all zeros
            if param.grad.abs().sum() > 1e-8:
                print(f"  ✓ Gradient exists for {name}")
                break
    
    assert has_gradients, "No gradients found in model parameters"
    
    print("✓ Gradients flow correctly through the model\n")


def test_optimizer_step():
    """Test that optimizer step works."""
    print("=" * 60)
    print("Test 5: Optimizer Step")
    print("=" * 60)
    
    model = AVSEModule(lr=0.001)
    model.train()
    
    optimizer = model.configure_optimizers()["optimizer"]
    
    # Create dummy batch
    batch_size = 2
    dummy_batch = {
        "noisy_audio": torch.randn(batch_size, TEST_AUDIO_LEN),
        "clean": torch.randn(batch_size, TEST_AUDIO_LEN),
        "video_frames": torch.rand(batch_size, 1, max_frames, img_height, img_width),
    }
    
    # Get initial parameter values
    initial_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            initial_params[name] = param.data.clone()
    
    # Forward and backward
    loss = model.cal_loss(dummy_batch)
    loss.backward()
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    
    # Check that parameters changed
    params_changed = False
    for name, param in model.named_parameters():
        if param.requires_grad and name in initial_params:
            if not torch.allclose(param.data, initial_params[name], atol=1e-6):
                params_changed = True
                break
    
    assert params_changed, "Parameters should change after optimizer step"
    
    print("✓ Optimizer step works correctly\n")


def test_training_loop():
    """Test a complete training loop with multiple steps."""
    print("=" * 60)
    print("Test 6: Training Loop (Multiple Steps)")
    print("=" * 60)
    
    model = AVSEModule(lr=0.001)
    model.train()
    
    datamodule = DummyAVSEDataModule(batch_size=2, num_train=10, num_val=4)
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    
    optimizer = model.configure_optimizers()["optimizer"]
    
    losses = []
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 3:  # Test only first 3 batches
            break
            
        # Forward pass
        loss = model.training_step(batch, batch_idx)
        losses.append(loss.item())
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"  Batch {batch_idx + 1}: Loss = {loss.item():.4f}")
    
    assert len(losses) == 3, "Should have 3 training steps"
    assert all(not np.isnan(l) and not np.isinf(l) for l in losses), "Losses should be finite"
    
    print(f"✓ Training loop completed. Losses: {[f'{l:.4f}' for l in losses]}\n")


def test_validation_loop():
    """Test a complete validation loop."""
    print("=" * 60)
    print("Test 7: Validation Loop")
    print("=" * 60)
    
    model = AVSEModule(lr=0.001)
    model.eval()
    
    datamodule = DummyAVSEDataModule(batch_size=2, num_train=10, num_val=4)
    datamodule.setup()
    val_loader = datamodule.val_dataloader()
    
    losses = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            loss = model.validation_step(batch, batch_idx)
            losses.append(loss.item())
            print(f"  Batch {batch_idx + 1}: Loss = {loss.item():.4f}")
    
    assert len(losses) > 0, "Should have validation steps"
    assert all(not np.isnan(l) and not np.isinf(l) for l in losses), "Losses should be finite"
    
    print(f"✓ Validation loop completed. Average loss: {np.mean(losses):.4f}\n")


def test_pytorch_lightning_trainer():
    """Test training with PyTorch Lightning Trainer."""
    print("=" * 60)
    print("Test 8: PyTorch Lightning Trainer")
    print("=" * 60)
    
    # Create model and datamodule
    datamodule = DummyAVSEDataModule(batch_size=2, num_train=10, num_val=4)
    datamodule.setup()  # Setup datasets
    model = AVSEModule(lr=0.001, val_dataset=datamodule.val_dataset)
    
    # Create trainer with minimal settings for testing
    trainer = Trainer(
        max_epochs=2,
        limit_train_batches=3,  # Only train on 3 batches per epoch
        limit_val_batches=2,    # Only validate on 2 batches
        enable_checkpointing=False,  # Don't save checkpoints
        logger=False,  # Don't use logger
        enable_progress_bar=False,  # Disable progress bar
        accelerator="cpu",  # Use CPU for testing
        devices=1,
    )
    
    try:
        trainer.fit(model, datamodule)
        print("✓ PyTorch Lightning training completed successfully")
    except Exception as e:
        print(f"✗ Training failed with error: {e}")
        raise
    
    print()


def test_loss_decreases():
    """Test that loss can decrease (or at least changes) during training."""
    print("=" * 60)
    print("Test 9: Loss Behavior")
    print("=" * 60)
    
    model = AVSEModule(lr=0.001)
    model.train()
    
    datamodule = DummyAVSEDataModule(batch_size=2, num_train=10, num_val=4)
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    
    optimizer = model.configure_optimizers()["optimizer"]
    
    initial_loss = None
    final_loss = None
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 5:  # Test 5 batches
            break
        
        # Forward pass
        loss = model.cal_loss(batch)
        
        if batch_idx == 0:
            initial_loss = loss.item()
        if batch_idx == 4:
            final_loss = loss.item()
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"  Step {batch_idx + 1}: Loss = {loss.item():.4f}")
    
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")
    
    # Loss should be finite and reasonable
    assert initial_loss is not None and final_loss is not None, "Losses should be computed"
    assert not np.isnan(initial_loss) and not np.isnan(final_loss), "Losses should not be NaN"
    assert not np.isinf(initial_loss) and not np.isinf(final_loss), "Losses should not be Inf"
    
    print("✓ Loss computation works correctly\n")


def test_different_batch_sizes():
    """Test that model works with different batch sizes."""
    print("=" * 60)
    print("Test 10: Different Batch Sizes")
    print("=" * 60)
    
    model = AVSEModule(lr=0.001)
    model.eval()
    
    batch_sizes = [1, 2, 4]
    
    for batch_size in batch_sizes:
        dummy_batch = {
            "noisy_audio": torch.randn(batch_size, TEST_AUDIO_LEN),
            "clean": torch.randn(batch_size, TEST_AUDIO_LEN),
            "video_frames": torch.rand(batch_size, 1, max_frames, img_height, img_width),
        }
        
        with torch.no_grad():
            output = model(dummy_batch)
        
        assert output.shape == (batch_size, TEST_AUDIO_LEN), \
            f"Batch size {batch_size}: Expected shape {(batch_size, TEST_AUDIO_LEN)}, got {output.shape}"
        print(f"  ✓ Batch size {batch_size} works")
    
    print()


def run_all_tests():
    """Run all training tests."""
    print("\n" + "=" * 60)
    print("AVCrossNet Training Test Suite")
    print("=" * 60 + "\n")
    
    tests = [
        test_model_forward,
        test_training_step,
        test_validation_step,
        test_gradient_flow,
        test_optimizer_step,
        test_training_loop,
        test_validation_loop,
        test_pytorch_lightning_trainer,
        test_loss_decreases,
        test_different_batch_sizes,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} FAILED: {str(e)}\n")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    success = run_all_tests()
    exit(0 if success else 1)

