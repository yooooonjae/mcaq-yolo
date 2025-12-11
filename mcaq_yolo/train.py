"""
Training script for MCAQ-YOLO
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml
import argparse
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO
from models.mcaq_yolo import MCAQYOLO
from core.curriculum import CurriculumScheduler, ComplexityBasedSampler
from utils.dataset import ComplexityDataset, compute_dataset_complexity
from utils.evaluation import evaluate_mcaq_yolo
from utils.visualization import plot_training_curves, visualize_complexity_map


class Trainer:
    """
    Main trainer class for MCAQ-YOLO
    """
    
    def __init__(self, config: dict):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device(config['device'])
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tensorboard
        self.writer = SummaryWriter(self.output_dir / 'tensorboard')
        
        # Training parameters
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']
        self.save_interval = config.get('save_interval', 10)
        self.eval_interval = config.get('eval_interval', 5)
        
        # Quantization parameters
        self.min_bits = config['quantization']['min_bits']
        self.max_bits = config['quantization']['max_bits']
        self.target_bits = config['quantization']['target_bits']
        
        # Initialize model
        self.model = self._init_model()
        
        # Initialize teacher model for distillation
        self.teacher_model = self._init_teacher_model()
        
        # Initialize datasets
        self.train_dataset, self.val_dataset = self._init_datasets()
        
        # Compute complexity scores
        self.complexity_scores = self._compute_complexity_scores()
        
        # Initialize curriculum
        self.curriculum = CurriculumScheduler(
            warmup_epochs=config['curriculum']['warmup_epochs'],
            total_epochs=self.epochs,
            initial_complexity=config['curriculum']['initial_complexity'],
            initial_temperature=config['curriculum']['initial_temperature'],
            curriculum_type=config['curriculum']['type']
        )
        
        # Initialize optimizer and scheduler
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.best_map = 0
        self.training_history = {
            'loss': [],
            'mAP': [],
            'avg_bits': [],
            'temperature': [],
            'learning_rate': []
        }
    
    def _init_model(self) -> nn.Module:
        """Initialize MCAQ-YOLO model."""
        model = MCAQYOLO(
            model_name=self.config['model']['name'],
            pretrained=self.config['model']['pretrained'],
            min_bits=self.min_bits,
            max_bits=self.max_bits,
            target_bits=self.target_bits,
            device=self.device
        )
        return model.to(self.device)
    
    def _init_teacher_model(self) -> nn.Module:
        """Initialize teacher model for knowledge distillation."""
        if self.config.get('distillation', {}).get('enabled', True):
            teacher = YOLO(self.config['model']['teacher_path'])
            teacher.model.eval()
            for param in teacher.model.parameters():
                param.requires_grad = False
            return teacher.model.to(self.device)
        return None
    
    # train.py의 _init_datasets 메서드 수정

def _init_datasets(self):
    """Initialize training and validation datasets."""
    # Use YOLOv8's dataset format
    from ultralytics.data import build_dataloader, build_yolo_dataset
    
    # For YOLOv8 compatibility
    train_dataset = build_yolo_dataset(
        cfg=self.config,
        img_path=self.config['data']['train_path'],
        batch=self.batch_size,
        augment=True,
        cache=False,
        data=self.config['data']
    )
    
    val_dataset = build_yolo_dataset(
        cfg=self.config,
        img_path=self.config['data']['val_path'],
        batch=self.batch_size,
        augment=False,
        cache=False,
        data=self.config['data']
    )
    
    return train_dataset, val_dataset

def train_epoch(self, epoch: int) -> dict:
    """Train for one epoch."""
    self.model.train()
    
    # Get curriculum parameters
    curr_params = self.curriculum.get_current_params()
    complexity_threshold = curr_params['complexity_threshold']
    temperature = curr_params['temperature']
    loss_weights = self.curriculum.get_loss_weights(epoch)
    
    # Create dataloader
    from ultralytics.data import build_dataloader
    
    dataloader = build_dataloader(
        dataset=self.train_dataset,
        batch_size=self.batch_size,
        workers=self.config['data']['num_workers'],
        shuffle=True
    )[0]  # build_dataloader returns tuple
    
    # ... rest of the method remains the same
    
    def _compute_complexity_scores(self) -> np.ndarray:
        """Compute or load complexity scores for curriculum learning."""
        cache_path = self.output_dir / 'complexity_scores.npy'
        
        if cache_path.exists():
            print(f"Loading complexity scores from {cache_path}")
            return np.load(cache_path)
        
        print("Computing complexity scores for curriculum learning...")
        scores = compute_dataset_complexity(
            self.train_dataset,
            model=self.model,
            batch_size=self.batch_size,
            device=str(self.device),
            save_path=str(cache_path)
        )
        
        return scores
    
    def _init_optimizer(self) -> torch.optim.Optimizer:
        """Initialize optimizer."""
        param_groups = [
            {'params': self.model.model.parameters(), 'lr': self.learning_rate},
            {'params': self.model.complexity_analyzer.parameters(), 'lr': self.learning_rate * 0.1},
            {'params': self.model.bit_mapper.parameters(), 'lr': self.learning_rate * 0.1}
        ]
        
        optimizer_type = self.config['optimizer']['type']
        
        if optimizer_type == 'adamw':
            optimizer = optim.AdamW(
                param_groups,
                weight_decay=self.config['optimizer']['weight_decay'],
                betas=(0.9, 0.999)
            )
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(
                param_groups,
                momentum=0.9,
                weight_decay=self.config['optimizer']['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        return optimizer
    
    def _init_scheduler(self):
        """Initialize learning rate scheduler."""
        scheduler_type = self.config['scheduler']['type']
        
        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=self.learning_rate * 0.01
            )
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config['scheduler']['step_size'],
                gamma=self.config['scheduler']['gamma']
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self, epoch: int) -> dict:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of epoch metrics
        """
        self.model.train()
        
        # Get curriculum parameters
        curr_params = self.curriculum.get_current_params()
        complexity_threshold = curr_params['complexity_threshold']
        temperature = curr_params['temperature']
        loss_weights = self.curriculum.get_loss_weights(epoch)
        
        # Create dataloader with curriculum sampling
        if self.config['curriculum']['enabled']:
            sampler = ComplexityBasedSampler(
                self.train_dataset,
                self.complexity_scores,
                self.batch_size
            )
            subset = sampler.get_curriculum_subset(complexity_threshold)
            dataloader = DataLoader(
                subset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.config['data']['num_workers'],
                pin_memory=True
            )
        else:
            dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.config['data']['num_workers'],
                pin_memory=True
            )
        
        # Training metrics
        epoch_losses = []
        epoch_metrics = {
            'loss_det': [],
            'loss_bit': [],
            'loss_smooth': [],
            'loss_kd': [],
            'avg_bits': []
        }
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{self.epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            # Get images and targets
            images = batch['img'].to(self.device)
            targets = batch['labels'].to(self.device)
            
            # Teacher forward pass
            teacher_outputs = None
            if self.teacher_model is not None:
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(images)
            
            # Student forward pass
            outputs, aux_info = self.model(images, temperature)
            
            # Compute loss
            loss, loss_dict = self.model.loss_fn(
                outputs,
                targets,
                aux_info,
                teacher_outputs,
                self.model.bit_mapper,
                loss_weights
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config['training']['grad_clip']
            )
            
            self.optimizer.step()
            
            # Update metrics
            epoch_losses.append(loss.item())
            for key in epoch_metrics:
                if key in loss_dict:
                    epoch_metrics[key].append(loss_dict[key].item())
                elif key == 'avg_bits' and 'avg_bits' in aux_info:
                    epoch_metrics[key].append(aux_info['avg_bits'].item())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'bits': f'{aux_info["avg_bits"].item():.2f}',
                'temp': f'{temperature:.2f}'
            })
            
            # Log to tensorboard
            global_step = epoch * len(dataloader) + batch_idx
            self.writer.add_scalar('train/loss', loss.item(), global_step)
            self.writer.add_scalar('train/avg_bits', aux_info['avg_bits'].item(), global_step)
        
        # Compute epoch averages
        metrics = {
            'loss': np.mean(epoch_losses),
            **{key: np.mean(values) for key, values in epoch_metrics.items() if values}
        }
        
        return metrics
    
    def validate(self) -> dict:
        """
        Run validation.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True
        )
        
        # Run evaluation
        metrics = evaluate_mcaq_yolo(
            self.model,
            dataloader,
            device=str(self.device),
            compute_complexity=True
        )
        
        self.model.train()
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config,
            'training_history': self.training_history
        }
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / 'latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if metrics.get('mAP@0.5', 0) > self.best_map:
            self.best_map = metrics['mAP@0.5']
            best_path = self.output_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"New best model saved with mAP@0.5: {self.best_map:.4f}")
        
        # Save periodic checkpoint
        if epoch % self.save_interval == 0:
            periodic_path = self.output_dir / f'epoch_{epoch}.pth'
            torch.save(checkpoint, periodic_path)
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.epochs} epochs...")
        print(f"Output directory: {self.output_dir}")
        
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            
            # Update curriculum
            self.curriculum.step()
            
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = {}
            if epoch % self.eval_interval == 0:
                print("\nRunning validation...")
                val_metrics = self.validate()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Log metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Avg Bits: {train_metrics.get('avg_bits', 0):.2f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            if val_metrics:
                print(f"  Val mAP@0.5: {val_metrics.get('mAP@0.5', 0):.4f}")
                print(f"  Val mAP@0.5:0.95: {val_metrics.get('mAP@0.5:0.95', 0):.4f}")
            
            # Update training history
            self.training_history['loss'].append(train_metrics['loss'])
            self.training_history['avg_bits'].append(train_metrics.get('avg_bits', 0))
            self.training_history['learning_rate'].append(current_lr)
            
            if val_metrics:
                self.training_history['mAP'].append(val_metrics.get('mAP@0.5', 0))
            
            # Save checkpoint
            self.save_checkpoint(epoch, {**train_metrics, **val_metrics})
            
            # Log to tensorboard
            self.writer.add_scalar('epoch/train_loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('epoch/avg_bits', train_metrics.get('avg_bits', 0), epoch)
            self.writer.add_scalar('epoch/learning_rate', current_lr, epoch)
            
            if val_metrics:
                self.writer.add_scalar('epoch/val_map50', val_metrics.get('mAP@0.5', 0), epoch)
        
        print("\nTraining completed!")
        
        # Save final model
        final_path = self.output_dir / 'final.pth'
        torch.save(self.model.state_dict(), final_path)
        
        # Plot training curves
        fig = plot_training_curves(self.training_history)
        fig.savefig(self.output_dir / 'training_curves.png')
        
        # Save training history
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=4)
        
        self.writer.close()
        
        return self.model


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train MCAQ-YOLO')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory for training results')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    config['device'] = args.device
    config['output_dir'] = args.output_dir
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config['output_dir'] = Path(config['output_dir']) / f'mcaq_yolo_{timestamp}'
    config['output_dir'].mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(config['output_dir'] / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if trainer.scheduler and checkpoint['scheduler_state_dict']:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.current_epoch = checkpoint['epoch']
        trainer.training_history = checkpoint.get('training_history', {})
        print(f"Resumed from checkpoint: {args.resume}")
        print(f"Starting from epoch {trainer.current_epoch}")
    
    # Train model
    model = trainer.train()
    
    print("Training completed successfully!")
    print(f"Results saved to: {config['output_dir']}")


if __name__ == '__main__':
    main()