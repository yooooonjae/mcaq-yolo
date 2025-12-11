#!/usr/bin/env python3
"""
Example training script for MCAQ-YOLO
"""

import sys
sys.path.append('..')

from mcaq_yolo.models.mcaq_yolo import MCAQYOLO
from mcaq_yolo.train import Trainer, load_config
import torch


def main():
    # Configuration
    config = {
        'model': {
            'name': 'yolov8n',
            'pretrained': True,
            'teacher_path': 'yolov8n.pt',
            'num_classes': 80
        },
        'data': {
            'train_path': 'path/to/train',
            'val_path': 'path/to/val',
            'img_size': 640,
            'num_workers': 4
        },
        'epochs': 100,
        'batch_size': 16,
        'learning_rate': 0.001,
        'quantization': {
            'min_bits': 2,
            'max_bits': 8,
            'target_bits': 4.0
        },
        'curriculum': {
            'enabled': True,
            'warmup_epochs': 10,
            'initial_complexity': 0.2,
            'initial_temperature': 10.0,
            'type': 'exponential'
        },
        'optimizer': {
            'type': 'adamw',
            'weight_decay': 0.05
        },
        'scheduler': {
            'type': 'cosine'
        },
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'output_dir': 'outputs/test_run'
    }
    
    # Create trainer
    trainer = Trainer(config)
    
    # Train model
    model = trainer.train()
    
    print("Training completed!")
    
    # Test inference
    test_image = torch.randn(1, 3, 640, 640).to(config['device'])
    with torch.no_grad():
        outputs, aux_info = model(test_image)
        print(f"Average bits used: {aux_info['avg_bits'].item():.2f}")


if __name__ == '__main__':
    main()