"""
Inference script for MCAQ-YOLO
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import cv2
import argparse
import time
import json
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from models.mcaq_yolo import MCAQYOLO
from utils.visualization import (
    visualize_complexity_map,
    visualize_bit_allocation
)


class Predictor:
    """
    MCAQ-YOLO inference class
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: str = 'cuda',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_det: int = 1000
    ):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to model checkpoint
            config_path: Path to configuration file
            device: Device to use
            conf_threshold: Confidence threshold
            iou_threshold: IOU threshold for NMS
            max_det: Maximum detections
        """
        self.device = torch.device(device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        
        # Load model
        self.model = self._load_model(model_path, config_path)
        self.model.eval()
        
        # Class names (placeholder - should load from config)
        self.class_names = self._load_class_names(config_path)
        
        # Warmup
        self._warmup()
    
    def _load_model(self, model_path: str, config_path: Optional[str]) -> nn.Module:
        """Load MCAQ-YOLO model."""
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get configuration
        if config_path:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = checkpoint.get('config', {})
        
        # Initialize model
        model = MCAQYOLO(
            model_name=config.get('model', {}).get('name', 'yolov8n'),
            pretrained=False,
            min_bits=config.get('quantization', {}).get('min_bits', 2),
            max_bits=config.get('quantization', {}).get('max_bits', 8),
            target_bits=config.get('quantization', {}).get('target_bits', 4),
            device=str(self.device)
        )
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model.to(self.device)
    
    def _load_class_names(self, config_path: Optional[str]) -> List[str]:
        """Load class names from configuration."""
        if config_path and Path(config_path).exists():
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('class_names', [])
        
        # Default COCO classes (subset)
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow'
        ]
    
    def _warmup(self):
        """Warmup model for consistent inference time."""
        dummy_input = torch.randn(1, 3, 640, 640).to(self.device)
        for _ in range(3):
            with torch.no_grad():
                _ = self.model(dummy_input, temperature=1.0, return_aux=False)
    
    def preprocess(self, image: np.ndarray, img_size: int = 640) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image (H, W, C)
            img_size: Target image size
            
        Returns:
            Preprocessed tensor
        """
        # Resize image
        h, w = image.shape[:2]
        scale = img_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # Pad to square
        pad_h = (img_size - new_h) // 2
        pad_w = (img_size - new_w) // 2
        
        padded = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        
        # Convert to tensor
        tensor = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        
        return tensor, (scale, pad_w, pad_h)
    
    def postprocess(
        self,
        predictions: torch.Tensor,
        scale_info: Tuple[float, int, int],
        original_shape: Tuple[int, int]
    ) -> List[Dict]:
        """
        Postprocess model predictions.
        
        Args:
            predictions: Raw model outputs
            scale_info: Scaling information (scale, pad_w, pad_h)
            original_shape: Original image shape (H, W)
            
        Returns:
            List of detections
        """
        # Apply NMS
        detections = self._apply_nms(predictions)
        
        # Scale back to original coordinates
        scale, pad_w, pad_h = scale_info
        h_orig, w_orig = original_shape
        
        results = []
        for det in detections:
            if len(det) > 0:
                # Scale coordinates
                det[:, 0] = (det[:, 0] - pad_w) / scale
                det[:, 2] = (det[:, 2] - pad_w) / scale
                det[:, 1] = (det[:, 1] - pad_h) / scale
                det[:, 3] = (det[:, 3] - pad_h) / scale
                
                # Clip to image bounds
                det[:, 0] = det[:, 0].clamp(0, w_orig)
                det[:, 2] = det[:, 2].clamp(0, w_orig)
                det[:, 1] = det[:, 1].clamp(0, h_orig)
                det[:, 3] = det[:, 3].clamp(0, h_orig)
                
                for d in det:
                    results.append({
                        'bbox': d[:4].cpu().numpy().tolist(),
                        'confidence': d[4].cpu().item(),
                        'class_id': int(d[5].cpu().item()),
                        'class_name': self.class_names[int(d[5])] if int(d[5]) < len(self.class_names) else 'unknown'
                    })
        
        return results
    
    def _apply_nms(self, predictions: torch.Tensor) -> List[torch.Tensor]:
        """Apply Non-Maximum Suppression."""
        # Simplified NMS implementation
        # In practice, would use torchvision.ops.nms
        
        output = []
        for pred in predictions:
            # Filter by confidence
            mask = pred[:, 4] > self.conf_threshold
            pred = pred[mask]
            
            if len(pred) == 0:
                output.append(torch.empty(0, 6))
                continue
            
            # Get boxes, scores, and classes
            boxes = pred[:, :4]
            scores = pred[:, 4]
            classes = pred[:, 5]
            
            # Simple NMS per class
            keep = []
            for cls_id in torch.unique(classes):
                cls_mask = classes == cls_id
                cls_boxes = boxes[cls_mask]
                cls_scores = scores[cls_mask]
                
                # Sort by score
                order = cls_scores.argsort(descending=True)
                cls_boxes = cls_boxes[order]
                cls_scores = cls_scores[order]
                
                # Apply NMS
                keep_cls = []
                while len(cls_boxes) > 0:
                    keep_cls.append(order[0])
                    
                    if len(cls_boxes) == 1:
                        break
                    
                    # Compute IoU
                    ious = self._compute_iou(cls_boxes[0:1], cls_boxes[1:])
                    
                    # Keep boxes with low IoU
                    mask = ious < self.iou_threshold
                    cls_boxes = cls_boxes[1:][mask]
                    order = order[1:][mask]
                
                keep.extend([i for i in range(len(pred)) if cls_mask[i] and i in keep_cls])
            
            # Limit detections
            keep = keep[:self.max_det]
            output.append(pred[keep])
        
        return output
    
    def _compute_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """Compute IoU between boxes."""
        # Get intersection
        inter_x1 = torch.max(box1[:, 0], box2[:, 0])
        inter_y1 = torch.max(box1[:, 1], box2[:, 1])
        inter_x2 = torch.min(box1[:, 2], box2[:, 2])
        inter_y2 = torch.min(box1[:, 3], box2[:, 3])
        
        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        
        # Get union
        box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / (union_area + 1e-7)
    
    def predict(
        self,
        image: np.ndarray,
        visualize: bool = False,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Run inference on image.
        
        Args:
            image: Input image
            visualize: Whether to visualize results
            save_path: Path to save visualization
            
        Returns:
            Dictionary with predictions and metrics
        """
        # Preprocess
        tensor, scale_info = self.preprocess(image)
        
        # Inference
        start_time = time.time()
        
        with torch.no_grad():
            outputs, aux_info = self.model(tensor, temperature=1.0, return_aux=True)
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Postprocess
        detections = self.postprocess(outputs, scale_info, image.shape[:2])
        
        # Prepare results
        results = {
            'detections': detections,
            'num_detections': len(detections),
            'inference_time_ms': inference_time,
            'avg_bits': aux_info['avg_bits'].item(),
            'complexity_map': aux_info['complexity_map'].cpu().numpy(),
            'bit_map': aux_info['bit_map'].cpu().numpy()
        }
        
        # Visualize if requested
        if visualize:
            vis_image = self.visualize_predictions(image, detections)
            results['visualization'] = vis_image
            
            # Visualize complexity and bit allocation
            complexity_fig = visualize_complexity_map(
                image,
                aux_info['complexity_map'].squeeze(),
                save_path=save_path.replace('.jpg', '_complexity.jpg') if save_path else None
            )
            
            bit_fig = visualize_bit_allocation(
                image,
                aux_info['bit_map'].squeeze(),
                save_path=save_path.replace('.jpg', '_bits.jpg') if save_path else None
            )
            
            results['complexity_visualization'] = complexity_fig
            results['bit_visualization'] = bit_fig
        
        return results
    
    def visualize_predictions(
        self,
        image: np.ndarray,
        detections: List[Dict]
    ) -> np.ndarray:
        """
        Visualize predictions on image.
        
        Args:
            image: Original image
            detections: List of detections
            
        Returns:
            Visualized image
        """
        vis_image = image.copy()
        
        # Define colors for different classes
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 255, 0), (255, 128, 0), (128, 0, 255)
        ]
        
        for det in detections:
            bbox = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']
            class_id = det['class_id']
            
            # Draw bounding box
            color = colors[class_id % len(colors)]
            cv2.rectangle(
                vis_image,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                color,
                2
            )
            
            # Draw label
            label = f'{class_name}: {conf:.2f}'
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            cv2.rectangle(
                vis_image,
                (int(bbox[0]), int(bbox[1]) - label_size[1] - 4),
                (int(bbox[0]) + label_size[0], int(bbox[1])),
                color,
                -1
            )
            
            cv2.putText(
                vis_image,
                label,
                (int(bbox[0]), int(bbox[1]) - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return vis_image
    
    def predict_batch(
        self,
        images: List[np.ndarray],
        batch_size: int = 8
    ) -> List[Dict]:
        """
        Run inference on multiple images.
        
        Args:
            images: List of images
            batch_size: Batch size for inference
            
        Returns:
            List of results
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # Preprocess batch
            tensors = []
            scale_infos = []
            
            for img in batch_images:
                tensor, scale_info = self.preprocess(img)
                tensors.append(tensor)
                scale_infos.append(scale_info)
            
            # Stack tensors
            batch_tensor = torch.cat(tensors, dim=0)
            
            # Inference
            with torch.no_grad():
                outputs, aux_info = self.model(batch_tensor, temperature=1.0, return_aux=True)
            
            # Postprocess each image
            for j, (img, scale_info) in enumerate(zip(batch_images, scale_infos)):
                detections = self.postprocess(
                    outputs[j:j+1],
                    scale_info,
                    img.shape[:2]
                )
                
                results.append({
                    'detections': detections,
                    'avg_bits': aux_info['avg_bits'][j].item() if aux_info['avg_bits'].dim() > 0 else aux_info['avg_bits'].item()
                })
        
        return results


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='MCAQ-YOLO Inference')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to image or directory')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou-threshold', type=float, default=0.45,
                       help='IOU threshold for NMS')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize results')
    parser.add_argument('--save-dir', type=str, default='outputs/predictions',
                       help='Directory to save results')
    args = parser.parse_args()
    
    # Create output directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize predictor
    predictor = Predictor(
        model_path=args.model,
        config_path=args.config,
        device=args.device,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold
    )
    
    # Process source
    source_path = Path(args.source)
    
    if source_path.is_file():
        # Single image
        image = cv2.imread(str(source_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Predict
        results = predictor.predict(
            image,
            visualize=args.visualize,
            save_path=str(save_dir / source_path.name) if args.visualize else None
        )
        
        # Print results
        print(f"\nResults for {source_path.name}:")
        print(f"  Detections: {results['num_detections']}")
        print(f"  Inference time: {results['inference_time_ms']:.2f} ms")
        print(f"  Average bits: {results['avg_bits']:.2f}")
        
        # Save results
        with open(save_dir / f"{source_path.stem}_results.json", 'w') as f:
            json.dump({
                'detections': results['detections'],
                'num_detections': results['num_detections'],
                'inference_time_ms': results['inference_time_ms'],
                'avg_bits': results['avg_bits']
            }, f, indent=4)
    
    elif source_path.is_dir():
        # Directory of images
        image_paths = list(source_path.glob('*.jpg'))
        image_paths.extend(list(source_path.glob('*.png')))
        
        print(f"Processing {len(image_paths)} images...")
        
        total_time = 0
        total_bits = 0
        total_detections = 0
        
        for img_path in image_paths:
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Predict
            results = predictor.predict(
                image,
                visualize=args.visualize,
                save_path=str(save_dir / img_path.name) if args.visualize else None
            )
            
            total_time += results['inference_time_ms']
            total_bits += results['avg_bits']
            total_detections += results['num_detections']
            
            print(f"  {img_path.name}: {results['num_detections']} detections, "
                  f"{results['inference_time_ms']:.2f} ms, {results['avg_bits']:.2f} bits")
        
        # Print summary
        print(f"\nSummary:")
        print(f"  Total images: {len(image_paths)}")
        print(f"  Total detections: {total_detections}")
        print(f"  Average inference time: {total_time / len(image_paths):.2f} ms")
        print(f"  Average bits: {total_bits / len(image_paths):.2f}")
    
    else:
        print(f"Error: {source_path} is neither a file nor a directory")
        return
    
    print(f"\nResults saved to: {save_dir}")


if __name__ == '__main__':
    main()