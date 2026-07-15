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

try:  # ultralytics >= 8.4 moved NMS out of ops
    from ultralytics.utils.nms import non_max_suppression
except ImportError:  # ultralytics 8.0-8.3
    from ultralytics.utils.ops import non_max_suppression

try:  # package context: python -m mcaq_yolo.inference / from mcaq_yolo.inference import Predictor
    from .models.mcaq_yolo import MCAQYOLO
    from .utils.visualization import (
        visualize_complexity_map,
        visualize_bit_allocation,
    )
except ImportError:  # legacy: executed as a bare script inside mcaq_yolo/
    from models.mcaq_yolo import MCAQYOLO
    from utils.visualization import (
        visualize_complexity_map,
        visualize_bit_allocation,
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
            grid_size=config.get('quantization', {}).get('grid_size', 8),
            bit_mapping=config.get('quantization', {}).get('bit_mapping', 'mlp'),
            device=str(self.device)
        )
        
        # Load weights — strict first (integrity), fall back to strict=False
        # for checkpoints from older code revisions (parameter names changed
        # across the paper-alignment refactor), with an explicit warning.
        state = checkpoint.get('model_state_dict', checkpoint)
        try:
            model.load_state_dict(state)
        except RuntimeError as e:
            print(f"[MCAQ][WARN] strict load failed ({str(e)[:200]}...)\n"
                  f"[MCAQ][WARN] retrying with strict=False — verify the "
                  f"checkpoint matches this code revision!")
            incompat = model.load_state_dict(state, strict=False)
            if incompat.missing_keys:
                print(f"[MCAQ][WARN] missing keys: {len(incompat.missing_keys)}")
            if incompat.unexpected_keys:
                print(f"[MCAQ][WARN] unexpected keys: {len(incompat.unexpected_keys)}")

        return model.to(self.device)
    
    def _load_class_names(self, config_path: Optional[str]) -> List[str]:
        """Load class names: config file > model's embedded names > empty."""
        if config_path and Path(config_path).exists():
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                names = config.get('class_names', [])
                if names:
                    return names

        # Fall back to the names embedded in the underlying YOLO model
        # (full COCO-80 for pretrained yolov8 weights)
        embedded = getattr(getattr(self.model, 'model', None), 'names', None)
        if isinstance(embedded, dict) and embedded:
            return [embedded[i] for i in sorted(embedded)]
        if isinstance(embedded, (list, tuple)) and embedded:
            return list(embedded)
        return []
    
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
    
    @staticmethod
    def _extract_prediction_tensor(predictions):
        """
        Pull the concatenated eval-mode prediction tensor (B, 4+nc, N) out of
        the raw model output. Ultralytics DetectionModel in eval returns
        (y_cat, raw_feature_list); training mode returns the raw list directly
        (not decodable here).
        """
        if torch.is_tensor(predictions):
            return predictions
        if isinstance(predictions, (list, tuple)) and len(predictions) > 0:
            if torch.is_tensor(predictions[0]):
                return predictions[0]
        raise TypeError(
            "Cannot extract decodable predictions — run the model in eval mode "
            f"(got {type(predictions)})"
        )

    def postprocess(
        self,
        predictions,
        scale_info: Tuple[float, int, int],
        original_shape: Tuple[int, int]
    ) -> List[Dict]:
        """
        Postprocess model predictions: official Ultralytics NMS (which decodes
        the (B, 4+nc, N) head output to xyxy+conf+cls), then map boxes from the
        letterboxed 640-space back to the original image.

        Args:
            predictions: Raw model outputs (eval mode)
            scale_info: Scaling information (scale, pad_w, pad_h)
            original_shape: Original image shape (H, W)

        Returns:
            List of detections
        """
        preds = self._extract_prediction_tensor(predictions)
        detections = non_max_suppression(
            preds,
            conf_thres=self.conf_threshold,
            iou_thres=self.iou_threshold,
            max_det=self.max_det,
        )  # list per image of (n, 6) [x1, y1, x2, y2, conf, cls]

        # Scale back to original coordinates
        scale, pad_w, pad_h = scale_info
        h_orig, w_orig = original_shape

        results = []
        for det in detections:
            if len(det) > 0:
                # Undo letterbox: subtract padding, divide by scale
                det[:, 0] = ((det[:, 0] - pad_w) / scale).clamp(0, w_orig)
                det[:, 2] = ((det[:, 2] - pad_w) / scale).clamp(0, w_orig)
                det[:, 1] = ((det[:, 1] - pad_h) / scale).clamp(0, h_orig)
                det[:, 3] = ((det[:, 3] - pad_h) / scale).clamp(0, h_orig)

                for d in det:
                    cls_id = int(d[5].item())
                    results.append({
                        'bbox': d[:4].cpu().numpy().tolist(),
                        'confidence': d[4].cpu().item(),
                        'class_id': cls_id,
                        'class_name': self.class_names[cls_id] if cls_id < len(self.class_names) else 'unknown'
                    })

        return results
    
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

        # aux maps are per-scale lists (backbone C3/C4/C5 quantization);
        # use the highest-resolution scale (P3) for reporting/visualization
        def _first_map(m):
            if isinstance(m, (list, tuple)):
                m = m[0] if m else torch.zeros(1, 8, 8)
            return m

        cmap = _first_map(aux_info['complexity_map'])
        bmap = _first_map(aux_info['bit_map'])

        # Prepare results
        results = {
            'detections': detections,
            'num_detections': len(detections),
            'inference_time_ms': inference_time,
            'avg_bits': aux_info['avg_bits'].item(),
            'complexity_map': cmap.detach().cpu().numpy(),
            'bit_map': bmap.detach().cpu().numpy()
        }
        
        # Visualize if requested
        if visualize:
            vis_image = self.visualize_predictions(image, detections)
            results['visualization'] = vis_image
            
            # Visualize complexity and bit allocation (P3 scale)
            complexity_fig = visualize_complexity_map(
                image,
                results['complexity_map'].squeeze(),
                save_path=save_path.replace('.jpg', '_complexity.jpg') if save_path else None
            )

            bit_fig = visualize_bit_allocation(
                image,
                results['bit_map'].squeeze(),
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

            # Decode + NMS once for the whole batch, then rescale per image
            preds = self._extract_prediction_tensor(outputs)
            batch_dets = non_max_suppression(
                preds,
                conf_thres=self.conf_threshold,
                iou_thres=self.iou_threshold,
                max_det=self.max_det,
            )

            avg_bits = aux_info['avg_bits'].item()  # spatial average (scalar)

            for j, (img, scale_info) in enumerate(zip(batch_images, scale_infos)):
                det = batch_dets[j]
                scale, pad_w, pad_h = scale_info
                h_orig, w_orig = img.shape[:2]

                detections = []
                if len(det) > 0:
                    det[:, 0] = ((det[:, 0] - pad_w) / scale).clamp(0, w_orig)
                    det[:, 2] = ((det[:, 2] - pad_w) / scale).clamp(0, w_orig)
                    det[:, 1] = ((det[:, 1] - pad_h) / scale).clamp(0, h_orig)
                    det[:, 3] = ((det[:, 3] - pad_h) / scale).clamp(0, h_orig)
                    for d in det:
                        cls_id = int(d[5].item())
                        detections.append({
                            'bbox': d[:4].cpu().numpy().tolist(),
                            'confidence': d[4].cpu().item(),
                            'class_id': cls_id,
                            'class_name': self.class_names[cls_id] if cls_id < len(self.class_names) else 'unknown'
                        })

                results.append({
                    'detections': detections,
                    'avg_bits': avg_bits
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