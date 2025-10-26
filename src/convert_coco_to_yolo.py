import json
from pathlib import Path


def convert_coco_to_yolo(coco_json_path: str, output_dir: str):
    """
    Convert COCO format annotations to YOLO format.
    
    Args:
        coco_json_path: Path to the COCO JSON file
        output_dir: Directory to save YOLO format txt files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load COCO annotations
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create image_id to filename mapping
    images = {img['id']: img for img in coco_data['images']}
    
    # Create category_id to class_id mapping (for YOLO, we need 0-indexed classes)
    categories = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}
    
    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    # Convert each image's annotations to YOLO format
    for image_id, image_info in images.items():
        img_width = image_info['width']
        img_height = image_info['height']
        img_filename = Path(image_info['file_name']).stem
        
        # Create YOLO format txt file
        yolo_txt_path = output_path / f"{img_filename}.txt"
        
        if image_id not in annotations_by_image:
            # No annotations for this image - create empty file
            yolo_txt_path.touch()
            continue
        
        with open(yolo_txt_path, 'w') as f:
            for ann in annotations_by_image[image_id]:
                # COCO bbox format: [x, y, width, height] (top-left corner)
                x, y, w, h = ann['bbox']
                
                # Convert to YOLO format: [class_id, x_center, y_center, width, height] (normalized)
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                norm_width = w / img_width
                norm_height = h / img_height
                
                class_id = categories[ann['category_id']]
                
                # Write YOLO format line
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
    
    print(f"✅ Converted {len(images)} images from COCO to YOLO format")
    print(f"   Output directory: {output_path}")


if __name__ == "__main__":
    import sys
    
    # Convert all splits
    dataset_dir = Path("barcode_dataset")
    
    for split in ['train', 'val', 'test']:
        coco_json = dataset_dir / split / "_annotations.coco.json"
        if coco_json.exists():
            print(f"\n📝 Converting {split} split...")
            convert_coco_to_yolo(
                str(coco_json),
                str(dataset_dir / split / "labels")
            )
        else:
            print(f"⚠️  {coco_json} not found, skipping {split} split")
    
    print("\n✨ Conversion complete!")
