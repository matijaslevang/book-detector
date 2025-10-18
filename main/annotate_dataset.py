import os
import pandas as pd
from pathlib import Path
import easyocr
import difflib
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true")

def annotate_dataset():
    project_root = Path(__file__).parent.parent
    adjusted_csv_path = project_root / "dataset" / 'final_dataset3.csv'

    # Load the CSV
    df = pd.read_csv(adjusted_csv_path)
    
    # Initialize EasyOCR reader for English
    reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if CUDA is available
    
    # Add new columns if not exist
    if 'bounding_boxes' not in df.columns:
        df['bounding_boxes'] = None
    if 'text' not in df.columns:
        df['text'] = None
    
    # Helper function to calculate if two bounding boxes are close enough to merge
    def are_bboxes_close(bbox1, bbox2, threshold=20):
        """
        Check if two bouncing boxes are within a certain distance threshold.
        bbox is a list of [x1, y1], [x2, y2], [x3, y3], [x4, y4].
        Returns True if the boxes are close enough to be merged.
        """
        x1_1, y1_1 = bbox1[0]
        x2_1, y2_1 = bbox1[2]
        x1_2, y1_2 = bbox2[0]
        x2_2, y2_2 = bbox2[2]
        
        # Check if the bounding boxes' x and y ranges overlap
        return (abs(x2_1 - x1_2) < threshold or abs(x2_2 - x1_1) < threshold) and \
               (abs(y2_1 - y1_2) < threshold or abs(y2_2 - y1_1) < threshold)

    # Process each row
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Annotating images"):
        img_path = str(row['img_paths'])
        
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
        
        # Run EasyOCR on the image
        results = reader.readtext(img_path, paragraph=True, min_size=20)
        
        if not results:
            print(f"No text detected in {img_path}")
            continue
        
        # Find the detection with text most similar to the book's name
        book_name = row['name'].strip().lower()
        best_match = None
        best_ratio = 0.0
        bounding_boxes = []
        texts = []
        
        # Set a stricter threshold for matching
        threshold = 0.4
        
        for bbox, text in results:
            detected_text = text.strip().lower()
            ratio = difflib.SequenceMatcher(None, book_name, detected_text).ratio()
            
            # Collect all bounding boxes with acceptable similarity
            if ratio >= threshold:
                bounding_boxes.append(bbox)
                texts.append(detected_text)
        
        # Merge nearby bounding boxes into one
        if bounding_boxes:
            merged_bbox = bounding_boxes[0]
            merged_text = texts[0]
            for i in range(1, len(bounding_boxes)):
                if are_bboxes_close(merged_bbox, bounding_boxes[i]):
                    # Expand the bounding box to include the new one
                    x1, y1 = min(merged_bbox[0][0], bounding_boxes[i][0][0]), min(merged_bbox[0][1], bounding_boxes[i][0][1])
                    x2, y2 = max(merged_bbox[2][0], bounding_boxes[i][2][0]), max(merged_bbox[2][1], bounding_boxes[i][2][1])
                    merged_bbox = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
                    merged_text += " " + texts[i]  # Merge the detected texts as well
                else:
                    # If the bounding boxes are not close, create a new merged entry
                    merged_bbox = bounding_boxes[i]
                    merged_text = texts[i]
                    
                # You can also add other conditions to filter text and bounding box combinations more carefully.
        
            bounding_boxes = [merged_bbox]
            texts = [merged_text]

        # Save the merged bounding box and text to CSV
        df.at[idx, 'bounding_boxes'] = json.dumps(bounding_boxes)
        df.at[idx, 'text'] = json.dumps(texts)
    
    # Save updated CSV
    df.to_csv(adjusted_csv_path, index=False)
    print(f"Updated CSV saved to {adjusted_csv_path}")


if __name__ == "__main__":
    annotate_dataset()