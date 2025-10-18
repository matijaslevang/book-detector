import os
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def preprocess_image(input_path, output_path, size=(512, 512)):
    try:
        img = cv2.imread(input_path)
        if img is None:
            print(f"Failed to read image: {input_path}")
            return False
        
        img_resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        cv2.imwrite(output_path, img_resized)
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def main():
    project_root = Path(__file__).parent.parent
    dataset_dir = project_root / "dataset" / "dataset"
    adjusted_dataset_dir = project_root / "dataset" / "adjusted_dataset"
    main_csv_path = project_root / "dataset" / "main_dataset.csv"
    adjusted_csv_path = project_root / "dataset" / "adjusted_dataset.csv"
    
    try:
        main_df = pd.read_csv(main_csv_path)
    except Exception as e:
        print(f"Error reading {main_csv_path}: {e}")
        return
    
    adjusted_df = pd.DataFrame(columns=["name", "author", "img_paths"])
    
    successful = 0
    total = len(main_df)
    
    for _, row in tqdm(main_df.iterrows(), total=total, desc="Preprocessing images"):
        relative_path = Path(row["img_paths"]).relative_to("dataset")
        input_path = str(dataset_dir / relative_path)
        output_path = str(adjusted_dataset_dir / relative_path)
        adjusted_img_path = str(Path("dataset/adjusted_dataset") / relative_path)
        
        if preprocess_image(input_path, output_path):
            adjusted_df = pd.concat([adjusted_df, pd.DataFrame([{
                "name": row["name"],
                "author": row["author"],
                "img_paths": adjusted_img_path
            }])], ignore_index=True)
            successful += 1
        else:
            print(f"Skipping {input_path} due to processing error")
    
    try:
        adjusted_df.to_csv(adjusted_csv_path, index=False)
        print(f"Successfully saved adjusted dataset CSV to {adjusted_csv_path}")
    except Exception as e:
        print(f"Error saving {adjusted_csv_path}: {e}")
    
    print(f"Processed {successful}/{total} images successfully")

if __name__ == "__main__":
    main()