import torch
from tqdm import tqdm
from main.main import CRNN, BookTitleDataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import torch.nn.functional as F

chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.-'&()"
char_to_idx = {char: idx + 1 for idx, char in enumerate(chars)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

def decode_prediction(logits):
    pred = torch.argmax(logits, dim=-1).cpu().numpy()[0]
    decoded = []
    prev = -1
    for p in pred:
        if p != prev and p != 0:
            decoded.append(idx_to_char.get(p, ''))
        prev = p
    return ''.join(decoded)

def evaluate_model(model, data_loader, device):
    model.eval()
    total, correct = 0, 0
    
    with torch.no_grad():
        for images, texts in tqdm(data_loader, desc="Evaluating"):
            if images is None:
                continue

            images = images.to(device)
            outputs = model(images)
            
            for i, text in enumerate(texts):
                pred = decode_prediction(outputs[i:i+1])
                gt = text.lower()
                
                if pred == gt:
                    correct += 1
                total += 1
                
                print(f"GT: '{gt}' | Pred: '{pred}' | Match: {pred == gt}")

    accuracy = correct / total if total > 0 else 0
    print(f"\n=== EVALUATION RESULTS ===")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None
    
    images, texts = zip(*batch)
    max_width = max(img.shape[2] for img in images)
    
    padded_images = []
    for img in images:
        pad_width = max_width - img.shape[2]
        if pad_width > 0:
            img = F.pad(img, (0, pad_width), value=0)
        padded_images.append(img)
    
    padded_images = torch.stack(padded_images, dim=0)
    return padded_images, texts

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "dataset" / "mixed_dataset.csv"
    saved_model_name = "crnn_text_recognition.pth"

    transform = transforms.Compose([transforms.Normalize(mean=[0.5], std=[0.5])])
    test_dataset = BookTitleDataset(csv_path, project_root, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(num_chars=len(chars) + 1).to(device)
    model.load_state_dict(torch.load(saved_model_name, map_location=device))
    evaluate_model(model, test_loader, device)