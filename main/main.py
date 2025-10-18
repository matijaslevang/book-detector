import os
import pandas as pd
from pathlib import Path
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings("ignore")

chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.-'&()"
char_to_idx = {char: idx + 1 for idx, char in enumerate(chars)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

class BookTitleDataset(Dataset):
    def __init__(self, csv_path, project_root, transform=None):
        self.df = pd.read_csv(csv_path)
        self.project_root = project_root
        self.transform = transform
        self.data = []
        self.labels = []
        self.bboxes = []

        for _, row in self.df.iterrows():
            img_path = str(project_root / row['img_paths'])
            if not os.path.exists(img_path):
                continue

            bbox_str = row.get('bounding_boxes', '')
            if not isinstance(bbox_str, str) or bbox_str.strip() == '':
                continue

            try:
                parsed = eval(bbox_str)
                if (isinstance(parsed, list) and len(parsed) == 1 and
                    isinstance(parsed[0], list) and len(parsed[0]) == 4 and
                    all(isinstance(p, list) and len(p) == 2 for p in parsed[0])):
                    bbox = parsed[0]
                else:
                    raise ValueError
            except Exception:
                continue

            self.data.append(img_path)
            self.labels.append(row['name'].strip())
            self.bboxes.append(bbox)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        text = self.labels[idx]
        bbox = self.bboxes[idx]

        img = cv2.imread(img_path)
        if img is None:
            return self.__getitem__(np.random.randint(0, self.__len__()))

        h, w = img.shape[:2]
        bbox = [[int(x), int(y)] for x, y in bbox]
        x1, y1 = max(0, min([p[0] for p in bbox])), max(0, min([p[1] for p in bbox]))
        x2, y2 = min(w, max([p[0] for p in bbox])), min(h, max([p[1] for p in bbox]))
        cropped_img = img[y1:y2, x1:x2]
        if cropped_img.size == 0:
            cropped_img = img

        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        target_height = 32
        aspect_ratio = cropped_img.shape[1] / cropped_img.shape[0]
        target_width = max(20, int(target_height * aspect_ratio))
        cropped_img = cv2.resize(cropped_img, (target_width, target_height))

        cropped_img = torch.from_numpy(cropped_img).float() / 255.0
        cropped_img = cropped_img.unsqueeze(0)

        if self.transform:
            cropped_img = self.transform(cropped_img)

        return cropped_img, text

class CRNN(nn.Module):
    def __init__(self, num_chars, hidden_size=256):
        super(CRNN, self).__init__()
        self.conv = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32->16, W->W/2
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16->8, W->W/4
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8->4, W->W/8
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(512),
            
            # Pool only height to 1, preserve width
            nn.MaxPool2d((2, 1), (2, 1)),  # 4->2, W same
            nn.MaxPool2d((2, 1), (2, 1)),  # 2->1, W same
        )
        self.rnn = nn.LSTM(512, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, num_chars)

    def forward(self, x):
        conv = self.conv(x)  # (batch, 512, 1, T)
        batch, channels, height, width = conv.shape
        conv = conv.squeeze(2)  # (batch, 512, T)
        conv = conv.permute(0, 2, 1)  # (batch, T, 512)
        rnn_out, _ = self.rnn(conv)
        output = self.linear(rnn_out)  # (batch, T, num_chars)
        return output

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

def train_model(dataset, num_epochs=10, batch_size=16, learning_rate=0.001):
    num_chars = len(chars) + 1
    transform = transforms.Compose([transforms.Normalize(mean=[0.5], std=[0.5])])
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=0, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(num_chars=num_chars).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        for images, texts in tqdm(data_loader, desc=f"Epoch {epoch+1}"):
            if images is None:
                continue
            
            images = images.to(device)
            if images.dim() == 3:
                images = images.unsqueeze(0)
            
            outputs = model(images)
            log_probs = outputs.log_softmax(2).permute(1, 0, 2)
            
            input_lengths = torch.full((images.size(0),), log_probs.size(0), dtype=torch.long)
            targets, target_lengths = [], []
            
            for text in texts:
                target = [char_to_idx.get(c.lower(), 0) for c in text if c.lower() in char_to_idx]
                targets.extend(target)
                target_lengths.append(len(target))
            
            if not target_lengths:  # Skip empty batch
                continue
                
            targets = torch.tensor(targets, dtype=torch.long).to(device)
            target_lengths = torch.tensor(target_lengths, dtype=torch.long)

            optimizer.zero_grad()
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

        if batch_count > 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss / batch_count:.4f}")

    torch.save(model.state_dict(), "crnn_text_recognition.pth")
    print("Training complete, model saved in crnn_text_recognition.pth")

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "dataset" / "mixed_dataset.csv"
    
    dataset = BookTitleDataset(csv_path, project_root)
    if len(dataset) == 0:
        print("No valid data found!")
    else:
        print(f"Training on {len(dataset)} samples")
        train_model(dataset, num_epochs=30, batch_size=16)