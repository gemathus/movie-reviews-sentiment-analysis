import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os

# Constants
VOCAB_SIZE = 10000
MAX_LENGTH = 200

class IMDBDataset(Dataset):
    def __init__(self, reviews, sentiments, vectorizer=None, is_training=False):
        self.reviews = reviews
        if is_training:
            # Fit and transform during training
            self.features = vectorizer.fit_transform(reviews).toarray()
        else:
            # Only transform during validation
            self.features = vectorizer.transform(reviews).toarray()
        self.sentiments = sentiments
        
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        return (torch.FloatTensor(self.features[idx]), 
                torch.tensor(1 if self.sentiments[idx] == 'positive' else 0))

class MovieReviewNN(nn.Module):
    def __init__(self, vocab_size):
        super(MovieReviewNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(vocab_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

def main():
    # Load data
    print("Loading IMDB dataset...")
    df = pd.read_csv('/app/data/imdb_dataset.csv')
    
    # Create vectorizer
    vectorizer = CountVectorizer(max_features=VOCAB_SIZE, max_df=0.95, min_df=2)
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Create datasets
    print("Preparing datasets...")
    train_dataset = IMDBDataset(
        train_df['review'].values, 
        train_df['sentiment'].values,
        vectorizer=vectorizer,
        is_training=True
    )
    
    val_dataset = IMDBDataset(
        val_df['review'].values, 
        val_df['sentiment'].values,
        vectorizer=vectorizer,
        is_training=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MovieReviewNN(VOCAB_SIZE).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("Starting training...")
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, "
                      f"Loss: {loss.item():.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                val_loss += criterion(outputs, labels.float().unsqueeze(1)).item()
                predicted = (outputs > 0.5).int().squeeze()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f"Epoch {epoch + 1} - Avg Loss: {total_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Accuracy: {100 * correct/total:.2f}%")
    
    # Save the model and vectorizer
    print("Saving model and vectorizer...")
    os.makedirs("model-storage", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'vectorizer': vectorizer
    }, "./model-storage/movie_sentiment_model.pt")
    print("Model and vectorizer saved to ./model-storage/movie_sentiment_model.pt")

if __name__ == "__main__":
    main()

