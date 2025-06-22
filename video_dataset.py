import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class VideoDataset(Dataset):
    def __init__(self, folder_path, num_videos=1000, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        video_files = os.listdir(folder_path)
        self.video_files = video_files[:num_videos]

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = os.path.join(self.folder_path, self.video_files[idx])
        video = self.load_video(video_path)
        video = video.float() / 255.0
        
        if self.transform:
            video = self.transform(video)
        
        return video

    def load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        frames = np.stack(frames, axis=0)
        return torch.from_numpy(frames)

