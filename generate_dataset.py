import torch
import torchvision
import os
import json
from torch.utils.data import Dataset


def string_to_ascii_tensor(input_string, max_length=1024):
    """
    [GENERATED WITH CHAT GPT]
    Convertit une chaîne de caractères en un tenseur ASCII de taille fixe.
    La chaîne est tronquée ou paddée avec des "2" pour correspondre à max_length.
    """
    ascii_values = [ord(c) for c in input_string[:max_length]]
    padded_ascii_values = ascii_values + [2] * (max_length - len(ascii_values))
    return torch.tensor(padded_ascii_values, dtype=torch.uint8)  # Retourne un tenseur


class CustomVideoDataset(Dataset):
    def __init__(self, video_dir, annotations_file, max_frames=64):
        self.video_dir = video_dir
        self.annotations = self.load_annotations(annotations_file)
        self.video_ids = self.get_video_ids()
        self.max_frames = max_frames

    def load_annotations(self, annotations_file):
        with open(annotations_file, "r") as f:
            return json.load(f)

    def get_video_ids(self):
        video_ids = []
        for annotation in self.annotations:
            video_id = annotation["video_id"]
            if video_id not in video_ids:
                video_ids.append(video_id)
        return video_ids

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        video_id = annotation["video_id"]
        video_path = os.path.join(self.video_dir, f"video{video_id}.mp4")

        video, audio, info = torchvision.io.read_video(video_path)
        audio_fps = info.get("audio_fps", None)

        # Limiting the number of frames
        total_frames = video.shape[0]
        frame_indices = torch.linspace(0, total_frames - 1, self.max_frames).long()
        video = video[frame_indices]

        # Setting constant audio samples
        audio_samples = 10000
        audio = audio[:, :audio_samples]

        video_input = video.permute(3, 0, 1, 2)  # Rearrange to [C, T, H, W]

        if audio_fps is None:
            audio_input = torch.zeros((1, 2, audio_samples), dtype=torch.float32)
        else:
            audio_input = audio

        # Normalisation of the video tensor between -1 and 1
        video_input = (video_input.float() / 255.0) * 2.0 - 1.0

        # Conversion of target and context_input into tensors ASCII
        target = string_to_ascii_tensor(annotation["answer"])
        context_input = string_to_ascii_tensor(annotation["question"])

        # Add the batch dimension
        video_input = video_input.unsqueeze(0)
        audio_input = audio_input.unsqueeze(0)
        context_input = context_input.unsqueeze(0)
        target = target.unsqueeze(0)

        return audio_input, video_input, context_input, target
