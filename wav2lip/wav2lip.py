import os
import subprocess
import platform
import numpy as np
import cv2
import torch
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from moviepy.editor import *
from . import audio
from .models import Wav2Lip

class FaceVideoMaker(object):
    def __init__(self, weights_file='wav2lip/weights/wav2lip_gan.pth', face_img='assets/face_200.png', audio_dir='temp', video_dir='temp', fps=15, device='cpu', azure_key=None, azure_endpoint=None):
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.device = device
        self.fps = fps
        self.frame = cv2.imread(face_img)
        self.img_size = 96
        self.mel_step_size = 16
        self.wav2lip_batch_size = 128

        # Initialize Azure Face client
        if azure_key is None or azure_endpoint is None:
            raise ValueError("Azure Face API key and endpoint are required")
        self.face_client = FaceClient(azure_endpoint, CognitiveServicesCredentials(azure_key))

        # Detect face and set coordinates
        self.detect_face()

        weights_path = os.path.join(os.getcwd(), weights_file)
        print('Running...')
        weights = torch.load(weights_path, map_location=torch.device(self.device))
        s = weights["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        model = Wav2Lip()
        model.load_state_dict(new_s)
        model = model.to(device)
        self.model = model.eval()

    def detect_face(self):
        # Save the frame as a temporary file
        temp_image_path = 'temp_face.jpg'
        cv2.imwrite(temp_image_path, self.frame)

        # Detect face using Azure Face API
        with open(temp_image_path, 'rb') as image_stream:
            detected_faces = self.face_client.face.detect_with_stream(
                image=image_stream,
                return_face_landmarks=True
            )

        # Remove temporary file
        os.remove(temp_image_path)

        if detected_faces:
            face = detected_faces[0]
            landmarks = face.face_landmarks

            # Get mouth coordinates
            mouth_left = landmarks.mouth_left
            mouth_right = landmarks.mouth_right
            upper_lip_top = landmarks.upper_lip_top
            under_lip_bottom = landmarks.under_lip_bottom

            # Set coordinates for the mouth region
            self.x1 = int(mouth_left.x)
            self.x2 = int(mouth_right.x)
            self.y1 = int(upper_lip_top.y)
            self.y2 = int(under_lip_bottom.y)

            # Add some padding
            padding = 10
            self.x1 = max(0, self.x1 - padding)
            self.y1 = max(0, self.y1 - padding)
            self.x2 = min(self.frame.shape[1], self.x2 + padding)
            self.y2 = min(self.frame.shape[0], self.y2 + padding)

            self.y1r = self.y1  # You might want to adjust this based on your specific needs
        else:
            raise Exception("No face detected in the image.")

        self.face = self.frame[self.y1:self.y2, self.x1:self.x2]
        self.face = cv2.resize(self.face, (self.img_size, self.img_size))

    def makeVideo(self, id):
        audio_path = os.path.join(os.getcwd(), self.audio_dir, f'{id}.wav')
        wav = audio.load_wav(audio_path, 16000)
        mel = audio.melspectrogram(wav)
        mel_chunks = []
        mel_idx_multiplier = 80./self.fps
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + self.mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - self.mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + self.mel_step_size])
            i += 1

        frame_h, frame_w = self.frame.shape[:-1]
        video_path = os.path.join(os.getcwd(), self.video_dir, f'{id}.avi')
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), self.fps, (frame_w, frame_h))
        for (img_batch, mel_batch) in self.datagen2(self.face, mel_chunks):
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)

            with torch.no_grad():
                pred = self.model(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

            for p in pred:
                f = self.frame.copy()
                p = cv2.resize(p.astype(np.uint8), (self.x2 - self.x1, self.y2 - self.y1))
                f[self.y1r:self.y2, self.x1:self.x2] = p[self.y1r-self.y1:]
                out.write(f)
        out.release()

        face_video_path = os.path.join(os.getcwd(), self.video_dir, f'{id}.mp4')

        audioclip = AudioFileClip(audio_path)
        videoclip = VideoFileClip(video_path)

        new_audioclip = CompositeAudioClip([audioclip])
        videoclip.audio = new_audioclip
        videoclip.write_videofile(face_video_path)

        os.remove(audio_path)
        os.remove(video_path)

    def datagen2(self, face, mels):
        img_batch, mel_batch = [], []

        for m in mels:
            img_batch.append(face.copy())
            mel_batch.append(m)

            if len(img_batch) >= self.wav2lip_batch_size:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, self.img_size//2:] = 0

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch
                img_batch, mel_batch = [], []

        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, self.img_size//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch

    def update_face_image(self, new_face_img):
        self.frame = cv2.imread(new_face_img)
        self.detect_face()