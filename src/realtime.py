"""
realtime.py
Real-time microphone inference demo (optional).
"""


import numpy as np
import sounddevice as sd
from model import DigitClassifier
from features import extract_mfcc
import time
import threading
import keyboard  # pip install keyboard

SR = 8000
DURATION = 3.0  # seconds max
MODEL_PATH = "digit_classifier.joblib"

def record_with_space(max_duration=3.0):
	print("Press SPACE to start recording. Recording will stop after 3 seconds or when you press SPACE again.")
	while True:
		if keyboard.is_pressed('space'):
			print("Recording...")
			recording = np.zeros((int(max_duration * SR), 1), dtype='float32')
			rec_thread = threading.Thread(target=sd.rec, args=(int(max_duration * SR),), kwargs={'samplerate': SR, 'channels': 1, 'dtype': 'float32', 'out': recording})
			rec_thread.start()
			start_time = time.time()
			while time.time() - start_time < max_duration:
				if keyboard.is_pressed('space') and time.time() - start_time > 0.5:
					break
				time.sleep(0.05)
			sd.stop()
			rec_thread.join()
			print("Recording stopped.")
			return recording.flatten()
		time.sleep(0.05)

def main():
	clf = DigitClassifier()
	clf.load(MODEL_PATH)
	y = record_with_space(DURATION)
	features = extract_mfcc(y, sr=SR, n_fft=512)
	pred = clf.predict([features])[0]
	print(f"Recognized digit: {pred}")

if __name__ == "__main__":
	main()
