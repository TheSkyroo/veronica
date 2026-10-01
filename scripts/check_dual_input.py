"""Live check: can two RawInputStreams read the mic concurrently on this Mac?"""
import threading
import time

import sounddevice as sd


def reader(tag, n=1280, secs=2.0):
    with sd.RawInputStream(samplerate=16000, channels=1, dtype="int16", blocksize=n) as s:
        t0 = time.time()
        frames = 0
        while time.time() - t0 < secs:
            s.read(n)
            frames += 1
        print(f"{tag}: {frames} frames ok")


a = threading.Thread(target=reader, args=("A",))
b = threading.Thread(target=reader, args=("B",))
a.start(); time.sleep(0.2); b.start(); a.join(); b.join()
print("DUAL INPUT OK")
