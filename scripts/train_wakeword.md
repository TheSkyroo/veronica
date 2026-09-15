# Training the "Hey Veronica" wake word

Veronica ships with openwakeword's pretrained `hey_jarvis` model by default, but the
target wake word is a custom-trained `hey_veronica` model.

1. Open openwakeword's custom-model training notebook in Colab:
   https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb
2. Set the target phrase to `hey veronica`, keep the notebook's other defaults, and run
   all cells.
3. Export the trained model as `hey_veronica.onnx` and download it.
4. Copy it into Veronica's models directory:

   ```bash
   cp ~/Downloads/hey_veronica.onnx ~/.veronica/models/hey_veronica.onnx
   ```

Until this file exists, Veronica falls back to the pretrained `hey_jarvis` model.
