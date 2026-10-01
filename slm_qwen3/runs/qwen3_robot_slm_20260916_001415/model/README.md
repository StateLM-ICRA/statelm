# Robot SLM runtime

Install a compatible PyTorch build for your host, then `pip install -r requirements.txt`.
The default Qwen3 base model is public; an HF token is optional.
Run `python slm_runtime.py --bundle .` from this folder. For CPU add `--device cpu`.
Use `/reset` for a new trial and `/quit` to exit. The trial ends after one recovery attempt.
Edit inventory.json to match your verified inventory. Default demo drawers are fictional.
Base-model weights are not included. Download/cache the revision named in runtime_config.json.
No ASR, TTS driver, ROS node, or robot motion control is included. Connect finalized ASR text to RobotSession.
Inventory validation prevents unknown IDs/locations, but cannot prove the selected item matches user intent.
