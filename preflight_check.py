import importlib

modules = [
    "torch", "torchaudio", "torchmetrics", "torchinfo", "torch_audiomentations",
    "acoustics", "audiomentations", "librosa", "onnx", "onnxruntime", 
    "onnxsim", "onnx2tf", "speechbrain", "datasets", "nlpaug", 
    "pronouncing", "mutagen", "scipy", "sklearn", "pandas", 
    "yaml", "sentencepiece", "tqdm", "julius", "diffq"
]

missing = []
for m in modules:
    try:
        importlib.import_module(m.replace("-", "_"))
    except ImportError:
        missing.append(m)

if missing:
    print(f"\n❌ Missing {len(missing)} modules. Run this command to fix:")
    print(f"pip install {' '.join(missing)}")
else:
    print("\n✅ Environment is 100% ready for training.")
