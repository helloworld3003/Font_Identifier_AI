import torch
from model import ConvNeXtFontEncoder
import os

print("Loading PyTorch model...")
model = ConvNeXtFontEncoder(embedding_dim=512)
model.load_state_dict(torch.load("best_model.pth", map_location='cpu', weights_only=True))
model.eval()

# Dummy input matching the expected shape: (batch_size, channels, height, width)
# From inference.py: resized to (256, 64) -> W=256, H=64. RGB = 3 channels.
dummy_input = torch.randn(1, 3, 64, 256)

print("Exporting to ONNX...")
torch.onnx.export(
    model, 
    dummy_input, 
    "best_model.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print("Successfully exported to best_model.onnx")
print(f"File size: {os.path.getsize('best_model.onnx') / (1024 * 1024):.2f} MB")
