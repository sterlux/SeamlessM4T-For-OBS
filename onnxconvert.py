import torch
import onnx

# Load your PyTorch model
model = torch.load('seamlessM4T_v2_large.pt')
model.eval()

# Create a dummy input tensor with the same shape as your model's input
dummy_input = torch.randn(1, 3, 224, 224)  # Adjust the shape based on your model

# Export the model to ONNX format
onnx_path = "seamless4t_model.onnx"
torch.onnx.export(model, dummy_input, onnx_path)
