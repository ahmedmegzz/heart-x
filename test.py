import numpy as np
import torch
from heartx.config import AAMI_CLASSES
from heartx.models.dual_stream import HeartXDualStream

data = np.load("outputs/processed/mitbih/mitbih_dual.npz", allow_pickle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = HeartXDualStream(
    num_classes=5,
    clinical_dim=data["clinical"].shape[1],
).to(device)

checkpoint = torch.load(
    "outputs/checkpoints/heartx_dual.pt",
    map_location=device,
    weights_only=False,
)
model.load_state_dict(checkpoint["model_state"])
model.eval()

index = 0
x1d = torch.from_numpy(data["x_1d"][index:index+1]).float().to(device)
x2d = torch.from_numpy(data["x_2d"][index:index+1]).float().to(device)
clinical = torch.from_numpy(data["clinical"][index:index+1]).float().to(device)

with torch.no_grad():
    probabilities = torch.softmax(model(x1d, x2d, clinical), dim=1)[0]

prediction = int(probabilities.argmax())
actual = int(data["y"][index])

print("Actual:    ", AAMI_CLASSES[actual])
print("Predicted: ", AAMI_CLASSES[prediction])
print("Probabilities:", {
    name: round(float(prob), 4)
    for name, prob in zip(AAMI_CLASSES, probabilities)
})
print("Prediction: ", prediction)
print("Actual: ", actual)