"""End-to-end demo: build synthetic data, train 3 epochs, run a prediction."""
import os
import tempfile

from eurosat_cnn.data import make_synthetic_dataset
from eurosat_cnn.train import train
from eurosat_cnn.predict import predict_image

work = tempfile.mkdtemp()
data = make_synthetic_dataset(os.path.join(work, "ds"), per_class=40, seed=0)
ckpt = os.path.join(work, "model.pt")
model = train(data, epochs=3, out=ckpt)

# Predict on one generated patch
sample = next(
    os.path.join(data, "River", f)
    for f in os.listdir(os.path.join(data, "River"))
)
from eurosat_cnn.predict import load_model
m, classes = load_model(ckpt)
label, conf, _ = predict_image(m, classes, sample)
print(f"Sample is a 'River' patch -> predicted: {label} ({conf:.1%})")
