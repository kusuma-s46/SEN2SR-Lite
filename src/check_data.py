import torch

from dataset_loader import SEN2NAIPDataset


print("Loading dataset...")

dataset = SEN2NAIPDataset()

print("\nReading first sample...")

lr, hr = dataset[0]

print("\nLR information")
print("------------------------")
print("Shape:", lr.shape)
print("Data type:", lr.dtype)
print("Minimum:", lr.min().item())
print("Maximum:", lr.max().item())
print("Mean:", lr.mean().item())

print("\nHR information")
print("------------------------")
print("Shape:", hr.shape)
print("Data type:", hr.dtype)
print("Minimum:", hr.min().item())
print("Maximum:", hr.max().item())
print("Mean:", hr.mean().item())


print("\nChecking each band...")

band_names = ["B02 Blue", "B03 Green", "B04 Red", "B08 NIR"]

for i, band in enumerate(band_names):

    print(
        band,
        "| LR min:", lr[i].min().item(),
        "| LR max:", lr[i].max().item(),
        "| HR min:", hr[i].min().item(),
        "| HR max:", hr[i].max().item()
    )