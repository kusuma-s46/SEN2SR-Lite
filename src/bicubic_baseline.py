import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dataset_loader import SEN2NAIPDataset


print("Loading SEN2NAIPv2 dataset...")

dataset = SEN2NAIPDataset()

print("\nReading sample 0...")

lr, hr = dataset[0]

print("Original LR shape:", lr.shape)
print("Original HR shape:", hr.shape)


# --------------------------------------------------
# Bicubic Upscaling
# --------------------------------------------------

# Add batch dimension
lr_batch = lr.unsqueeze(0)

print("\nApplying Bicubic upscaling...")

sr = F.interpolate(
    lr_batch,
    size=(520, 520),
    mode="bicubic",
    align_corners=False
)

# Remove batch dimension
sr = sr.squeeze(0)

print("Bicubic SR shape:", sr.shape)


# --------------------------------------------------
# Calculate MSE and PSNR
# --------------------------------------------------

mse = torch.mean((sr - hr) ** 2)

if mse == 0:
    psnr = float("inf")
else:
    data_range = hr.max() - hr.min()
    psnr = 20 * torch.log10(data_range / torch.sqrt(mse))

print("\nBicubic Results")
print("----------------------")
print("MSE :", mse.item())
print("PSNR:", psnr.item(), "dB")


# --------------------------------------------------
# Display RGB images
# --------------------------------------------------

def normalize(image):

    image = image.detach().cpu().float()

    min_value = image.min()
    max_value = image.max()

    if max_value > min_value:
        image = (image - min_value) / (max_value - min_value)

    return image


# B04 = Red
# B03 = Green
# B02 = Blue

lr_rgb = lr[[2, 1, 0], :, :]
sr_rgb = sr[[2, 1, 0], :, :]
hr_rgb = hr[[2, 1, 0], :, :]

lr_rgb = normalize(lr_rgb)
sr_rgb = normalize(sr_rgb)
hr_rgb = normalize(hr_rgb)


# --------------------------------------------------
# Plot
# --------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(lr_rgb.permute(1, 2, 0))
axes[0].set_title("Original Sentinel-2 LR")
axes[0].axis("off")

axes[1].imshow(sr_rgb.permute(1, 2, 0))
axes[1].set_title("Bicubic ×4")
axes[1].axis("off")

axes[2].imshow(hr_rgb.permute(1, 2, 0))
axes[2].set_title("HR Reference")
axes[2].axis("off")

plt.tight_layout()

plt.savefig(
    "results/bicubic_comparison.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("\nBicubic comparison saved to:")
print("results/bicubic_comparison.png")