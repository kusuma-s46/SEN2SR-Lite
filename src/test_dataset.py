import tacoreader.v1 as tacoreader
import matplotlib.pyplot as plt
import rasterio

print("Connecting to SEN2NAIPv2...")

dataset = tacoreader.load("tacofoundation:sen2naipv2-unet")

print("Dataset loaded successfully!")
print("Number of samples:", len(dataset))

# Read sample 0
sample = dataset.read(0)

print("\nSample information:")
print(sample)

# Get LR and HR file paths
lr_path = sample.loc[sample["tortilla:id"] == "lr", "internal:subfile"].iloc[0]
hr_path = sample.loc[sample["tortilla:id"] == "hr", "internal:subfile"].iloc[0]

print("\nLR path:")
print(lr_path)

print("\nHR path:")
print(hr_path)

# Open images
with rasterio.open(lr_path) as lr_src:
    lr_image = lr_src.read()

with rasterio.open(hr_path) as hr_src:
    hr_image = hr_src.read()

print("\nLR shape:", lr_image.shape)
print("HR shape:", hr_image.shape)

# Display RGB
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# B04 = Red, B03 = Green, B02 = Blue
lr_rgb = lr_image[[2, 1, 0], :, :]
hr_rgb = hr_image[[2, 1, 0], :, :]

# Normalize for display
def normalize(image):
    image = image.astype(float)
    min_val = image.min()
    max_val = image.max()

    if max_val > min_val:
        image = (image - min_val) / (max_val - min_val)

    return image

lr_rgb = normalize(lr_rgb)
hr_rgb = normalize(hr_rgb)

axes[0].imshow(lr_rgb.transpose(1, 2, 0))
axes[0].set_title("Sentinel-2 LR")
axes[0].axis("off")

axes[1].imshow(hr_rgb.transpose(1, 2, 0))
axes[1].set_title("High-Resolution Reference")
axes[1].axis("off")

plt.tight_layout()
plt.show()