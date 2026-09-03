from dataset_loader import SEN2NAIPDataset

print("Loading dataset...")

dataset = SEN2NAIPDataset()

print("\nTesting different samples...")

test_indices = [0, 1, 10, 100, 1000, 4000]

for index in test_indices:

    print(f"\nTesting sample {index}...")

    try:
        lr, hr = dataset[index]

        print("SUCCESS")
        print("LR shape:", lr.shape)
        print("HR shape:", hr.shape)

    except Exception as e:

        print("FAILED")
        print("Error:", type(e).__name__)