import os, sys
from shutil import copyfile

sys.path.insert(0, os.path.join(".."))
import pythonscraper.db as db
sys.path.insert(0, os.path.join("supervised_learning"))



dest_dir = os.path.join("..", "data")
origin_dir = os.path.join("..", "comics")

# Go through all images in dest_dir and copy them in the correct
# folder 
def give_label(publication_year):
    if(int(publication_year) < 1954):
        return "Golden Age"
    elif(int(publication_year) < 1970):
        return "Silver Age"
    elif(int(publication_year) < 1986):
        return "Bronze Age"
    else:
        return "Modern Age"

golden_age_images = []
silver_age_images = []
bronze_age_images = []
modern_age_images = []

for dirpath, dirnames, filenames in os.walk(origin_dir):
    if not dirnames:
        for file in filenames:
            publication_year = int(dirpath.split("/")[-1])
            label = give_label(publication_year)
            image_path = str(dirpath) + "/" + str(file)
            if(label == "Golden Age"):
                golden_age_images.append(image_path)
            elif (label == "Silver Age"):
                silver_age_images.append(image_path)
            elif (label == "Bronze Age"):
                bronze_age_images.append(image_path)
            else:
                modern_age_images.append(image_path)


n_gold = len(golden_age_images)
print(n_gold)
n_silver = len(silver_age_images)
print(n_silver)
n_bronze = len(bronze_age_images)
print(n_bronze)
n_modern = len(modern_age_images)
print(n_modern)

# Copie les dans le bon dossier : train/val/test
ratio_train = 0.6
ratio_val = 0.2
ratio_test = 0.2

splits = {
    "split_gold": [int(ratio_train * n_gold), int(ratio_val * n_gold) + int(ratio_train * n_gold)],
    "split_silver": [int(ratio_train * n_silver), int(ratio_val * n_silver) + int(ratio_train * n_silver)],
    "split_bronze": [int(ratio_train * n_bronze), int(ratio_val * n_bronze) + int(ratio_train * n_bronze)],
    "split_modern": [int(ratio_train * n_modern), int(ratio_val * n_modern) + int(ratio_train * n_modern)]
}
print(splits)

# Copy golden images
for i in range(splits["split_gold"][0]):
    copyfile(golden_age_images[i], os.path.join(dest_dir, "train", "Golden Age", golden_age_images[i].split("/")[-1]))
for i in range(splits["split_gold"][0], splits["split_gold"][1]):
    copyfile(golden_age_images[i], os.path.join(dest_dir, "val", "Golden Age", golden_age_images[i].split("/")[-1]))
for i in range(splits["split_gold"][1], n_gold):
    copyfile(golden_age_images[i], os.path.join(dest_dir, "test", "Golden Age", golden_age_images[i].split("/")[-1]))

# Copy silver images
for i in range(splits["split_silver"][0]):
    copyfile(silver_age_images[i], os.path.join(dest_dir, "train", "Silver Age", silver_age_images[i].split("/")[-1]))
for i in range(splits["split_silver"][0], splits["split_silver"][1]):
    copyfile(silver_age_images[i], os.path.join(dest_dir, "val", "Silver Age", silver_age_images[i].split("/")[-1]))
for i in range(splits["split_silver"][1], n_silver):
    copyfile(silver_age_images[i], os.path.join(dest_dir, "test", "Silver Age", silver_age_images[i].split("/")[-1]))

# Copy bronze images
for i in range(splits["split_bronze"][0]):
    copyfile(bronze_age_images[i], os.path.join(dest_dir, "train", "Bronze Age", bronze_age_images[i].split("/")[-1]))
for i in range(splits["split_bronze"][0], splits["split_bronze"][1]):
    copyfile(bronze_age_images[i], os.path.join(dest_dir, "val", "Bronze Age", bronze_age_images[i].split("/")[-1]))
for i in range(splits["split_bronze"][1], n_bronze):
    copyfile(bronze_age_images[i], os.path.join(dest_dir, "test", "Bronze Age", bronze_age_images[i].split("/")[-1]))

# Copy modern images
for i in range(splits["split_modern"][0]):
    copyfile(modern_age_images[i], os.path.join(dest_dir, "train", "Modern Age", modern_age_images[i].split("/")[-1]))
for i in range(splits["split_modern"][0], splits["split_modern"][1]):
    copyfile(modern_age_images[i], os.path.join(dest_dir, "val", "Modern Age", modern_age_images[i].split("/")[-1]))
for i in range(splits["split_modern"][1], n_modern):
    copyfile(modern_age_images[i], os.path.join(dest_dir, "test", "Modern Age", modern_age_images[i].split("/")[-1]))