coarse_split_dir = (
    ""  # test_coarse_assembly.txt, train_coarse_assembly.txt, val_coarse_assembly.txt
)
fine_split_dir = ""  # test.csv, train.csv, validation.csv
skill_dir = ""  # skill_1.txt ... skill_5.txt

dir_to_check = 1  # 0 coarse_split_dir, 1 fine_split_dir

coarse_splits = ["test", "train", "val"]
fine_splits = ["test", "train", "validation"]
skill_splits = {
    split: [0] * 5 for split in (coarse_splits if dir_to_check == 0 else fine_splits)
}

if dir_to_check == 0:
    for split in coarse_splits:
        with open(f"{coarse_split_dir}/{split}_coarse_assembly.txt") as f:
            split_files = f.read().splitlines()
            split_files = [
                file.split("assembly_")[1].split(".txt")[0] for file in split_files
            ]
            print(f"Number of files in {split} split: {len(split_files)}")
        for skill in range(1, 6):
            with open(f"{skill_dir}/skill_{skill}.txt") as f:
                skill_files = f.read().splitlines()
            skill_splits[split][skill - 1] = len(
                set(split_files).intersection(set(skill_files))
            )
elif dir_to_check == 1:
    for split in fine_splits:
        with open(f"{fine_split_dir}/{split}.csv") as f:
            split_files = f.read().splitlines()
            split_files = [file.split(",")[1] for file in split_files]
            # filter split_files to only include files that have "HMC" in their name
            split_files = [file.split("/")[0] for file in split_files if "HMC" in file]
            split_files = list(set(split_files))
            print(f"Number of files in {split} split: {len(split_files)}")
        for skill in range(1, 6):
            with open(f"{skill_dir}/skill_{skill}.txt") as f:
                skill_files = f.read().splitlines()
            skill_splits[split][skill - 1] = len(
                set(split_files).intersection(set(skill_files))
            )

print(skill_splits)
# print the sum of all numbers in skill_splits to check if all files are accounted for
print(
    sum(
        [
            sum(skill_splits[split])
            for split in (coarse_splits if dir_to_check == 0 else fine_splits)
        ]
    )
)
