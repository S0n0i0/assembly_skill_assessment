import os

from utils.functions import all_subsets

views = ["ego", "fixed"]
video_paths = {view: f"D:/data/{view}_recordings/" for view in views}
offsets_paths = {
    view: f"D:/data/annotations/coarse-annotations/{view}_offsets.csv" for view in views
}
not_used_files_path = f"D:/data/annotations/coarse-annotations/not_used_videos.csv"
new_not_used = False

fix = True
sequences = {view: set(os.listdir(video_paths[view])) for view in views}

# Find sequences that are in one but not the other
print("Sequences:")
for view in views:
    print(f"- {view} sequences: {len(sequences[view])}")

print("Non common sequences:")
non_common = {}
for vs in zip(views, views[::-1]):
    non_common[vs[0]] = sequences[vs[0]] - sequences[vs[1]]
    print(
        f"- {vs[0]} sequences but not in {vs[1]} ({len(non_common[vs[0]])}):",
        non_common[vs[0]],
    )

print("Sequences without files in the view, but not in the other:")
no_files = {
    "ego": set(),
    "fixed": set(),
}
for view in views:
    for name in sequences[view]:
        if len(os.listdir(os.path.join(video_paths[view], name))) == 0:
            no_files[view].add(name)
    print(
        f"- {view} ({len(no_files[view])}):",
        no_files[view],
    )


if fix:
    print("\nFixes:")
    for view in views:
        if len(non_common[view]) == 0 and len(no_files[view]) == 0:
            print(f"No fixes needed for {view} view.")
            continue
        print(f"Fixing {view} view:")

        for collection, code, message in [
            (
                non_common[view],
                "no_ego/fixed_videos",
                "- Deleted non common sequences.",
            ),
            (
                no_files[view],
                "no_ego/fixed_videos",
                "- Deleted sequences without files.",
            ),
        ]:
            if not new_not_used:
                with open(not_used_files_path, "r") as f:
                    lines = set([l.split(",")[0] for l in f.readlines()])
            with open(not_used_files_path, "w" if new_not_used else "a") as f:
                for name in collection:
                    if new_not_used or name not in lines:
                        f.write(f"{name},{code}\n")

            for name in collection:
                for file in os.listdir(os.path.join(video_paths[view], name)):
                    os.remove(os.path.join(video_paths[view], name, file))
                os.rmdir(os.path.join(video_paths[view], name))
            print(message)

        # Remove the sequences from the offsets file and adjust the ids
        with open(offsets_paths[view], "r") as f:
            lines = f.readlines()
        with open(offsets_paths[view], "w") as f:
            count = -1
            last_id = -1
            for line in lines[1:]:
                line = line.split(",")
                id = int(line[0])
                name = line[1].split("/")[0]
                if name not in no_files[view] and name not in non_common[view]:
                    if id != last_id:
                        count += 1
                        last_id = id
                    f.write(f"{count},{','.join(line[1:])}")
        print("- Adjusted ego offsets.")

        # Remove
