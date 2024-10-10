import os

not_used_path = "D:/data/annotations/coarse-annotations/not_used_videos.csv"
coarse_labels_path = "D:/data/annotations/coarse-annotations/coarse_labels/"

with open(not_used_path, "r") as f:
    not_cropped = set(
        [
            line.split(",")[0]
            for line in f.readlines()
            if line.strip().split(",")[1] == "no_coarse_annotations"
        ]
    )
coarse_labels = set(
    [f[9:] for f in os.listdir(coarse_labels_path) if f[:9] == "assembly_"]
)
diff = coarse_labels - not_cropped

if len(coarse_labels) == len(diff):
    print(
        f"All non-cropped videos ({len(not_cropped)}/{len(not_cropped) + len(coarse_labels)}) are not in the coarse labels."
    )
else:
    print("Some non-cropped videos are in the coarse labels.")
    print(diff)

present_people = set([s.split("_")[4] for s in coarse_labels])
print(f"Total people ({len(present_people)})")
not_cropped_people = set([s.split("_")[4] for s in not_cropped])
print(f"People with non-cropped videos ({len(not_cropped_people)})")
missing_people = not_cropped_people - present_people
print(f"- People with all videos not cropped: {len(missing_people)}")

not_cropped_by_people = {p: [] for p in not_cropped_people}
for v in not_cropped:
    not_cropped_by_people[v.split("_")[4]].append(v)

non_cropped_counter = {}
for p in not_cropped_by_people:
    if len(not_cropped_by_people[p]) not in non_cropped_counter:
        non_cropped_counter[len(not_cropped_by_people[p])] = []
    non_cropped_counter[len(not_cropped_by_people[p])].append(p)
for i in non_cropped_counter:
    print(f"- People with {i} videos not cropped: {len(non_cropped_counter[i])}")
