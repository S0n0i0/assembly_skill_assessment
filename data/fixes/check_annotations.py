def read_file(file_path, file_element_fn):
    with open(file_path, "r") as file:
        lines = file.readlines()
        return lines, set([file_element_fn(line) for line in lines])


def write_file(file_path, ref_lines, original_lines=None, file_element_fn=None):
    with open(file_path, "w") as file:
        if original_lines is not None and file_element_fn is not None:
            for line in original_lines:
                if file_element_fn(line) in ref_lines:
                    file.write(line)
        else:
            for line in ref_lines:
                file.write(line)


def main():
    file1_path = "D:/data/annotations/fine-grained-annotations/old/head_actions.txt"
    file1_element_fn = lambda line: line.strip()
    file2_path = "D:/data/annotations/fine-grained-annotations/actions.csv"
    file2_element_fn = lambda line: line.split(",")[4]
    common_file_path = "D:/data/annotations/fine-grained-annotations/head_actions.txt"
    differences_file_path = None
    """(
        "D:/data/annotations/fine-grained-annotations/differences.csv"
    )"""

    original_file1, lines_file1 = read_file(file1_path, file1_element_fn)
    _, lines_file2 = read_file(file2_path, file2_element_fn)
    print("File1 lines:", len(lines_file1))
    print("File2 lines:", len(lines_file2))

    common_lines = lines_file1.intersection(lines_file2)
    print("Common lines between the files:", len(common_lines))
    if common_file_path is not None:
        write_file(common_file_path, common_lines, original_file1, file1_element_fn)

    differences = lines_file1.symmetric_difference(lines_file2)
    print("Differences between the files:", len(differences))
    if differences_file_path is not None:
        write_file(differences_file_path, differences, original_file1, file1_element_fn)


if __name__ == "__main__":
    main()
