from os import listdir, path


def get_dir_files(dir, *, split=''):
    """Return all files in all directories in the given directory."""
    files = []
    for directory in listdir(dir):
        if split and split not in directory:
            continue

        for filename in listdir(path.join(dir, directory)):
            files.append(path.join(directory, filename))
    return sorted(files)
