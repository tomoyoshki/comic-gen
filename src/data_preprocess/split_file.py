import os

def find_series_with_at_least_4_panels(root_folder):
    """
    Find all series under root_folder with at least 4 panels.
    """
    series_paths = []

    # Iterate through all folders N under the root_folder
    for N in os.listdir(root_folder):
        N_path = os.path.join(root_folder, N)

        # Make sure we're looking at a folder
        if os.path.isdir(N_path):
            series_counts = {}  # Dictionary to track counts for each series
            panel_files = os.listdir(N_path)
            for panel_file in panel_files:
                panel_file_elems = panel_file.split("_")
                panel_file_id = panel_file_elems[0]
                if panel_file_id not in series_counts:
                    series_counts[panel_file_id] = 0

                series_counts[panel_file_id] += 1

                if series_counts[panel_file_id] == 4:
                    full_path = os.path.join(N_path, panel_file_id)
                    series_paths.append(full_path)
    return series_paths

def write_to_index_file(index_file_path, series_paths):
    """
    Write the paths of series to the index file.
    """
    with open(index_file_path, "w") as index_file:
        for path in series_paths:
            index_file.write(path + "\n")

def main():
    root_folder = "C:/Users/Tomoyoshi/Documents/cs546/raw_panel_images"
    index_file_path = "index.txt"
    
    series_paths = find_series_with_at_least_4_panels(root_folder)
    write_to_index_file(index_file_path, series_paths)

if __name__ == "__main__":
    main()
