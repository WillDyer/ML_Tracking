import os


source_dir = "/run/media/will/Will_s SSD1/University_Projects/YR3/Twelvefold/previsfootage/cm30_reversed_000/"


def rename_files(source_dir, old_range_start, old_range_end, new_range_start):
    frame_offset = new_range_start - old_range_start

    for filename in os.listdir(source_dir):
        if filename.endswith(".exr") and "frame_" in filename:
            old_frame_number = int(filename.split('_')[1].split('.')[0])

            if old_range_start <= old_frame_number <= old_range_end:
                new_frame_number = old_frame_number + frame_offset

                new_filename = f"frame_{new_frame_number:04d}.exr"
                old_filepath = os.path.join(source_dir, filename)
                new_filepath = os.path.join(source_dir, new_filename)

                os.rename(old_filepath, new_filepath)
                print(f"Renamed {filename} to {new_filename}")


rename_files(source_dir, old_range_start=0, old_range_end=91, new_range_start=233)
