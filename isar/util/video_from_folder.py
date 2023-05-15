import cv2
import os
import argparse

def generate_video(folder_path, output_name, frame_rate):
    # Get the list of frames
    frame_files = sorted(os.listdir(folder_path))

    # Read the first frame to get the dimensions (height, width)
    first_frame = cv2.imread(os.path.join(folder_path, frame_files[0]))
    height, width, _ = first_frame.shape

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_name, fourcc, frame_rate, (width, height))

    # Iterate through the frames and write them to the video
    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(folder_path, frame_file))
        video_writer.write(frame)

    # Release the VideoWriter object
    video_writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark, computes evaluation metrics for a DAVIS and a Habitat dataset")

    parser.add_argument(
        "-i", "--img_folder", type=str, 
        default="/home/nico/semesterproject/vis_interim_pres/habitat_data_demo/apartment_0_turn_chair4/combined_vis/", 
        help="Path to image folder"
    )

    parser.add_argument(
        "-o", "--out_file", type=str, 
        default="/home/nico/semesterproject/vis_interim_pres/habitat_data_demo/apartment_0_turn_chair4/habitat_turn_chair_video.avi",
        help="Path to outfile"
    )

    parser.add_argument(
        "-f", "--framerate", type=int, default=24,
        help="framerate of video"
    )

    args = parser.parse_args()

    generate_video(args.img_folder, args.out_file, args.framerate)
