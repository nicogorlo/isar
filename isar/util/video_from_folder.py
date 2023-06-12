import cv2
import os
import argparse

def generate_video(folder_path, semantic_path, output_name, frame_rate, visualize_semantics = False):

    frame_files = sorted(os.listdir(folder_path))

    first_frame = cv2.imread(os.path.join(folder_path, frame_files[0]))
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_name, fourcc, frame_rate, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(folder_path, frame_file))
        semantic = cv2.imread(os.path.join(semantic_path, frame_file.replace("jpg", "png")))
        combined = cv2.addWeighted(frame.astype('uint8'), 0.8, semantic.astype('uint8'), 0.2, 0).astype('uint8')

        if visualize_semantics:
            video_writer.write(combined)
        else:
            video_writer.write(frame)

    video_writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark, computes evaluation metrics for a DAVIS and a Habitat dataset")

    parser.add_argument(
        "-i", "--img_folder", type=str, 
        default="", 
        help="Path to image folder"
    )

    parser.add_argument(
        "-g", "--gt_folder", type=str,
        default="",
        help="Path to ground truth folder"
    )

    parser.add_argument(
        "-o", "--out_file", type=str, 
        default="",
        help="Path to outfile"
    )

    parser.add_argument(
        "-vs", "--visualize_semantics", type=bool, default=False,
        help="Visualize semantics"
    )

    parser.add_argument(
        "-f", "--framerate", type=int, default=30,
        help="framerate of video"
    )

    args = parser.parse_args()

    generate_video(args.img_folder, args.gt_folder, args.out_file, args.framerate, args.visualize_semantics)
