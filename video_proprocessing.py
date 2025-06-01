import imageio_ffmpeg
import ffmpeg
import os
import argparse

def process_videos(input_dir, output_dir, resolution=(64, 64), fps=8):
    os.makedirs(output_dir, exist_ok=True)
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    for file in os.listdir(input_dir):
        if file.endswith(".mp4"):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file)
            (
                ffmpeg
                .input(input_path)
                .filter("fps", fps=fps)
                .filter("scale", resolution[0], resolution[1])
                .output(output_path)
                .overwrite_output()
                .run(cmd=ffmpeg_path, quiet=True)
            )
            print(f"processed: {file}")
    print("Done!")

def main():
    parser = argparse.ArgumentParser(description="Batch process videos with FFmpeg")
    parser.add_argument("input_dir", help="Directory containing input .mp4 videos")
    parser.add_argument("output_dir", help="Directory to save processed videos")
    parser.add_argument("--fps", type=int, default=8, help="Target frames per second")
    parser.add_argument("--resolution", nargs=2, type=int, default=[64, 64],
                        help="Target resolution: width height (default: 64 64)")
    
    args = parser.parse_args()

    process_videos(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        resolution=tuple(args.resolution),
        fps=args.fps
    )

if __name__ == "__main__":
    main()
