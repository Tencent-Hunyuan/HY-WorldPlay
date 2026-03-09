"""Video utilities for visualization."""

import os
import cv2
import numpy as np
import subprocess
from PIL import Image


def video_to_image_frames(input_video_path, save_directory=None, fps=1):
    """Extracts image frames from a video file at the specified frame rate and saves them as JPEG
    format. Supports regular video files, webcam captures, WebM files, and GIF files, including
    incomplete files.

    Args:
        input_video_path: Path to the input video file
        save_directory: Directory to save extracted frames (default: None)
        fps: Number of frames to extract per second (default: 1)

    Returns: List of file paths to extracted frames
    """
    extracted_frame_paths = []

    # For GIF files, use PIL library for better handling
    if input_video_path.lower().endswith(".gif"):
        try:
            print(f"Processing GIF file using PIL: {input_video_path}")

            with Image.open(input_video_path) as gif_img:
                # Get GIF properties
                frame_duration_ms = gif_img.info.get(
                    "duration", 100
                )  # Duration per frame in milliseconds
                gif_frame_rate = (
                    1000.0 / frame_duration_ms
                    if frame_duration_ms > 0
                    else 10.0
                )  # Convert to frame rate

                print(
                    f"GIF properties: {gif_img.n_frames} frames, {gif_frame_rate:.2f} FPS, {frame_duration_ms}ms per frame"
                )

                # Calculate sampling interval
                sampling_interval = (
                    max(1, int(gif_frame_rate / fps))
                    if fps < gif_frame_rate
                    else 1
                )

                saved_count = 0
                for current_frame_index in range(gif_img.n_frames):
                    gif_img.seek(current_frame_index)

                    # Sample frames based on desired frame rate
                    if current_frame_index % sampling_interval == 0:
                        # Convert to RGB format if necessary
                        rgb_frame = gif_img.convert("RGB")

                        # Convert PIL image to numpy array
                        frame_ndarray = np.array(rgb_frame)

                        # Save frame as JPEG format
                        frame_output_path = os.path.join(
                            save_directory, f"frame_{saved_count:06d}.jpg"
                        )
                        pil_image = Image.fromarray(frame_ndarray)
                        pil_image.save(frame_output_path, "JPEG", quality=95)
                        extracted_frame_paths.append(frame_output_path)
                        saved_count += 1

                if extracted_frame_paths:
                    print(
                        f"Successfully extracted {len(extracted_frame_paths)} frames from GIF using PIL"
                    )
                    return extracted_frame_paths

        except Exception as error:
            print(
                f"PIL GIF extraction error: {str(error)}, falling back to OpenCV"
            )

    # For WebM files, use FFmpeg directly for more stable processing
    if input_video_path.lower().endswith(".webm"):
        try:
            print(f"Processing WebM file using FFmpeg: {input_video_path}")

            # Create a unique output pattern for the frames
            output_frame_pattern = os.path.join(
                save_directory, "frame_%04d.jpg"
            )

            # Use FFmpeg to extract frames at specified frame rate
            ffmpeg_command = [
                "ffmpeg",
                "-i",
                input_video_path,
                "-vf",
                f"fps={fps}",  # Specified frames per second
                "-q:v",
                "2",  # High quality
                output_frame_pattern,
            ]

            # Run FFmpeg process
            ffmpeg_process = subprocess.Popen(
                ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            process_stdout, process_stderr = ffmpeg_process.communicate()

            # Collect all extracted frames
            for filename in sorted(os.listdir(save_directory)):
                if filename.startswith("frame_") and filename.endswith(".jpg"):
                    full_frame_path = os.path.join(save_directory, filename)
                    extracted_frame_paths.append(full_frame_path)

            if extracted_frame_paths:
                print(
                    f"Successfully extracted {len(extracted_frame_paths)} frames from WebM using FFmpeg"
                )
                return extracted_frame_paths

            print("FFmpeg extraction failed, falling back to OpenCV")
        except Exception as error:
            print(
                f"FFmpeg extraction error: {str(error)}, falling back to OpenCV"
            )

    # Standard OpenCV method for non-WebM files or as fallback
    try:
        video_capture = cv2.VideoCapture(input_video_path)

        # For WebM files, try setting more robust decoder options
        if input_video_path.lower().endswith(".webm"):
            video_capture.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"VP80")
            )

        source_fps = video_capture.get(cv2.CAP_PROP_FPS)
        extraction_interval = max(
            1, int(source_fps / fps)
        )  # Extract at specified frame rate
        processed_frame_count = 0

        # Set error mode to suppress console warnings
        cv2.setLogLevel(0)

        while True:
            read_success, current_frame = video_capture.read()
            if not read_success:
                break

            if processed_frame_count % extraction_interval == 0:
                try:
                    # Additional check for valid frame data
                    if current_frame is not None and current_frame.size > 0:
                        rgb_converted_frame = cv2.cvtColor(
                            current_frame, cv2.COLOR_BGR2RGB
                        )
                        frame_output_path = os.path.join(
                            save_directory,
                            f"frame_{processed_frame_count:06d}.jpg",
                        )
                        cv2.imwrite(
                            frame_output_path,
                            cv2.cvtColor(
                                rgb_converted_frame, cv2.COLOR_RGB2BGR
                            ),
                        )
                        extracted_frame_paths.append(frame_output_path)
                except Exception as error:
                    print(
                        f"Warning: Failed to process frame {processed_frame_count}: {str(error)}"
                    )

            processed_frame_count += 1

            # Safety limit to prevent infinite loops
            if processed_frame_count > 1000:
                break

        video_capture.release()
        print(
            f"Extracted {len(extracted_frame_paths)} frames from video using OpenCV"
        )

    except Exception as error:
        print(f"Error extracting frames: {str(error)}")

    return extracted_frame_paths
