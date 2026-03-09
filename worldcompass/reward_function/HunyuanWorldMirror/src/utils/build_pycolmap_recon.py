import numpy as np
import pycolmap


def _create_camera_params(
    frame_idx, cam_matrices, model_type, distortion_coeffs=None
):
    """Build camera parameter array for different model types."""
    if model_type == "PINHOLE":
        return np.array(
            [
                cam_matrices[frame_idx][0, 0],
                cam_matrices[frame_idx][1, 1],
                cam_matrices[frame_idx][0, 2],
                cam_matrices[frame_idx][1, 2],
            ]
        )
    elif model_type == "SIMPLE_PINHOLE":
        focal_avg = (
            cam_matrices[frame_idx][0, 0] + cam_matrices[frame_idx][1, 1]
        ) / 2
        return np.array(
            [
                focal_avg,
                cam_matrices[frame_idx][0, 2],
                cam_matrices[frame_idx][1, 2],
            ]
        )
    elif model_type == "SIMPLE_RADIAL":
        raise NotImplementedError("SIMPLE_RADIAL model not supported")
    else:
        raise ValueError(f"Unsupported camera model: {model_type}")


def _setup_camera_object(
    frame_idx, cam_matrices, img_dims, model_type, use_shared
):
    """Create and configure camera object."""
    if use_shared and frame_idx > 0:
        return None

    params = _create_camera_params(frame_idx, cam_matrices, model_type)
    return pycolmap.Camera(
        model=model_type,
        width=img_dims[0],
        height=img_dims[1],
        params=params,
        camera_id=frame_idx + 1,
    )


def _process_frame_points(scene_points, point_coords, frame_idx):
    """Extract and process 2D points belonging to specific frame."""
    frame_mask = point_coords[:, 2].astype(np.int32) == frame_idx
    valid_indices = np.nonzero(frame_mask)[0]

    point2d_list = []
    for idx, batch_idx in enumerate(valid_indices):
        point3d_id = batch_idx + 1
        xy_coords = point_coords[batch_idx][:2]
        point2d_list.append(pycolmap.Point2D(xy_coords, point3d_id))

        # Update track information
        track = scene_points.points3D[point3d_id].track
        track.add_element(frame_idx + 1, idx)

    return point2d_list


def build_pycolmap_reconstruction(
    points,
    pixel_coords,
    point_colors,
    poses,
    intrinsics,
    image_size,
    shared_camera_model=False,
    camera_model="SIMPLE_PINHOLE",
):
    """Convert numpy arrays to pycolmap reconstruction format.

    Creates 3D scene structure without track optimization. Suitable for initialization of neural
    rendering methods.
    """
    num_frames = len(poses)
    num_points = len(points)

    scene = pycolmap.Reconstruction()

    # Add 3D points to scene
    for pt_idx in range(num_points):
        scene.add_point3D(
            points[pt_idx], pycolmap.Track(), point_colors[pt_idx]
        )

    current_camera = None

    # Process each frame
    for frame_idx in range(num_frames):
        # Setup camera if needed
        if current_camera is None or not shared_camera_model:
            current_camera = _setup_camera_object(
                frame_idx,
                intrinsics,
                image_size,
                camera_model,
                shared_camera_model,
            )
            scene.add_camera(current_camera)

        # Create pose transformation
        rotation_matrix = poses[frame_idx][:3, :3]
        translation_vec = poses[frame_idx][:3, 3]
        world_to_cam = pycolmap.Rigid3d(
            pycolmap.Rotation3d(rotation_matrix), translation_vec
        )

        # Create image object
        frame_image = pycolmap.Image(
            id=frame_idx + 1,
            name=f"frame_{frame_idx + 1}",
            camera_id=current_camera.camera_id,
            cam_from_world=world_to_cam,
        )

        # Process 2D points for this frame
        frame_points = _process_frame_points(scene, pixel_coords, frame_idx)

        # Set image points and registration status
        try:
            frame_image.points2D = pycolmap.ListPoint2D(frame_points)
            frame_image.registered = True
        except:
            print(f"Warning: Frame {frame_idx + 1} has no valid points")
            frame_image.registered = False

        scene.add_image(frame_image)

    return scene
