# generate_test_videos.py
# Generate 100 test videos with patterns and variations for robust ASI testing

import cv2
import numpy as np
import random
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baby_asi import ASISeed, CuriosityStream, to_device


# Define our vocabulary
COLORS = {
    'red': (0, 0, 255),
    'blue': (255, 0, 0),
    'green': (0, 255, 0),
    'yellow': (0, 255, 255),
    'purple': (255, 0, 255),
}

SHAPES = ['circle', 'square', 'triangle']
MOTIONS = ['horizontal', 'vertical', 'diagonal', 'circular', 'zigzag', 'rotating']
SIZES = {'small': 8, 'medium': 12, 'large': 16}


def create_video_writer(filename, fps=30, size=(64, 64)):
    """Create a video writer for MP4 output"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(filename, fourcc, fps, size)


def draw_circle(frame, center, radius, color):
    """Draw a filled circle on frame"""
    cv2.circle(frame, center, radius, color, -1)


def draw_square(frame, center, size, color):
    """Draw a filled square on frame"""
    half_size = size // 2
    top_left = (center[0] - half_size, center[1] - half_size)
    bottom_right = (center[0] + half_size, center[1] + half_size)
    cv2.rectangle(frame, top_left, bottom_right, color, -1)


def draw_triangle(frame, center, size, color, angle=0):
    """Draw a filled triangle rotated by angle (degrees)"""
    height = int(size * np.sqrt(3) / 2)
    points = np.array([
        [0, -height * 2 // 3],
        [-size // 2, height // 3],
        [size // 2, height // 3]
    ], dtype=np.int32)

    angle_rad = np.radians(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    points = points @ rotation_matrix.T

    points[:, 0] += center[0]
    points[:, 1] += center[1]
    points = points.astype(np.int32)

    cv2.fillPoly(frame, [points], color)


def get_position(frame_num, total_frames, motion_type, width, height, margin):
    """Calculate position based on motion type"""
    progress = frame_num / total_frames

    if motion_type == 'horizontal':
        # Left to right, 2 passes
        x = int(((progress * 2) % 1.0) * (width - 2 * margin) + margin)
        y = height // 2
        angle = 0

    elif motion_type == 'vertical':
        # Top to bottom, 2 passes
        x = width // 2
        y = int(((progress * 2) % 1.0) * (height - 2 * margin) + margin)
        angle = 0

    elif motion_type == 'diagonal':
        # Diagonal movement, 2 passes
        t = (progress * 2) % 1.0
        x = int(t * (width - 2 * margin) + margin)
        y = int(t * (height - 2 * margin) + margin)
        angle = 0

    elif motion_type == 'circular':
        # Circular motion, 3 complete circles
        angle_rad = progress * 2 * np.pi * 3
        radius = min(width, height) // 3
        x = int(width // 2 + radius * np.cos(angle_rad))
        y = int(height // 2 + radius * np.sin(angle_rad))
        angle = 0

    elif motion_type == 'zigzag':
        # Zigzag horizontal motion
        t = (progress * 2) % 1.0
        x = int(t * (width - 2 * margin) + margin)
        y = int(height // 2 + 10 * np.sin(t * 4 * np.pi))
        angle = 0

    elif motion_type == 'rotating':
        # Stay in center, rotate
        x = width // 2
        y = height // 2
        angle = progress * 360 * 3  # 3 full rotations

    return x, y, angle


def generate_video(output_path, color_name, shape, motion, size_name, fps=30, duration=10):
    """Generate a single video with specified parameters"""
    width, height = 64, 64
    size = SIZES[size_name]
    color = COLORS[color_name]

    writer = create_video_writer(output_path, fps, (width, height))
    total_frames = fps * duration

    for frame_num in range(total_frames):
        # White background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Get position
        x, y, angle = get_position(frame_num, total_frames, motion, width, height, size)

        # Draw shape
        if shape == 'circle':
            draw_circle(frame, (x, y), size, color)
        elif shape == 'square':
            draw_square(frame, (x, y), size * 2, color)
        elif shape == 'triangle':
            draw_triangle(frame, (x, y), size * 2, color, angle)

        writer.write(frame)

    writer.release()


def generate_dataset():
    """Generate 100 videos with patterns and variations"""
    print("=" * 70)
    print("GENERATING 100 TEST VIDEOS WITH PATTERNS")
    print("=" * 70)
    print()

    output_dir = "test_videos"
    os.makedirs(output_dir, exist_ok=True)

    # Track what we generate
    train_videos = []
    test_videos = []

    video_count = 0

    # TRAINING SET: Systematic coverage with repetitions
    print("Generating TRAINING videos (80 videos)...")

    # Pattern 1: All red circles with different motions (learn "red circle")
    for motion in ['horizontal', 'vertical', 'diagonal', 'circular']:
        for size in ['small', 'medium']:
            video_count += 1
            filename = f"train_{video_count:03d}_red_circle_{motion}_{size}.mp4"
            path = os.path.join(output_dir, filename)
            generate_video(path, 'red', 'circle', motion, size)
            train_videos.append({
                'id': video_count,
                'filename': filename,
                'color': 'red',
                'shape': 'circle',
                'motion': motion,
                'size': size,
                'type': 'train'
            })
            print(f"  {video_count}/100: {filename}")

    # Pattern 2: All blue squares with different motions (learn "blue square")
    for motion in ['horizontal', 'vertical', 'zigzag']:
        for size in ['small', 'medium']:
            video_count += 1
            filename = f"train_{video_count:03d}_blue_square_{motion}_{size}.mp4"
            path = os.path.join(output_dir, filename)
            generate_video(path, 'blue', 'square', motion, size)
            train_videos.append({
                'id': video_count,
                'filename': filename,
                'color': 'blue',
                'shape': 'square',
                'motion': motion,
                'size': size,
                'type': 'train'
            })
            print(f"  {video_count}/100: {filename}")

    # Pattern 3: All green triangles rotating (learn "green triangle")
    for size in ['small', 'medium', 'large']:
        video_count += 1
        filename = f"train_{video_count:03d}_green_triangle_rotating_{size}.mp4"
        path = os.path.join(output_dir, filename)
        generate_video(path, 'green', 'triangle', 'rotating', size)
        train_videos.append({
            'id': video_count,
            'filename': filename,
            'color': 'green',
            'shape': 'triangle',
            'motion': 'rotating',
            'size': size,
            'type': 'train'
        })
        print(f"  {video_count}/100: {filename}")

    # Pattern 4: Yellow shapes (various)
    for shape in ['circle', 'square']:
        for motion in ['horizontal', 'vertical']:
            video_count += 1
            filename = f"train_{video_count:03d}_yellow_{shape}_{motion}_medium.mp4"
            path = os.path.join(output_dir, filename)
            generate_video(path, 'yellow', shape, motion, 'medium')
            train_videos.append({
                'id': video_count,
                'filename': filename,
                'color': 'yellow',
                'shape': shape,
                'motion': motion,
                'size': 'medium',
                'type': 'train'
            })
            print(f"  {video_count}/100: {filename}")

    # Pattern 5: Purple shapes (introducing new color)
    for shape in ['circle', 'square', 'triangle']:
        video_count += 1
        motion = 'circular' if shape == 'circle' else 'horizontal'
        filename = f"train_{video_count:03d}_purple_{shape}_{motion}_medium.mp4"
        path = os.path.join(output_dir, filename)
        generate_video(path, 'purple', shape, motion, 'medium')
        train_videos.append({
            'id': video_count,
            'filename': filename,
            'color': 'purple',
            'shape': shape,
            'motion': motion,
            'size': 'medium',
            'type': 'train'
        })
        print(f"  {video_count}/100: {filename}")

    # Pattern 6: More variations to reach 80 videos
    random.seed(42)  # Reproducible
    while len(train_videos) < 80:
        color = random.choice(list(COLORS.keys()))
        shape = random.choice(SHAPES)
        motion = random.choice(MOTIONS)
        size = random.choice(list(SIZES.keys()))

        video_count += 1
        filename = f"train_{video_count:03d}_{color}_{shape}_{motion}_{size}.mp4"
        path = os.path.join(output_dir, filename)
        generate_video(path, color, shape, motion, size)
        train_videos.append({
            'id': video_count,
            'filename': filename,
            'color': color,
            'shape': shape,
            'motion': motion,
            'size': size,
            'type': 'train'
        })
        print(f"  {video_count}/100: {filename}")

    # TEST SET: Novel combinations (20 videos)
    print("\nGenerating TEST videos (20 novel combinations)...")

    # Novel combinations that should test abstraction
    novel_combos = [
        # Seen: red circle horizontal, blue square horizontal
        # Novel: red square horizontal (new shape+color combo)
        ('red', 'square', 'horizontal', 'medium'),

        # Seen: blue square vertical, red circle vertical
        # Novel: blue circle vertical
        ('blue', 'circle', 'vertical', 'medium'),

        # Seen: green triangle rotating, red circle circular
        # Novel: green circle circular
        ('green', 'circle', 'circular', 'large'),

        # Seen: red circle diagonal
        # Novel: blue triangle diagonal
        ('blue', 'triangle', 'diagonal', 'medium'),

        # Completely novel: yellow triangle zigzag
        ('yellow', 'triangle', 'zigzag', 'small'),

        # Novel: purple triangle vertical
        ('purple', 'triangle', 'vertical', 'large'),

        # Novel size combinations
        ('red', 'circle', 'horizontal', 'large'),
        ('blue', 'square', 'circular', 'small'),
        ('green', 'square', 'horizontal', 'medium'),
        ('yellow', 'circle', 'zigzag', 'medium'),

        # More novel combinations
        ('red', 'triangle', 'vertical', 'small'),
        ('blue', 'circle', 'zigzag', 'large'),
        ('green', 'square', 'vertical', 'small'),
        ('yellow', 'square', 'circular', 'large'),
        ('purple', 'circle', 'diagonal', 'small'),
        ('red', 'square', 'circular', 'large'),
        ('blue', 'triangle', 'horizontal', 'small'),
        ('green', 'circle', 'zigzag', 'medium'),
        ('yellow', 'triangle', 'vertical', 'medium'),
        ('purple', 'square', 'rotating', 'large'),
    ]

    for color, shape, motion, size in novel_combos:
        video_count += 1
        filename = f"test_{video_count:03d}_{color}_{shape}_{motion}_{size}.mp4"
        path = os.path.join(output_dir, filename)
        generate_video(path, color, shape, motion, size)
        test_videos.append({
            'id': video_count,
            'filename': filename,
            'color': color,
            'shape': shape,
            'motion': motion,
            'size': size,
            'type': 'test'
        })
        print(f"  {video_count}/100: {filename}")

    # Save metadata
    metadata = {
        'total_videos': video_count,
        'train_videos': len(train_videos),
        'test_videos': len(test_videos),
        'colors': list(COLORS.keys()),
        'shapes': SHAPES,
        'motions': MOTIONS,
        'sizes': list(SIZES.keys()),
        'videos': train_videos + test_videos
    }

    metadata_path = os.path.join(output_dir, 'dataset_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print()
    print("=" * 70)
    print("✓ ALL 100 VIDEOS GENERATED!")
    print("=" * 70)
    print(f"\nDataset saved in: ./{output_dir}/")
    print(f"Metadata saved: {metadata_path}")
    print("\nDataset breakdown:")
    print(f"  Training videos: {len(train_videos)} (familiar combinations)")
    print(f"  Test videos: {len(test_videos)} (novel combinations)")
    print()
    print("Training patterns:")
    print("  - Red circles: Multiple motions (learn 'red circle' concept)")
    print("  - Blue squares: Multiple motions (learn 'blue square' concept)")
    print("  - Green triangles: Rotating (learn 'green triangle' concept)")
    print("  - Yellow/Purple: Various combinations")
    print()
    print("Test patterns (novel combinations):")
    print("  - Familiar properties in NEW combinations")
    print("  - Tests if AI learned COLOR, SHAPE, MOTION separately")
    print("  - Tests compositional generalization")
    print()
    print("Video specs:")
    print("  - Resolution: 64x64 pixels")
    print("  - FPS: 30")
    print("  - Duration: 10 seconds (300 frames each)")
    print("  - Total frames: 30,000")
    print("  - Format: MP4")
    print()
    print("Next: Run video_learning_test.py to train and validate!")
    print("=" * 70)


if __name__ == "__main__":
    generate_dataset()