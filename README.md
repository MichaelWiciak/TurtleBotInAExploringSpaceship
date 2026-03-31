# RoboNaut

> A TurtleBot that explores spacecraft modules, finds windows showing Earth and the Moon, captures images, and does some basic astronomy on the side.

[![Watch the demo](https://img.youtube.com/vi/UU7TQqW6gh0/0.jpg)](https://www.youtube.com/watch?v=UU7TQqW6gh0)

## What is this?

A robotics project where we threw a [TurtleBot 3 Burger](https://robotis.co.uk/turtle-burg.html) into a simulated spacecraft and told it to go find Earth and the Moon. The robot had to:

1. Navigate through a spacecraft with two modules
2. Figure out which module was "safe" (green status light) vs "danger" (red)
3. Hunt down windows in that module
4. Look through them to spot Earth and the Moon
5. Take photos, stitch them together, and calculate the distance between the two

The whole thing runs on ROS2 with Nav2 for navigation and a custom computer vision pipeline for detecting windows and identifying planets.

This was a group project for university - I handled the computer vision side of things (window/module detection, celestial body identification, CNN model training) and but designed the movement algorithms and other bits.

For the full technical write-up, check out [the project report](./Group30.pdf).

## The Interesting Bits

### The Spiral Search

The robot doesn't just drive around randomly. It starts at the center of the module and expands outward in a spiral pattern, sampling random points within an increasing radius. It rotates in place at each point to get a 360-degree view. This is a nice balance between coverage and efficiency, you're not wasting time in already-explored areas, but you're also not missing corners. This was crucial to complete the task within 5 minutes, which was our time limit.

### Window Detection Pipeline

Finding windows through a robot's camera is trickier than it sounds:

1. **Contour analysis** - Windows in this world are rectangular and have white frames. We find contours, filter by area and aspect ratio, and check if they're roughly quadrilaterals.

2. **Perspective correction** - Windows are rarely perfectly frontal. We use a four-point transform to unwarp them so we're looking at them head-on.

3. **Inside the window** - Once we have a clean window view, we look for planets. Stars are common and annoying, so we mask them out using a white pixel threshold and dilation.

### Camera to World Alignment

When the robot spots a window, it needs to actually go to it. We calculate the pixel offset from the camera center, convert that to degrees using the camera's field of view (`~62 degrees`), and rotate until the window is centered. Then we drive forward until the window fills enough of the frame (measured in pixel area) to know we are close enough for a good photo.

### SIFT for Duplicate Detection

Scanning the same window twice is a waste. We use SIFT (Scale-Invariant Feature Transform) to compare new window captures against previously captured ones. If the feature match ratio is high enough, we skip it.

### Real Hardware

Getting this to work on an actual TurtleBot (not just simulation) meant dealing with:

- Variable lighting conditions
- Camera differences between simulation and reality
- Physical navigation challenges (wheel slip, obstacles)
- Retraining the CNN model with real-world data

The CNN (MobileNetV2, fine-tuned) classifies planets as Earth, Moon, or other. We trained it on augmented data to handle the variety of angles and lighting we'd encounter.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         RoboNaut                            │
├─────────────────────────────────────────────────────────────┤
│  Navigation (Nav2)          │  Vision Pipeline              │
│  ├─ Path planning           │  ├─ Window detection          │
│  ├─ Pose estimation         │  ├─ Planet detection          │
│  └─ Obstacle avoidance      │  ├─ CNN classification        │
│                             │  ├─ Image stitching           │
│                             │  └─ Distance calculation      │
│                              │                               │
│  State Machine (Goals/Actions)                              │
│  ├─ Find correct module     │  HUD Overlay                  │
│  ├─ Scan for windows        │  ├─ Live map                  │
│  ├─ Capture windows         │  ├─ Detection overlays        │
│  └─ Compute measurements    │  └─ Robot telemetry           │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
.
├── group_project/          # Main ROS2 package
│   ├── robonaut.py         # Main state machine and robot logic
│   ├── cv_detect.py        # Window and status light detection
│   ├── cv_find_planets.py  # Planet separation and classification
│   ├── cv_planet_model.py  # CNN model interface
│   ├── astro_stitch.py     # Image stitching (SIFT + RANSAC)
│   ├── cv_compare.py       # SIFT-based image comparison
│   ├── distance_calculations.py  # Triangulation math
│   ├── hud.py              # Live HUD overlay
│   └── coordinates.py      # Module coordinate definitions
├── launch/                 # ROS2 launch files
├── worlds/                 # Gazebo worlds and maps
│   ├── spacecraft_easy/
│   ├── spacecraft_moderate/
│   ├── spacecraft_hard/
│   └── real_world/
├── moonDistance/           # Standalone planet detection experiments
└── supporting_files/       # CNN model weights, templates
```

## Running It

### Simulation

```bash
# Launch the world and robot
ros2 launch group_project world.launch.py world:=hard

# In another terminal, run navigation
ros2 launch group_project navigation.launch.py world:=hard

# And the main RoboNaut node
ros2 launch group_project robotnaut_go.launch.py world:=hard
```

Available worlds: `easy`, `moderate`, `hard`, `real`

### Real Hardware

The `real` world is configured for actual TurtleBot deployment. Expect some tuning may be needed depending on your setup.

## Tech Stack

- **ROS2** - Robot operating system
- **Gazebo** - Robot simulation
- **Nav2** - Navigation stack
- **OpenCV** - Computer vision (contour detection, Hough transforms, SIFT)
- **PyTorch** - CNN model (MobileNetV2)
- **Python** - All the code

## Video

Watch the robot do its thing: [RoboNaut Demo on YouTube](https://www.youtube.com/watch?v=UU7TQqW6gh0)

## More Details

For the full technical breakdown - architecture decisions, algorithm details, results, and lessons learned - see [the project report](./Group30.pdf).
