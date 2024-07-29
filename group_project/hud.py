from .cv_detections import (
    StatusDetection,
    PosterDetection,
    WindowDetection,
    DetectMetadata,
    StatusEnum,
)
import cv2
from datetime import datetime
from dataclasses import dataclass
from .coordinates import get_module_coordinates


def black_only(image, threshold=0):
    b = cv2.split(image)
    b = b[0]
    b[b > threshold] = 255
    b[b <= threshold] = 0

    return cv2.merge([b])


@dataclass
class MapData:
    image: any
    map_resolution: float
    map_origin: list[float]
    map_origin_x: float
    map_origin_y: float


map_data: MapData | None = None


def load_map(map_path):

    with open(f"{map_path}/map.yaml", "r") as f:
        import yaml

        yaml_data = yaml.safe_load(f)
        map_image_path = yaml_data["image"]

        map_image = cv2.imread(f"{map_path}/{map_image_path}", -1)
        map_image = black_only(map_image)
        map_image = cv2.GaussianBlur(map_image, (5, 5), 0)
        map_image = black_only(map_image, 250)
        map_image = 255 - map_image
        map_image = cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB)

        # from .cv_detect import show
        # show(map_image, "map", wait=True)

        map_resolution = yaml_data["resolution"]
        map_origin = yaml_data["origin"]
        map_origin_x = map_origin[0]
        map_origin_y = map_origin[1]

        coordinates = get_module_coordinates(f"{map_path}/../coordinates.yaml")

        coords = [
            (coordinates.module_1.entrance.x, coordinates.module_1.entrance.y),
            (coordinates.module_2.entrance.x, coordinates.module_2.entrance.y),
            (coordinates.module_1.center.x, coordinates.module_1.center.y),
            (coordinates.module_2.center.x, coordinates.module_2.center.y),
        ]

        global map_data
        map_data = MapData(
            image=map_image,
            map_resolution=map_resolution,
            map_origin=map_origin,
            map_origin_x=map_origin_x,
            map_origin_y=map_origin_y,
        )

        for coord in coords:
            x, y = coord
            draw_coordinates(map_image, x, y)


colour_map = {
    "green": (0, 255, 0),
    "red": (38, 38, 220),
    "outline": (238, 211, 34),
    "outline2": (235, 99, 37),
    "text": (0, 0, 0),
    "text_drop": (255, 255, 255),
}


def draw_coordinates(image, x, y, colour=(36, 190, 251)):
    map_height = image.shape[0]
    map_width = image.shape[1]

    map_resolution = map_data.map_resolution
    map_origin_x = map_data.map_origin_x
    map_origin_y = map_data.map_origin_y

    offset_x = (x - map_origin_x) / map_resolution
    offset_y = (y - map_origin_y) / map_resolution

    offset_y = -offset_y + map_height

    offset_x = int(offset_x)
    offset_y = int(offset_y)

    cv2.circle(image, (offset_x, offset_y), 6, colour, -1)

    return map


def format_s(seconds: float) -> str:
    return f"{seconds * 1000:.3f}ms"


class HudData:

    def __init__(
        self,
        detections: list[StatusDetection | WindowDetection | PosterDetection],
        detect_metadata: "DetectMetadata",
        start_time: datetime = None,
        module1_status: StatusEnum | None = None,
        module2_status: StatusEnum | None = None,
        x: float = 0,
        y: float = 0,
        current_goal: str | None = None,
    ) -> None:
        self.detections = detections
        self.detect_metadata = detect_metadata
        self.module1_status: StatusEnum | None = module1_status
        self.module2_status: StatusEnum | None = module2_status
        self.start_time = start_time
        self.x = x
        self.y = y
        self.current_goal = current_goal


def get_map_with_coordinates(map, x, y, height):
    width = int(height * (map.shape[1] / map.shape[0]))
    map = map.copy()

    draw_coordinates(map, x, y, (38, 38, 220))

    resized = cv2.resize(map, (width, height))
    return resized


def draw_time(drawing, data: HudData):

    start_time = data.start_time if data.start_time is not None else datetime.now()

    now = datetime.now()
    delta = now - start_time

    height = drawing.shape[0]
    hud_height, hud_width, _, _ = get_hud_size(drawing)

    # calculations from:
    # https://stackoverflow.com/a/539360/13121213
    # auto_text(f"Time: {delta}")

    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    is_greater_than_5_minutes = delta.seconds > 60 * 5

    label = f"T+{hours:02}:{minutes:02}:{seconds:02}"

    textsize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = int((drawing.shape[1] - textsize[0]) / 2)
    text_y = int(height - hud_height + 10 + textsize[1])

    cv2.putText(
        drawing,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        colour_map["text"],
        2,
    )
    cv2.putText(
        drawing,
        label,
        (text_x + 1, text_y + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        colour_map["text_drop"] if not is_greater_than_5_minutes else colour_map["red"],
        2,
    )

    label = "Robonaut"
    font_scale = 1.75
    thickness = 2
    textsize = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, font_scale, thickness)[0]
    text_x = int((drawing.shape[1] - textsize[0]) / 2)
    text_y = text_y + textsize[1] + 12

    cv2.putText(
        drawing,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_PLAIN,
        font_scale,
        colour_map["text"],
        thickness,
    )
    cv2.putText(
        drawing,
        label,
        (text_x + 1, text_y + 1),
        cv2.FONT_HERSHEY_PLAIN,
        font_scale,
        colour_map["text_drop"],
        thickness,
    )

    label = f"x: {data.x:.2f}, y: {data.y:.2f}"
    font_scale = 1
    thickness = 1
    textsize = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, font_scale, thickness)[0]
    text_x = int((drawing.shape[1] - textsize[0]) / 2)
    text_y = text_y + textsize[1] + 12

    cv2.putText(
        drawing,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_PLAIN,
        font_scale,
        colour_map["text"],
        thickness,
    )
    cv2.putText(
        drawing,
        label,
        (text_x + 1, text_y + 1),
        cv2.FONT_HERSHEY_PLAIN,
        font_scale,
        colour_map["text_drop"],
        thickness,
    )

    target_map_height = height - text_y - 20

    if map_data is not None:
        resized = get_map_with_coordinates(
            map_data.image, data.x, data.y, target_map_height
        )
        map_height = resized.shape[0]
        map_width = resized.shape[1]
        offset_x = int((drawing.shape[1] - map_width) / 2)
        offset_y = text_y + 10

        # only copy the non-black pixels
        chunk = drawing[
            offset_y : offset_y + map_height,
            offset_x : offset_x + map_width,
        ]

        r_c, g_c, b_c = cv2.split(chunk)
        r_m, g_m, b_m = cv2.split(resized)

        r_c[r_m != 0] = r_m[r_m != 0]
        g_c[g_m != 0] = g_m[g_m != 0]
        b_c[b_m != 0] = b_m[b_m != 0]

        chunk = cv2.merge([r_c, g_c, b_c])

        drawing[
            offset_y : offset_y + map_height,
            offset_x : offset_x + map_width,
        ] = chunk

    return


def create_auto_text(image, x: int = 0, y: int = 0, align="left"):
    y_offset = y

    def auto_text(text):
        nonlocal y_offset

        textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        x_offset = (x) if align == "left" else image.shape[1] - textsize[0] - x

        cv2.putText(
            image,
            text,
            (x_offset, y_offset + textsize[1]),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            colour_map["text"],
            1,
        )
        cv2.putText(
            image,
            text,
            (x_offset + 1, y_offset + 1 + textsize[1]),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            colour_map["text_drop"],
            1,
        )
        y_offset += 15

        return (x_offset, y_offset), textsize

    return auto_text


def create_auto_image(target_image):
    x_offset = 10
    y_offset = target_image.shape[0] - 10
    max_x = target_image.shape[1]

    drawn_height = 100
    drawn_width = int(drawn_height * 1.5)

    def auto_image(image):
        image = cv2.resize(image, (drawn_width, drawn_height))
        nonlocal x_offset
        nonlocal y_offset
        target_image[
            y_offset - image.shape[0] : y_offset, x_offset : x_offset + image.shape[1]
        ] = image
        x_offset += image.shape[1] + 10
        if x_offset > max_x - drawn_width:
            x_offset = 10
            y_offset -= drawn_height - 10

    return auto_image


def get_hud_size(image):
    height = image.shape[0]
    width = image.shape[1]

    # hud is full width and 1/4 height of the image at the bottom
    hud_height = int(height / 3)
    hud_width = width

    hud_height_offset = height - hud_height
    hud_width_offset = 0

    return hud_height, hud_width, hud_height_offset, hud_width_offset


def draw_hud_background(drawing, data: HudData):
    hud = drawing.copy()

    # hud is full width and 1/4 height of the image at the bottom
    hud_height, hud_width, _, _ = get_hud_size(drawing)

    # draw a black rectangle
    cv2.rectangle(
        hud,
        (0, hud.shape[0] - (hud_height + 10)),
        (hud_width, hud.shape[0]),
        (0, 0, 0),
        -1,
    )

    opacity = 0.5
    weighted = cv2.addWeighted(drawing, 1 - opacity, hud, opacity, 0)

    # drawing[::] = weighted[::]

    blurred = cv2.GaussianBlur(weighted, (0, 0), 3)

    drawing[
        hud.shape[0] - hud_height : hud.shape[0],
        0 : hud.shape[1],
    ] = blurred[
        hud.shape[0] - hud_height : hud.shape[0],
        0 : hud.shape[1],
    ]

    pass


def colour_for_status(status: StatusEnum) -> tuple[int, int, int]:
    if status == StatusEnum.GREEN:
        return colour_map["green"]
    elif status == StatusEnum.RED:
        return colour_map["red"]
    else:
        return (128, 128, 128)


def draw_module_status(drawing, data: HudData):
    height = drawing.shape[0]
    hud_height, hud_width, _, _ = get_hud_size(drawing)

    auto_text = create_auto_text(
        drawing, x=10, y=height - hud_height + 10, align="right"
    )

    for index, module in enumerate([data.module1_status, data.module2_status]):
        offset, textsize = auto_text(f"Module {index+1}")
        colour = colour_for_status(module)

        cv2.circle(
            drawing,
            (offset[0] - 12, offset[1] - textsize[1]),
            6,
            colour,
            -1,
        )

    pass


def draw_current_goal(drawing, data: HudData):
    height = drawing.shape[0]
    hud_height, hud_width, _, _ = get_hud_size(drawing)

    auto_text = create_auto_text(
        drawing, x=10, y=height - hud_height + 50, align="right"
    )

    offset, textsize = auto_text("Current goal:")
    offset, textsize = auto_text(f"{data.current_goal}")

    pass


def draw_detections(drawing, data: HudData):
    height = drawing.shape[0]
    hud_height, hud_width, _, _ = get_hud_size(drawing)

    auto_text = create_auto_text(drawing, x=10, y=height - hud_height + 10)
    auto_image = create_auto_image(drawing)

    window_took = format_s(data.detect_metadata.window_s)
    status_light_took = format_s(data.detect_metadata.status_s)

    auto_text(f"Detect window: {window_took}")
    auto_text(f"Detect status: {status_light_took}")

    for detection in data.detections:
        auto_text(str(detection))
        if isinstance(detection, WindowDetection):
            auto_image(detection.capture_drawing)
        if isinstance(detection, PosterDetection):
            auto_image(detection.capture_drawing)

    pass


def draw_hud(drawing, data: HudData):
    draw_hud_background(drawing, data)

    draw_detections(drawing, data)
    draw_module_status(drawing, data)
    draw_current_goal(drawing, data)
    draw_time(drawing, data)

    return
