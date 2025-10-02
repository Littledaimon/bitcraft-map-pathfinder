from PIL import Image, ImageDraw, ImageFont
import numpy as np
import heapq
import math
from collections import defaultdict

# -------- CONFIG -------- #
# todo: replace cost with speed parameter to use deployable speeds directly as input
COLOR_COST_RANGES = [
    ((0, 100, 0), (100, 255, 100), 15, "Green Biomes"), # Green-ish - higher cost
    ((30, 40, 60), (40, 60, 80), 10, "Ocean") # Blue-ish - water, relatively cheap cause fast ship
    # Add more ranges as needed
]
CROSS_ZONE_COST = 1000  # or any value you like
DEFAULT_COST = 20


# --------- HELPERS ---------- #
def color_to_cost(color):
    r, g, b = color
    for lower, upper, cost in COLOR_COST_RANGES:
        if all(lower[i] <= color[i] <= upper[i] for i in range(3)):
            return cost
    return DEFAULT_COST


def get_neighbors(pos, shape):
    x, y = pos
    h, w = shape
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),  # Cardinal directions
        (-1, -1), (-1, 1), (1, -1), (1, 1)  # Diagonal directions
    ]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < h and 0 <= ny < w:
            yield (nx, ny), (dx, dy)


def heuristic(a, b):
    # Manhattan distance
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# -------- A* PATHFINDING -------- #
import time

def astar(cost_map, start, end, zone_map=None, cross_zone_cost=5, log_interval=10000):
    h, w = cost_map.shape
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, end), 0, start))
    came_from = {}
    g_score = {start: 0}

    visited = set()
    steps = 0
    start_time = time.time()

    print(f"Starting A* pathfinding from {start} to {end}...")

    while open_set:
        _, current_cost, current = heapq.heappop(open_set)
        steps += 1

        if current == end:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            end_time = time.time()
            print(f"✅ Path found in {steps} steps, time: {end_time - start_time:.2f}s, length: {len(path)}")
            return path[::-1]

        if current in visited:
            continue
        visited.add(current)

        if steps % log_interval == 0:
            print(f"Visited {steps} nodes... still searching.")

        current_zone = zone_map[current] if zone_map is not None else -1

        for (neighbor, direction) in get_neighbors(current, (h, w)):
            step_cost = cost_map[neighbor]

            # Adjust cost if moving diagonally
            if abs(direction[0]) == 1 and abs(direction[1]) == 1:
                step_cost *= 1.4142  # Approx √2

            # Add cross-zone penalty if applicable
            if zone_map is not None:
                current_zone = zone_map[current]
                neighbor_zone = zone_map[neighbor]
                if current_zone != neighbor_zone:
                    step_cost += cross_zone_cost

            tentative_g = g_score[current] + step_cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, end)
                heapq.heappush(open_set, (f_score, tentative_g, neighbor))

    print(f"❌ No path found after {steps} steps. Time: {time.time() - start_time:.2f}s")
    return None


def generate_cost_map(pixels, color_cost_ranges, default_cost=1):
    h, w, _ = pixels.shape
    cost_map = np.full((h, w), default_cost, dtype=np.float32)
    zone_map = np.full((h, w), -1, dtype=np.int32)  # -1 = default zone

    for zone_id, (lower, upper, cost, *_) in enumerate(color_cost_ranges):
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = np.all((pixels >= lower) & (pixels <= upper), axis=2)
        cost_map[mask] = cost
        zone_map[mask] = zone_id

    return cost_map, zone_map


def cost_to_color(cost, min_cost, max_cost):
    # Normalize cost between 0 and 1
    norm = (cost - min_cost) / (max_cost - min_cost) if max_cost != min_cost else 0

    # Interpolate from green → yellow → red
    if norm < 0.5:
        # Green → Yellow
        r = int(2 * norm * 255)
        g = 255
        b = 0
    else:
        # Yellow → Red
        r = 255
        g = int((1 - 2 * (norm - 0.5)) * 255)
        b = 0
    return r, g, b


def draw_marker(coord, label, draw, image):
    x, y = coord
    radius = 3

    # Draw the green dot with black border
    draw.ellipse([(y - radius - 1, x - radius - 1), (y + radius + 1, x + radius + 1)], fill=(0, 0, 0))
    draw.ellipse([(y - radius, x - radius), (y + radius, x + radius)], fill=(0, 255, 0))

    # Use default font
    font = ImageFont.load_default()

    # Coordinates for top-left of text
    text_x = y + radius + 5
    text_y = x

    # Calculate bounding box for the label text
    bbox = draw.textbbox((text_x, text_y), label, font=font)
    # bbox = (left, top, right, bottom)

    # Create transparent overlay for background
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    padding = 2
    # Add semi-transparent black rectangle behind text
    bbox_padded = (
        bbox[0] - padding,
        bbox[1] - padding,
        bbox[2] + padding,
        bbox[3] + padding
    )

    # Draw padded background
    overlay_draw.rectangle(bbox_padded, fill=(0, 0, 0, 192))

    # Draw white text on top
    overlay_draw.text((text_x, text_y), label, font=font, fill=(255, 255, 255, 255))

    # Paste overlay onto the image using alpha channel as mask
    image.paste(overlay, (0, 0), overlay)

# -------- MAIN FUNCTION -------- #
def draw_path_on_image(
    image_path,
    start,
    end,
    output_path="output.png",
    path_width=1,
    downscale_factor=1  # Set to >1 to reduce resolution
):
    image = Image.open(image_path).convert("RGB")

    original_height = image.height
    def adjust_x_coord(coord):
        x, y = coord
        return (round(original_height - (x/1.1547005)), y)
    start = adjust_x_coord(start)
    end = adjust_x_coord(end)

    # Downscale image if needed
    if downscale_factor > 1:
        original_size = image.size
        new_size = (image.width // downscale_factor, image.height // downscale_factor)
        print(f"🔧 Downscaling image from {original_size} to {new_size}")
        image = image.resize(new_size, resample=Image.NEAREST)

        # Scale coordinates accordingly
        def scale_coord(coord):
            return (coord[0] // downscale_factor, coord[1] // downscale_factor)

        start = scale_coord(start)
        end = scale_coord(end)

    pixels = np.array(image)
    h, w, _ = pixels.shape

    # Generate cost map (vectorized)
    cost_map, zone_map = generate_cost_map(pixels, COLOR_COST_RANGES, DEFAULT_COST)

    # Pathfinding
    path = astar(cost_map, start, end, zone_map=zone_map, cross_zone_cost=CROSS_ZONE_COST)
    if not path:
        print("❌ No path found.")
        return

    # Extract costs along the path
    path_costs = [cost_map[x, y] for (x, y) in path if np.isfinite(cost_map[x, y])]
    min_cost = min(path_costs)
    max_cost = max(path_costs)

    # --- Distance by zone and zone crossing analysis ---
    zone_distances = defaultdict(float)
    zone_crossings = 0

    prev_zone = zone_map[path[0]]
    for (a, b) in zip(path[:-1], path[1:]):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        step_distance = math.sqrt(2) if dx == 1 and dy == 1 else 1

        current_zone = zone_map[b]
        zone_distances[current_zone] += step_distance

        if current_zone != prev_zone:
            zone_crossings += 1
        prev_zone = current_zone

    total_distance = sum(zone_distances.values())

    # --- Print result ---
    print("\n📏 Distance Traveled by Zone:")
    for zone_id, dist in sorted(zone_distances.items()):
        if zone_id == -1:
            cost = DEFAULT_COST
            name = "Default Zone"
        else:
            _, _, cost, name = COLOR_COST_RANGES[zone_id]
        print(f"  Zone {zone_id:>2} ({name}, cost {cost}): {dist:.2f} units")

    print(f"\n🔀 Zone Crossings: {zone_crossings}")
    print(f"🧮 Total Distance: {total_distance:.2f} units\n")

    # Draw result
    draw = ImageDraw.Draw(image)

    # Draw path with cost-based color
    for (x, y) in path:
        cost = cost_map[x, y]
        if not np.isfinite(cost):  # Skip impassable
            continue
        color = cost_to_color(cost, min_cost, max_cost)

        if path_width <= 0.5:
            draw.point((y, x), fill=color)
        else:
            draw.ellipse([(y - path_width, x - path_width), (y + path_width, x + path_width)], fill=color)

    draw_marker(start, "Start", draw, image)
    draw_marker(end, "End", draw, image)

    image.save(output_path)
    print(f"✅ Path drawn and saved to {output_path}")


# --------- EXAMPLE USAGE --------- #
if __name__ == "__main__":
    # Replace with your actual coordinates (row, col)
    start_coord = (4429, 3848)
    end_coord = (1191, 4109)
    draw_path_on_image("clean_map.png", start_coord, end_coord, "output.png", 0.75, 3)
