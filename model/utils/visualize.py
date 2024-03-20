import numpy as np
import cairo
import random

# Show temporal segmentation of a single sample
def visualize(prediction, target, num_classes, output_file=None):

    # Create one color for each action class
    colors = []
    random.seed(6969)
    for _ in range(num_classes):
        colors.append((random.random(), random.random(), random.random()))

    frames = len(prediction)
    with cairo.ImageSurface(cairo.FORMAT_ARGB32, frames, 100 + 100 + 20) as surface:
        ctx = cairo.Context(surface)

        ctx.set_source_rgb(0, 0, 0)
        ctx.rectangle(0.0, 100.0, float(frames), 20.0)
        ctx.fill()

        for frame in range(frames):
            pred, tar = int(prediction[frame]), int(target[frame])

            # ground truth
            if tar == 0:
                ctx.set_source_rgb(0, 0, 0)
            else:
                r, g, b = colors[tar]
                ctx.set_source_rgb(r, g, b)
            
            ctx.move_to(float(frame), 0.0)
            ctx.line_to(float(frame), 100.0)
            ctx.stroke()

            # result prediction
            if pred == 0:
                ctx.set_source_rgb(0, 0, 0)
            else:
                r, g, b = colors[pred]
                ctx.set_source_rgb(r, g, b)

            ctx.move_to(float(frame), 120.0)
            ctx.line_to(float(frame), 220.0)
            ctx.stroke()

        if output_file is not None:
            surface.write_to_png(output_file)

# =================================================================
# TEST
            
def random_segment(data, num_classes):
    width = random.randint(1, 200)
    start = random.randint(0, len(data) - width)
    value = random.randint(0, num_classes - 1)
    data[start:start+width] = value

if __name__ == '__main__':

    SIZE=5000
    CLASSES=10
    SEGMENTS=10

    ground_truth = np.zeros(SIZE)
    for _ in range(SEGMENTS):
        random_segment(ground_truth, CLASSES)

    prediction = np.zeros(SIZE)
    for _ in range(SEGMENTS):
        random_segment(prediction, CLASSES)

    visualize(prediction, ground_truth, CLASSES, 'data/output/test.png')