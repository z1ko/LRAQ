import numpy as np
import matplotlib.pyplot as plt
import random

def surface_to_np(surface):
    """ Transforms a Cairo surface into a numpy array. """
    im = +np.frombuffer(surface.get_data(), np.uint8)
    H,W = surface.get_height(), surface.get_width()
    im.shape = (H,W, 4) # for RGBA
    return im[:,:,:3]

def generate_n_colors(n, seed=6969):
    colors = []
    random.seed(seed)
    for _ in range(n):
        colors.append((random.random(), random.random(), random.random()))
    return colors

def show_segm(segmentation, num_classes):
    import cairo

    # Create one color for each action class
    colors = generate_n_colors(num_classes)

    frames = min(len(segmentation), 20000)
    with cairo.ImageSurface(cairo.FORMAT_ARGB32, frames, 100) as surface:
        ctx = cairo.Context(surface)

        # Background
        ctx.set_source_rgb(1.0, 1.0, 1.0)
        ctx.rectangle(0.0, 100.0, float(frames), 100.0)
        ctx.fill()

        for i, seg in enumerate(segmentation):
            seg = seg.item()
            if seg == -100:
                continue

            r, g, b = colors[seg]
            ctx.set_source_rgb(r, g, b)
            ctx.move_to(float(i), 0.0)
            ctx.line_to(float(i), 100.0)
            ctx.stroke()

        # Visualize using plot
        data = surface_to_np(surface)
        plt.imshow(data)
            


# Show temporal segmentation of a single sample
def visualize(prediction, target, num_classes, output_file=None):
    import cairo

    # Create one color for each action class
    colors = generate_n_colors(num_classes)

    frames = len(prediction)
    with cairo.ImageSurface(cairo.FORMAT_ARGB32, frames, 100 + 100 + 20) as surface:
        ctx = cairo.Context(surface)

        ctx.set_source_rgb(0, 0, 0)
        ctx.rectangle(0.0, 100.0, float(frames), 20.0)
        ctx.fill()

        for frame in range(frames):
            pred, tar = int(prediction[frame]), int(target[frame])

            # The ignore class sets everything to black
            if tar == -100:
                ctx.set_source_rgb(0, 0, 0)
                ctx.move_to(float(frame), 0.0)
                ctx.line_to(float(frame), 220.0)
                ctx.stroke()
                continue

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
        else:
            data = surface_to_np(surface)
            plt.imshow(data)

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
