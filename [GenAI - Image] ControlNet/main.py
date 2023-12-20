from controlnet_operator import ControlNetOperator
from image_preprocessor import preprocess_image
import matplotlib.pyplot as plt
import numpy as np


# Setup arguments
device = "mps"
image_path = "./sample/samoyed.jpg"
prompt = "corgi wearing glasses"
h, w = 400, 400

if __name__ == "__main__":
    operator = ControlNetOperator(device=device)
    source_image, canny_image = preprocess_image(image_path)
    canny_image.save("./output/canny_image.png")
    output_image = operator.infer(
        image=canny_image, prompt=prompt, h=h, w=w, CFG=7.0, steps=30, seed=0
    )
    output_image.save("./output/output_image.png")

    src_np = np.array(source_image)
    canny_np = np.array(canny_image)
    output_np = np.array(output_image)

    # Create subplot
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 3), tight_layout=True)

    legends = ["Source", "Canny", "Output"]
    for i, ax in enumerate(axes):
        ax.set_title(legends[i])
        ax.axis("off")

    image1 = axes[0].imshow(src_np)
    image2 = axes[1].imshow(canny_np)
    image3 = axes[2].imshow(output_np)

    # 顯示圖像
    plt.show()
