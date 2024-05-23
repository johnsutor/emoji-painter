# Emoji Painter
![Side by Side](./images/side-by-side.jpg)
This repo includes the code for teaching a model to paint using emojis. You can provide 
images or emojis to paint with, and it will attempt to recreate images using emojis.

Much of this code is adopted from the [Paint Transformer](https://github.com/Huage001/PaintTransformer) paper. In this code base, I treat emojis like "brushes", and use a Gumbel Softmax-based lookup to choose an emoji to paste to the canvas during training (similar to how attention uses a softmax to select keys during training).

## Demo 
Please visit [here](https://replicate.com/johnsutor/emoji-painter) for the demo application. Try adjusting the scale of the image for more detail (though, be warned, you very likely will run out of memory based on the GPU you are using).

## Caveats 
Outputs don't always look very similar to the target image. Also, the emojis are often placed at 45 degree angles. I'll work on some fine-tuning in the near future to make sure there's a fair amout of variety (perhaps with sampling) in the angle the emojis are placed at. 