# `examples/data` — bundled reference images

Small set of grayscale test images shipped with the repository so the example
binaries can run with no command-line arguments. Format is PGM (P5, binary
8-bit grayscale); the `salt_pepper_filter` example reads them through its own
tiny parser, so no image-decoding library is required at build or run time.

## `astronaut.pgm`

- **Source:** [scikit-image's bundled `astronaut.png`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/data/astronaut.png) — a 512×512 colour photograph of NASA astronaut Eileen Collins.
- **Origin:** NASA Great Images database. NASA images are public domain in the
  United States and may be redistributed without restriction (subject to NASA's
  [media usage guidelines](https://www.nasa.gov/multimedia/guidelines/index.html)).
- **License:** Public domain.
- **Conversion to grayscale PGM:**

  ```python
  from PIL import Image
  Image.open("astronaut.png").convert("L").save("astronaut.pgm", format="PPM")
  ```

  ITU-R 601-2 luma transform (`L = 0.299 R + 0.587 G + 0.114 B`), saved as
  P5 binary PGM (512×512, 8-bit, max value 255).

This image is the modern de-facto replacement for the historically common
"Lenna" test image (which has been formally retired by IEEE and most
image-processing venues for both ethical and practical reasons).
