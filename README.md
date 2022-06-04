# Final-Year-Project


This project is about the fruit adulteration and fruit grading system, where the sample is taken from the database and image is processed(changing raw dataset into clean dataset).

after image processing is done, features of that particular fruit image is extracted(where data is transformed into numerical factors).

in feature extraction:
  1. GRAY SCALE CONVERSION: the image is converted into gray scale for better result and to reduce complexity.

  2. NOISE REMOVAL: then noise is removed from the image(i.e. removing random variation of brightness).

  3. THRESHOLDING / IMAGE SEGMENTATION: then image is broken in several sub-groups called image segment which also reduces complexity of image analysis.

  4. IMAGE SHARPENING: it is done to overcome blurness introduced by camera, to draw attention to certian areas and increase image qualtiy.


then the image is classified if it's edible to eat or not. Here classification is done using supervised machine algorithm.

in the final step, the disease is predicted in the fruit and output is displayed on the screen.
