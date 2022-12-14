# CLASSIFY_CHIPS
![CLASSIFY_CHIPS.py](https://github.com/stevensmiley1989/CLASSIFY_CHIPS/blob/main/misc/CLASSIFY_CHIPS_Screenshot.png)
CLASSIFY_CHIPS.py is GUI for creating custom PyTorch/Tensorflow Image Classifier models with on Chips.

It is written in Python and uses Tkinter for its graphical interface.

Prerequisites
------------------

Ensure you put the "imagenet_resnet_v1_101_feature_vector_5" weights in the "resources" directory.  You can get these weights from:
["imagenet_resnet_v1_101_feature_vector_5"](https://tfhub.dev/google/imagenet/resnet_v1_101/feature_vector/5?tf-hub-format=compressed)

Ensure you put the "tf2-preview_mobilenet_v2_feature_vector_4" weights in the "resources" directory.  You can get these weights from:
["tf2-preview_mobilenet_v2_feature_vector_4"](https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4?tf-hub-format=compressed)

Ensure you put the "resnet50-0676ba61.pth" weights in the "resources" directory.  You can get these weights from:
["resnet50-0676ba61.pth"](https://drive.google.com/file/d/1TN96v1yxSv3PY5CM1Q37y0mdxUTLm2gL/view?usp=sharing)

Ensure your "resources" has the following file structure:
![Ensure your resources has the following file structure:](https://github.com/stevensmiley1989/CLASSIFY_CHIPS/blob/main/misc/CLASSIFY_CHIPS_Screenshot_resources.png)

------------------

Installation
------------------
~~~~~~~

Python 3 + Tkinter
.. code:: shell
    cd ~/
    python3 -m venv venv_CLASSIFY_CHIPS
    source venv_CLASSIFY_CHIPS/bin/activate
    cd CLASSIFY_CHIPS
    pip3 install -r requirements.txt

~~~~~~~

## Contact-Info<a class="anchor" id="4"></a>

Feel free to contact me to discuss any issues, questions, or comments.

* Email: [stevensmiley1989@gmail.com](mailto:stevensmiley1989@gmail.com)
* GitHub: [stevensmiley1989](https://github.com/stevensmiley1989)
* LinkedIn: [stevensmiley1989](https://www.linkedin.com/in/stevensmiley1989)
* Kaggle: [stevensmiley](https://www.kaggle.com/stevensmiley)

### License <a class="anchor" id="5"></a>
MIT License

Copyright (c) 2022 Steven Smiley

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*I am providing code and resources in this repository to you under an open source license.  Because this is my personal repository, the license you receive to my code and resources is from me and not my employer. 

