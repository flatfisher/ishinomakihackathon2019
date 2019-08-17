# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A demo to classify Raspberry Pi camera stream."""

import argparse
import io
import time

from edgetpu.classification.engine import ClassificationEngine
from edgetpu.utils import dataset_utils
import numpy as np
import picamera

# orverlay
o = None

def displayImage(code:int):
  if global o == None:
    camera.remove_overlay(o)
    global o = None

  if code == 0:
    global o = camera.add_overlay('./images/demo1.png')
  elif code == 1:
    global o = camera.add_overlay('./images/demo2.png')
  elif code == 2:
    global o = camera.add_overlay('./images/demo3.png')
  elif code == 3:
    global o = camera.add_overlay('./images/demo4.png')

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model', help='File path of Tflite model.', required=True)
  parser.add_argument('--label', help='File path of label file.', required=True)
  args = parser.parse_args()

  labels = dataset_utils.ReadLabelFile(args.label)
  engine = ClassificationEngine(args.model)

  with picamera.PiCamera() as camera:
    camera.resolution = (640, 480)
    camera.framerate = 30
    _, height, width, _ = engine.get_input_tensor_shape()
    camera.start_preview()
    try:
      stream = io.BytesIO()
      for _ in camera.capture_continuous(
          stream, format='rgb', use_video_port=True, resize=(width, height)):
        stream.truncate()
        stream.seek(0)
        input_tensor = np.frombuffer(stream.getvalue(), dtype=np.uint8)
        start_ms = time.time()
        results = engine.ClassifyWithInputTensor(input_tensor, top_k=1)
        elapsed_ms = time.time() - start_ms
        if results:
         #print(results[0][1])
         #camera.annotate_text = displayString(results[0][0])
         print(displayImage(results[0][0]))
         camera.annotate_text = '%s %.2f\n%.2fms' % (
            displayImage(results[0][0]), results[0][1], elapsed_ms * 1000.0)
    finally:
      camera.stop_preview()


if __name__ == '__main__':
  main()
