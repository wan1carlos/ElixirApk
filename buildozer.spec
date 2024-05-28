[app]
title = Face Color Detector
package.name = facecolordetector
package.domain = org.example
source.dir = .
source.include_exts = py,png,jpg,kv,atlas
version = 0.1
requirements = python3==3.7.6,hostpython3.7.6,pillow,kivy,cv2,numpy,google
osx.python_version = 3.7.6
osx.kivy_version = 1.9.1

[buildozer]
log_level = 2
warn_on_root = 1
