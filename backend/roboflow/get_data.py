from roboflow import Roboflow
rf = Roboflow(api_key="Jt4udXThDcxb1hqWB130")
project = rf.workspace("test-mm1q9").project("skinsafeai-3mupa")
version = project.version(2)
dataset = version.download("yolov8")
                