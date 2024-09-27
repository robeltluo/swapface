import os
from typing import Any, List
import cv2
import insightface
import threading

import modules.globals
import modules.processors.frame.core
from modules.core import update_status
from modules.face_analyser import get_one_face, get_many_faces,write_detect_face
from modules.typing import Face, Frame
from modules.utilities import conditional_download, resolve_relative_path, is_image, is_video,rename_img_name

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'DLC.FACE-SWAPPER'


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://huggingface.co/hacksider/deep-live-cam/blob/main/inswapper_128_fp16.onnx'])
    return True


def pre_start() -> bool:
    if not is_image(modules.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False
    elif not get_one_face(cv2.imread(modules.globals.source_path)):
        update_status('No face in source path detected.', NAME)
        return False
    if not is_image(modules.globals.target_path) and not is_video(modules.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            # model_path = resolve_relative_path('../models/inswapper_128_fp16.onnx')
            model_path = resolve_relative_path('../models/inswapper_128_fp16.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=modules.globals.execution_providers)
    return FACE_SWAPPER


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    # return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)
    return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)


def process_frame(source_face: Face, temp_frame: Frame) -> Frame:
    if modules.globals.many_faces:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                temp_frame = swap_face(source_face, target_face, temp_frame)
    else:
        target_face = get_one_face(temp_frame)
        if target_face:
            temp_frame = swap_face(source_face, target_face, temp_frame)
    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], progress: Any = None) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    write_detect_face(source_path,"/test/temp/source_detect_face.png")
    print("write source face:", source_path)

    for temp_frame_path in temp_frame_paths:
        print("process_frames temp_frame_path:",temp_frame_path)
        temp_frame = cv2.imread(temp_frame_path)
        # 保存初始图片
        origin_frame = rename_img_name(temp_frame_path,"_origin")
        print("save origin frame: ", origin_frame)
        # cv2.imwrite(origin_frame, temp_frame)
        try:
            detect_frame = rename_img_name(temp_frame_path,"_detect")
            print("save detect frame: ", detect_frame)
            # write_detect_face(temp_frame_path,detect_frame)
            result = process_frame(source_face, temp_frame)
            cv2.imwrite(temp_frame_path, result)
        except Exception as exception:
            print(exception)
            pass
        if progress:
            progress.update(1)


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    print("process_image. source_path",source_path," target_path:",target_path," output_path:",output_path)
    write_detect_face(source_path,"/test/temp/source_detect_face.png")
    write_detect_face(target_path,"/test/temp/target_detect_face.png")
    source_face = get_one_face(cv2.imread(source_path))
    target_frame = cv2.imread(target_path)
    result = process_frame(source_face, target_frame)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    modules.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
