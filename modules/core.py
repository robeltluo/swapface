import os
import sys

# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
from typing import List, Any
import platform
import signal
import shutil
import argparse
import torch
import onnxruntime
import tensorflow

import modules.globals
import modules.metadata
# import modules.ui as ui
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, \
    get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path

if 'ROCMExecutionProvider' in modules.globals.execution_providers:
    del torch

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser()
    program.add_argument('-s', '--source', help='select an source image', dest='source_path')
    program.add_argument('-t', '--target', help='select an target image or video', dest='target_path')
    program.add_argument('-o', '--output', help='select output file or directory', dest='output_path')
    program.add_argument('--frame-processor', help='pipeline of frame processors', dest='frame_processor',
                         default=['face_swapper'], choices=['face_swapper', 'face_enhancer'], nargs='+')
    program.add_argument('--keep-fps', help='keep original fps', dest='keep_fps', action='store_true', default=False)
    program.add_argument('--keep-audio', help='keep original audio', dest='keep_audio', action='store_true',
                         default=True)
    program.add_argument('--keep-frames', help='keep temporary frames', dest='keep_frames', action='store_true',
                         default=False)
    program.add_argument('--many-faces', help='process every face', dest='many_faces', action='store_true',
                         default=False)
    program.add_argument('--video-encoder', help='adjust output video encoder', dest='video_encoder', default='libx264',
                         choices=['libx264', 'libx265', 'libvpx-vp9'])
    program.add_argument('--video-quality', help='adjust output video quality', dest='video_quality', type=int,
                         default=18, choices=range(52), metavar='[0-51]')
    program.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory', type=int,
                         default=suggest_max_memory())
    program.add_argument('--execution-provider', help='execution provider', dest='execution_provider', default=['cpu'],
                         choices=suggest_execution_providers(), nargs='+')
    program.add_argument('--execution-threads', help='number of execution threads', dest='execution_threads', type=int,
                         default=suggest_execution_threads())
    program.add_argument('-v', '--version', action='version',
                         version=f'{modules.metadata.name} {modules.metadata.version}')

    # register deprecated args
    program.add_argument('-f', '--face', help=argparse.SUPPRESS, dest='source_path_deprecated')
    program.add_argument('--cpu-cores', help=argparse.SUPPRESS, dest='cpu_cores_deprecated', type=int)
    program.add_argument('--gpu-vendor', help=argparse.SUPPRESS, dest='gpu_vendor_deprecated')
    program.add_argument('--gpu-threads', help=argparse.SUPPRESS, dest='gpu_threads_deprecated', type=int)

    args = program.parse_args()

    modules.globals.source_path = args.source_path
    modules.globals.target_path = args.target_path
    modules.globals.output_path = normalize_output_path(modules.globals.source_path, modules.globals.target_path,
                                                        args.output_path)
    modules.globals.frame_processors = args.frame_processor
    modules.globals.headless = args.source_path or args.target_path or args.output_path
    modules.globals.keep_fps = True
    modules.globals.keep_audio = True
    modules.globals.keep_frames = True
    modules.globals.many_faces = args.many_faces
    modules.globals.video_encoder = args.video_encoder
    modules.globals.video_quality = args.video_quality
    modules.globals.max_memory = args.max_memory
    modules.globals.execution_providers = decode_execution_providers(args.execution_provider)
    modules.globals.execution_threads = args.execution_threads

    # for ENHANCER tumbler:
    if 'face_enhancer' in args.frame_processor:
        modules.globals.fp_ui['face_enhancer'] = True
    else:
        modules.globals.fp_ui['face_enhancer'] = False

    modules.globals.nsfw = False

    # translate deprecated args
    if args.source_path_deprecated:
        print('\033[33mArgument -f and --face are deprecated. Use -s and --source instead.\033[0m')
        modules.globals.source_path = args.source_path_deprecated
        modules.globals.output_path = normalize_output_path(args.source_path_deprecated, modules.globals.target_path,
                                                            args.output_path)
    if args.cpu_cores_deprecated:
        print('\033[33mArgument --cpu-cores is deprecated. Use --execution-threads instead.\033[0m')
        modules.globals.execution_threads = args.cpu_cores_deprecated
    if args.gpu_vendor_deprecated == 'apple':
        print('\033[33mArgument --gpu-vendor apple is deprecated. Use --execution-provider coreml instead.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['coreml'])
    if args.gpu_vendor_deprecated == 'nvidia':
        print('\033[33mArgument --gpu-vendor nvidia is deprecated. Use --execution-provider cuda instead.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['cuda'])
    if args.gpu_vendor_deprecated == 'amd':
        print('\033[33mArgument --gpu-vendor amd is deprecated. Use --execution-provider cuda instead.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['rocm'])
    if args.gpu_threads_deprecated:
        print('\033[33mArgument --gpu-threads is deprecated. Use --execution-threads instead.\033[0m')
        modules.globals.execution_threads = args.gpu_threads_deprecated

    print("source_path: ", modules.globals.source_path)
    print("target_path: ", modules.globals.target_path)
    print("output_path: ", modules.globals.output_path)
    print("frame_processors: ", modules.globals.frame_processors)
    print("keep_fps: ", modules.globals.keep_fps)
    print("keep_audio: ", modules.globals.keep_audio)
    print("keep_frames: ", modules.globals.keep_frames)
    print("many_faces: ", modules.globals.many_faces)
    print("video_encoder: ", modules.globals.video_encoder)
    print("video_quality: ", modules.globals.video_quality)
    print("keep_audio: ", modules.globals.keep_audio)
    print("max_memory: ", modules.globals.max_memory)
    print("execution_providers: ", modules.globals.execution_providers)
    print("execution_threads: ", modules.globals.execution_threads)
    print("headless: ", modules.globals.headless)
    print("log_level: ", modules.globals.log_level)
    print("fp_ui: ", modules.globals.fp_ui)
    print("nsfw: ", modules.globals.nsfw)
    print("camera_input_combobox: ", modules.globals.camera_input_combobox)
    print("webcam_preview_running: ", modules.globals.webcam_preview_running)


def set_params(data):
    modules.globals.source_path = data["source_path"]
    modules.globals.target_path = data["target_path"]
    modules.globals.output_path = normalize_output_path(modules.globals.source_path, modules.globals.target_path,
                                                        data["output_path"])
    modules.globals.headless = modules.globals.source_path or modules.globals.target_path or modules.globals.output_path
    modules.globals.frame_processors = data.get("frame_processors", ["face_swapper"])
    modules.globals.keep_fps = data.get("keep_fps", True)
    modules.globals.keep_audio = data.get("keep_audio", True)
    modules.globals.keep_frames = data.get("keep_frames", True)
    modules.globals.many_faces = data.get("many_faces", False)
    modules.globals.video_encoder = data.get("video_encoder", "libx264")
    modules.globals.video_quality = data.get("video_quality", 18)
    modules.globals.max_memory = data.get("max_memory", 16)
    modules.globals.execution_providers = data.get("execution_providers", ["CPUExecutionProvider"])
    modules.globals.execution_threads = data.get("execution_threads", 4)
    modules.globals.log_level = data.get("log_level", "error")

    print("source_path: ", modules.globals.source_path)
    print("target_path: ", modules.globals.target_path)
    print("output_path: ", modules.globals.output_path)
    print("frame_processors: ", modules.globals.frame_processors)
    print("keep_fps: ", modules.globals.keep_fps)
    print("keep_audio: ", modules.globals.keep_audio)
    print("keep_frames: ", modules.globals.keep_frames)
    print("many_faces: ", modules.globals.many_faces)
    print("video_encoder: ", modules.globals.video_encoder)
    print("video_quality: ", modules.globals.video_quality)
    print("keep_audio: ", modules.globals.keep_audio)
    print("max_memory: ", modules.globals.max_memory)
    print("execution_providers: ", modules.globals.execution_providers)
    print("execution_threads: ", modules.globals.execution_threads)
    print("headless: ", modules.globals.headless)
    print("log_level: ", modules.globals.log_level)
    print("fp_ui: ", modules.globals.fp_ui)
    print("nsfw: ", modules.globals.nsfw)
    print("camera_input_combobox: ", modules.globals.camera_input_combobox)
    print("webcam_preview_running: ", modules.globals.webcam_preview_running)


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(),
                                                                     encode_execution_providers(
                                                                         onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]


def suggest_max_memory() -> int:
    if platform.system().lower() == 'darwin':
        return 4
    return 16


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    if 'DmlExecutionProvider' in modules.globals.execution_providers:
        return 1
    if 'ROCMExecutionProvider' in modules.globals.execution_providers:
        return 1
    return 4


def limit_resources() -> None:
    # prevent tensorflow memory leak
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tensorflow.config.experimental.set_memory_growth(gpu, True)
    # limit memory usage
    if modules.globals.max_memory:
        memory = modules.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = modules.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def release_resources() -> None:
    if 'CUDAExecutionProvider' in modules.globals.execution_providers:
        torch.cuda.empty_cache()


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    if not shutil.which('ffmpeg'):
        update_status('ffmpeg is not installed.')
        return False
    return True


def update_status(message: str, scope: str = 'DLC.CORE') -> None:
    print(f'[{scope}] {message}')
    if not modules.globals.headless:
        ui.update_status(message)


def start(execution_id=None) -> None:
    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        if not frame_processor.pre_start():
            return
    # process image to image
    if has_image_extension(modules.globals.target_path):
        if modules.globals.nsfw == False:
            from modules.predicter import predict_image
            if predict_image(modules.globals.target_path):
                destroy()
        shutil.copy2(modules.globals.target_path, modules.globals.output_path)
        for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
            update_status('Progressing...', frame_processor.NAME)
            frame_processor.process_image(modules.globals.source_path, modules.globals.output_path,
                                          modules.globals.output_path)
            release_resources()
        if is_image(modules.globals.target_path):
            update_status('Processing to image succeed!')
            write_mark_file(execution_id, isSuccess=True)
        else:
            update_status('Processing to image failed!')
            write_mark_file(execution_id, isSuccess=False)
        return
    # process image to videos
    if modules.globals.nsfw == False:
        from modules.predicter import predict_video
        if predict_video(modules.globals.target_path):
            destroy()
    update_status('Creating temp resources...')
    create_temp(modules.globals.target_path)
    update_status('Extracting frames...')
    extract_frames(modules.globals.target_path)
    temp_frame_paths = get_temp_frame_paths(modules.globals.target_path)
    print("temp_frame_paths:", ",".join(temp_frame_paths))
    print("modules.globals.frame_processors", " ".join(modules.globals.frame_processors))
    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        update_status('Progressing...', frame_processor.NAME)
        frame_processor.process_video(modules.globals.source_path, temp_frame_paths)
        release_resources()
    # handles fps
    if modules.globals.keep_fps:
        update_status('Detecting fps...')
        fps = detect_fps(modules.globals.target_path)
        update_status(f'Creating video with {fps} fps...')
        create_video(modules.globals.target_path, fps)
    else:
        update_status('Creating video with 30.0 fps...')
        create_video(modules.globals.target_path)
    # handle audio
    if modules.globals.keep_audio:
        if modules.globals.keep_fps:
            update_status('Restoring audio...')
        else:
            update_status('Restoring audio might cause issues as fps are not kept...')
        restore_audio(modules.globals.target_path, modules.globals.output_path)
    else:
        move_temp(modules.globals.target_path, modules.globals.output_path)
    # clean and validate
    # clean_temp(modules.globals.target_path)
    if is_video(modules.globals.target_path):
        update_status('Processing to video succeed!')
        write_mark_file(execution_id, isSuccess=True)
    else:
        update_status('Processing to video failed!')
        write_mark_file(execution_id, isSuccess=False)


def write_mark_file(execution_id, isSuccess=False) -> None:
    if not execution_id:
        return
    markFile = get_mark_file_name(execution_id, isSuccess)
    os.makedirs(os.path.dirname(markFile), exist_ok=True)
    print("markFile:", markFile)
    with open(markFile, 'w') as file:
        print("markFile created:", markFile)
        pass


def get_mark_file_name(execution_id, success=False) -> str:
    path = os.path.dirname(modules.globals.output_path)
    name = os.path.basename(modules.globals.output_path)
    if success:
        return f"{path}/result/{execution_id}_success"
    else:
        return f"{path}/result/{execution_id}_fail"


def getExecutionStatus(executionId) -> Any:
    successFile = get_mark_file_name(executionId, success=True)
    failFile = get_mark_file_name(executionId, success=False)
    if os.path.exists(successFile) or os.path.exists(failFile):
        return {"code": 200, "msg": "success", "data": {"status": "finished"}}
    else:
        return {"code": 200, "msg": "success", "data": {"status": "isRunning"}}


def destroy() -> None:
    if modules.globals.target_path:
        clean_temp(modules.globals.target_path)
    quit()


def run() -> None:
    parse_args()
    if not pre_check():
        return
    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        if not frame_processor.pre_check():
            return
    limit_resources()
    if modules.globals.headless:
        start()
    else:
        window = ui.init(start, destroy)
        window.mainloop()
