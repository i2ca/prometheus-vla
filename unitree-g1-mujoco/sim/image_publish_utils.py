import multiprocessing as mp
from multiprocessing import shared_memory
import time
from typing import Any, Dict
import numpy as np
import cv2  # Adicionado para conversões de cor super rápidas

def get_multiprocessing_info(verbose: bool = True):
    if verbose: print(f"Available start methods: {mp.get_all_start_methods()}")
    return mp.get_start_method()

class ImagePublishProcess:
    def __init__(self, camera_configs: Dict[str, Any], image_dt: float, zmq_port: int = 5555, start_method: str = "spawn", verbose: bool = False):
        self.camera_configs = camera_configs
        self.image_dt = image_dt
        self.zmq_port = zmq_port
        self.verbose = verbose
        self.shared_memory_blocks = {}
        self.shared_memory_info = {}
        self.process = None

        self.mp_context = mp.get_context(start_method)
        self.stop_event = self.mp_context.Event()
        self.data_ready_event = self.mp_context.Event()
        self.stop_event.clear()
        self.data_ready_event.clear()

        for camera_name, camera_config in camera_configs.items():
            height = camera_config["height"]
            width = camera_config["width"]
            target_name = f"g1_{camera_name}_shm" 

            # ALOCAÇÃO DE MEMÓRIA REALISTA PARA CADA SENSOR
            if 'depth' in camera_name.lower():
                size = height * width * 2  # uint16 (Milímetros)
                shape = (height, width, 1)
                dtype = np.uint16
            elif 'ir_' in camera_name.lower():
                size = height * width * 1  # uint8 (Monocromático/Grayscale)
                shape = (height, width, 1)
                dtype = np.uint8
            else:
                size = height * width * 3  # uint8 (RGB)
                shape = (height, width, 3)
                dtype = np.uint8

            try:
                shm = shared_memory.SharedMemory(name=target_name, create=True, size=size)
            except FileExistsError:
                shm = shared_memory.SharedMemory(name=target_name)

            self.shared_memory_blocks[camera_name] = shm
            self.shared_memory_info[camera_name] = {
                "name": target_name, "size": size, "shape": shape, "dtype": dtype,
            }

    def start_process(self):
        self.process = self.mp_context.Process(
            target=self._image_publish_worker,
            args=(self.shared_memory_info, self.image_dt, self.zmq_port, self.stop_event, self.data_ready_event, self.verbose),
        )
        self.process.start()

    def update_shared_memory(self, render_caches: Dict[str, np.ndarray]):
        images_updated = 0
        for camera_name in self.camera_configs.keys():
            image_key = f"{camera_name}_image"
            if image_key in render_caches:
                image = render_caches[image_key]

                # FILTRO BLINDADO E PROCESSAMENTO DE SINAL
                if 'depth' in camera_name.lower():
                    if image.dtype == np.float32 or image.dtype == np.float64:
                        processed_img = (image * 1000.0).astype(np.uint16)
                    else:
                        processed_img = image
                    if len(processed_img.shape) == 2:
                        processed_img = processed_img[..., np.newaxis]
                        
                elif 'ir_' in camera_name.lower():
                    # Converte o RGB do simulador para a visão Infravermelha real (Grayscale)
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        processed_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                        processed_img = processed_img[..., np.newaxis]
                    else:
                        processed_img = image
                        
                else:
                    # Converte RGB nativo do MuJoCo para BGR do OpenCV (corrige o azul/laranja na fonte)
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        processed_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    else:
                        processed_img = image

                shm = self.shared_memory_blocks.get(camera_name)
                if shm is None: continue
                
                shared_array = np.ndarray(
                    self.shared_memory_info[camera_name]["shape"],
                    dtype=self.shared_memory_info[camera_name]["dtype"],
                    buffer=shm.buf,
                )
                np.copyto(shared_array, processed_img)
                images_updated += 1

        if images_updated > 0:
            self.data_ready_event.set()

    def stop(self):
        self.stop_event.set()
        if self.process and self.process.is_alive():
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join()

        for camera_name, shm in self.shared_memory_blocks.items():
            try:
                shm.close()
                shm.unlink()
            except Exception: pass
        self.shared_memory_blocks.clear()

    @staticmethod
    def _image_publish_worker(shared_memory_info, image_dt, zmq_port, stop_event, data_ready_event, verbose):
        from .sensor_utils import ImageMessageSchema, ImageUtils, SensorServer
        try:
            sensor_server = SensorServer()
            sensor_server.start_server(port=zmq_port)
            shared_arrays, shm_blocks = {}, {}
            
            for camera_name, info in shared_memory_info.items():
                shm = shared_memory.SharedMemory(name=info["name"])
                shm_blocks[camera_name] = shm
                shared_arrays[camera_name] = np.ndarray(info["shape"], dtype=info["dtype"], buffer=shm.buf)

            while not stop_event.is_set():
                if data_ready_event.wait(timeout=min(image_dt, 0.1)):
                    data_ready_event.clear()
                    try:
                        image_copies = {name: arr.copy() for name, arr in shared_arrays.items()}
                        message_dict = {"images": image_copies, "timestamps": {name: time.time() for name in image_copies.keys()}}
                        image_msg = ImageMessageSchema(timestamps=message_dict["timestamps"], images=message_dict["images"])
                        serialized_data = image_msg.serialize()

                        for camera_name, image_copy in image_copies.items():
                            if 'depth' in camera_name.lower():
                                serialized_data[f"{camera_name}"] = ImageUtils.encode_depth_image(image_copy)
                            else:
                                serialized_data[f"{camera_name}"] = ImageUtils.encode_image(image_copy)

                        sensor_server.send_message(serialized_data)
                    except Exception as e: print(f"Error publishing images: {e}")
                else:
                    time.sleep(0.001)
        finally:
            for shm in shm_blocks.values(): shm.close()
            sensor_server.stop_server()