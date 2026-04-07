import multiprocessing as mp
from multiprocessing import shared_memory
import time
from typing import Any, Dict
import numpy as np
import cv2

def get_multiprocessing_info(verbose: bool = True):
    if verbose: print(f"Available start methods: {mp.get_all_start_methods()}")
    return mp.get_start_method()

class ImagePublishProcess:
    def __init__(self, camera_configs: Dict[str, Any], image_dt: float, zmq_port: int = 5555, start_method: str = "spawn", verbose: bool = False):
        self.camera_configs = camera_configs
        self.image_dt = image_dt
        self.zmq_port = zmq_port # Porta 5555 configurada
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

            # SOLUÇÃO PARA O LEROBOT (.mp4): 
            # Todas as memórias forçadas para 3 Canais (RGB) e 8-bits!
            size = height * width * 3
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

                if 'depth' in camera_name.lower():
                    # =========================================================
                    # 🌟 MÁGICA DA COR (IDÊNTICA À REALSENSE D435i)
                    # =========================================================
                    # O simulador (MuJoCo) manda a distância em metros.
                    # Limitamos até 2.0 metros (2000mm) para bater com a escala real.
                    depth_clipped = np.clip(image, 0.0, 2.0)
                    
                    # Converte de 2.0m para a escala de 8-bits (0 a 255)
                    depth_8bit = (depth_clipped * (255.0 / 2.0)).astype(np.uint8)
                    
                    # Aplica o mapa de calor (Vermelho = perto, Azul = longe)
                    depth_colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
                    
                    # Converte BGR (gerado pelo OpenCV) para RGB para manter o padrão LeRobot
                    processed_img = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
                        
                elif 'ir_' in camera_name.lower():
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                        processed_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    else:
                        processed_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                        
                else:
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        # Convertendo a imagem normal RGB para BGR para o OpenCV empacotar certo
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
        # =========================================================
        # CORREÇÃO DE PACOTE NO MULTIPROCESSING 
        # =========================================================
        import sys
        import os
        import time
        import numpy as np

        # 1. Força a pasta raiz do simulador a entrar no path do novo processo
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(current_dir)
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)

        # 2. Faz o import absoluto do utils (agora o Python acha com certeza!)
        from sim.sensor_utils import ImageUtils, SensorServer
        
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
                        timestamps = {name: time.time() for name in image_copies.keys()}
                        
                        # Codifica as imagens dinamicamente baseando-se no nome que o MuJoCo enviar
                        encoded_images = {}
                        for camera_name, image_copy in image_copies.items():
                            encoded_images[camera_name] = ImageUtils.encode_image(image_copy)
                            
                        serialized_data = {"timestamps": timestamps, "images": encoded_images}
                        sensor_server.send_message(serialized_data)
                    except Exception as e: 
                        print(f"Error publishing images: {e}")
                else:
                    time.sleep(0.001)
        finally:
            for shm in shm_blocks.values(): 
                shm.close()
            sensor_server.stop_server()