from PIL import Image
import numpy.typing as npt
from typing import Union

import grpc
import detection_pb2_grpc as comms_grpc
import detection_pb2 as comms
import numpy as np

CHUNK_SIZE = 1024

class ServerHandler:
    def __init__(self):
        # Set up gRPC connection
        channel = grpc.insecure_channel('localhost:50051')
        self.stub = comms_grpc.DetectionAndDescriptionStub(channel)


    def unpack_boundingboxes(self, coordinates_list):
        bounding_boxes = []
        for coordinates in coordinates_list:
            # Rescale the coordinates using the scaling_ratio
            x1 = coordinates.x1
            y1 = coordinates.y1
            x2 = coordinates.x2
            y2 = coordinates.y2
            print(f"[CLIENT] Coordinates: x1={coordinates.x1}, y1={coordinates.y1}, "
                f"x2={coordinates.x2}, y2={coordinates.y2}")
            # Append the rescaled bounding box to the list
            bounding_boxes.append([x1, y1, x2, y2])
        return bounding_boxes
    

    def generate_description_request(self, image: Image, mask: npt.ArrayLike, bbox: npt.ArrayLike):
        try:
            height, width, pixel_data, mask_data, num_chunks = self.prepare_request_data(image, mask)
            coordinates = comms.Coordinates(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])
            for i in range(num_chunks):
                start = i * CHUNK_SIZE
                end = start + CHUNK_SIZE
                yield comms.DescriptionRequest(
                            height = height, width = width, 
                            image = pixel_data[start:end],
                            mask = mask_data[start:end],
                            coords = coordinates)
        except Exception as ex:
            print(f"[CLIENT] An exception of type {type(ex).__name__} occurred. Arguments:\n{ex.args}")


    def generate_detection_request(self, image: Image, mask: npt.ArrayLike):
        try:
            height, width, pixel_data, mask_data, num_chunks = self.prepare_request_data(image, mask)
            for i in range(num_chunks):
                start = i * CHUNK_SIZE
                end = start + CHUNK_SIZE
                yield comms.DetectionRequest(
                            height = height, width = width,
                            image = pixel_data[start:end], 
                            mask = mask_data[start:end])
        except Exception as ex:
            print(f"[CLIENT] An exception of type {type(ex).__name__} occurred. Arguments:\n{ex.args}")

    def prepare_request_data(self, image: Image, mask: npt.ArrayLike):
        try:
            image_array = np.array(image)
            height, width = image_array.shape[:2]

            pixel_data = image_array.tobytes()
            mask_data = mask.tobytes()
            num_chunks = len(pixel_data) // CHUNK_SIZE + (1 if len(pixel_data) % CHUNK_SIZE else 0)
            return height, width, pixel_data, mask_data, num_chunks
        except Exception as ex:
                template = "[CLIENT] An exception of type {0} occurred. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                print(message)


    def request_detection(self, image: Image, mask: npt.ArrayLike):
        request = self.generate_detection_request(image, mask)
        response = self.stub.GetBoundingBoxes(request)
        if response.status.success == comms.Status.SUCCESS:
            return self.unpack_boundingboxes(response.coordinates_list)
        else:
            raise ValueError(response.ResponseStatus.err_message)


    def request_description(self, image: Image, mask: npt.ArrayLike, bbox):
        request = self.generate_description_request(image, mask, bbox)
        response = self.stub.GetDescription(request) 
        if response.status.success == comms.Status.SUCCESS:
            print(f"[CLIENT] Bounding box ({bbox[0]},{bbox[1]})  ({bbox[2]},{bbox[3]})")
            for conf in response.confidence_list:
                print(f"{conf.name}:{conf.confidence}")
        else:   
            raise ValueError(response.ResponseStatus.err_message)
