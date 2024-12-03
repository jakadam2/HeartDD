from PIL import Image
import numpy.typing as npt
import grpc
import detection_pb2_grpc as comms_grpc
import detection_pb2 as comms
import numpy as np
from config import parser

SERVER_IP = parser.get("DEFAULT", "server_ip")
SERVER_PORT = parser.get("DEFAULT", "server_port")
CHUNK_SIZE = parser.getint("DEFAULT", "request_chunk_size")
HEALTH_CHECK_LIMIT = parser.getint("DEFAULT", "health_check_limit")

stub = None

def connect():
    global stub
    # Set up gRPC connection
    address = f"{SERVER_IP}:{SERVER_PORT}"
    channel = grpc.insecure_channel(address)
    stub = comms_grpc.DetectionAndDescriptionStub(channel)


def unpack_boundingboxes(coordinates_list):
    bounding_boxes = []
    for coordinates in coordinates_list:
        # Rescale the coordinates using the scaling_ratio
        x1 = coordinates.x1
        y1 = coordinates.y1
        x2 = coordinates.x2
        y2 = coordinates.y2
        # Append the rescaled bounding box to the list
        bounding_boxes.append([x1, y1, x2, y2])
    return bounding_boxes

def unpack_description(server_response):
    confidence_list = []
    for item in server_response:
        confidence = {}
        for entry in item.entries:
            confidence[entry.name] = entry.confidence
        confidence_list.append(confidence)
    return confidence_list


def generate_description_request(image: Image, mask: npt.ArrayLike, bboxes: npt.ArrayLike):
    try:
        height, width, pixel_data, mask_data, num_chunks = prepare_request_data(image, mask)
        coordinate_list = []
        for bbox in bboxes:
            coordinate_list.append(comms.Coordinates(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]))
        for i in range(num_chunks):
            start = i * CHUNK_SIZE
            end = start + CHUNK_SIZE
            yield comms.DescriptionRequest(
                height=height, width=width,
                image=pixel_data[start:end],
                mask=mask_data[start:end],
                coords=coordinate_list)
    except Exception as ex:
        print(f"An exception of type {type(ex).__name__} occurred. Arguments:\n{ex.args}")


def generate_detection_request(image: Image, mask: npt.ArrayLike):
    try:
        height, width, pixel_data, mask_data, num_chunks = prepare_request_data(image, mask)
        for i in range(num_chunks):
            start = i * CHUNK_SIZE
            end = start + CHUNK_SIZE
            yield comms.DetectionRequest(
                height=height, width=width,
                image=pixel_data[start:end],
                mask=mask_data[start:end])
    except Exception as ex:
        print(f"An exception of type {type(ex).__name__} occurred. Arguments:\n{ex.args}")


def prepare_request_data(image: Image, mask: npt.ArrayLike):
    try:
        image_array = np.array(image)
        height, width = image_array.shape[:2]

        pixel_data = image_array.tobytes()
        mask_data = mask.tobytes()
        num_chunks = len(pixel_data) // CHUNK_SIZE + (1 if len(pixel_data) % CHUNK_SIZE else 0)
        return height, width, pixel_data, mask_data, num_chunks
    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)


def request_detection(image: Image, mask: npt.ArrayLike):
    request = generate_detection_request(image, mask)
    response = stub.GetBoundingBoxes(request)
    if response.status.success == comms.Status.SUCCESS:
        return unpack_boundingboxes(response.coordinates_list)
    else:
        raise ValueError(response.ResponseStatus.err_message)


def request_description(image: Image, mask: npt.ArrayLike, bboxes: npt.ArrayLike):
    request = generate_description_request(image, mask, bboxes)
    response = stub.GetDescription(request)
    if response.status.success == comms.Status.SUCCESS:
        # Iterate over each Confidence object in the response
        return unpack_description(response.confidence_list)
    else:
        raise RuntimeError(response.ResponseStatus.err_message)
