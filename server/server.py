
# image_detection_pb2_grpc.py and image_detection_pb2.py should be generated from your .proto file using protoc

# Add the project root directory to sys.path
import grpc
from concurrent import futures
import detection_pb2_grpc as comms_grpc
import detection_pb2 as comms
from PIL import Image
import io
import numpy as np
import random
import endpoints as ep

class DetectionAndDescriptionServicer(comms_grpc.DetectionAndDescriptionServicer):

    def unpack_request(self, request_iterator, context, mode="detection"):
        image_data = io.BytesIO()  # To accumulate image bytes
        mask_data = io.BytesIO()
        width = None
        height = None
        coordinates = []

        for request in request_iterator:
            if mode == "description":
                if not coordinates:
                    coordinates = [
                        request.coords.x1,
                        request.coords.y1,
                        request.coords.x2,
                        request.coords.y2
                    ]
            if not width:
                width = request.width
            if not height:
                height = request.height
            
            # Write the incoming image bytes to the BytesIO object
            mask_data.write(request.mask)
            image_data.write(request.image)
        # At this point, the whole image is received in 'image_data'
        image_data.seek(0)  # Reset the buffer's position to the start
        mask_data.seek(0)
        try:
            bit_mask = np.frombuffer(mask_data.getvalue(), dtype=np.uint8)
            bit_mask = bit_mask.reshape((height, width))
            #Load image data into actual image
            pixel_array = np.frombuffer(image_data.getvalue(), dtype=np.uint8)
            pixel_array = pixel_array.reshape((height, width))  # Reshape to 3D array
            image = Image.fromarray(pixel_array)
            return image, bit_mask, width, height, coordinates
        except Exception as ex:
            raise ValueError(ex.args)


    def GetBoundingBoxes(self, request_iterator, context):
        print("[SERVER] Detection request inboud")
        response = None
        try:
            image, bit_mask, width, height, coordinates = self.unpack_request(request_iterator, context)
            # Create and return the response
            coordinates_list = []
            status = comms.ResponseStatus(success = comms.Status.SUCCESS)
            response = comms.DetectionResponse(status =status, coordinates_list = coordinates_list)

            #find bounding boxes, this is a placeholder
            bounding_boxes = ep.detect_bounding_boxes(image = image, mask = bit_mask)
            #bounding_boxes = self.return_test_bboxes()
            for box in bounding_boxes:
                coordinates = comms.Coordinates(x1=box['x1'], y1=box['y1'], x2=box['x2'], y2=box['y2'])
                response.coordinates_list.append(coordinates)
            ###################

        except Exception as ex:
            template = "[SERVER] An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            return comms.DetectionResponse(
                comms.ResponseStatus(
                    success = comms.Status.FAILURE,
                    err_message = "Something went wrong"
                ),
                coordinates_list = [])
      
        return response

 
    def GetDescription(self, request_iterator, context):
        print("[SERVER] Description request inboud")
        response = None
        try:
            image, bit_mask, width, height, coordinates = self.unpack_request(request_iterator, context)
            # Create and return the response
            confidence_list = []
            status = comms.ResponseStatus(success = comms.Status.SUCCESS)
            response = comms.DescriptionResponse(status =status, confidence_list = confidence_list)
            #find bounding boxes, this is a placeholder
            confidences = ep.describe_bbox(image = image,mask = bit_mask, bboxes = coordinates)
            for description in confidences:
                confidence = comms.Confidence(name=description["name"], confidence=description["confidence"])
                response.confidence_list.append(confidence)
            ###################

        except Exception as ex:
            template = "[SERVER] An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            return comms.DetectionResponse(
                comms.ResponseStatus(
                    success = comms.Status.FAILURE,
                    err_message = "Something went wrong"
                ),
                coordinates_list = [])
      
        return response

    def return_test_bboxes(self):
        image_width = 512
        image_height = 512

        # Generate two random bounding boxes within the 512x512 image dimensions
        def random_box():
            x1 = round(random.uniform(0, image_width - 1), 2)
            y1 = round(random.uniform(0, image_height - 1), 2)
            x2 = round(random.uniform(x1 + 1, image_width), 2)
            y2 = round(random.uniform(y1 + 1, image_height), 2)
            return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

        box1 = random_box()
        box2 = random_box()

        return [box1, box2]


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    comms_grpc.add_DetectionAndDescriptionServicer_to_server(DetectionAndDescriptionServicer(), server)
    server.add_insecure_port('[::]:50051')
    print("[SERVER] Server is starting on port 50051...")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
