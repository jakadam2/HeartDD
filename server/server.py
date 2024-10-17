
# image_detection_pb2_grpc.py and image_detection_pb2.py should be generated from your .proto file using protoc

import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import grpc
from concurrent import futures
import detection_grpc.detection_pb2_grpc as comms_grpc
import detection_grpc.detection_pb2 as comms
from PIL import Image
import io
import numpy as np

class DetectionAndDescriptionServicer(comms_grpc.DetectionAndDescriptionServicer):
    def unpack_request(self, request_iterator):
        pass        


    def GetBoundingBoxes(self, request_iterator, context):
        print("Request inboud")
        image_data = io.BytesIO()  # To accumulate image bytes
        width = None
        height = None
        file_type = None

        for request in request_iterator:
            if not file_type:
                file_type = request.file_type
            if not width:
                width = request.width
            if not height:
                height = request.height
            
            # Write the incoming image bytes to the BytesIO object
            image_data.write(request.image)
        # At this point, the whole image is received in 'image_data'
        image_data.seek(0)  # Reset the buffer's position to the start
       
        response = None
        try:
            #Load image data into actual image
            pixel_array = np.frombuffer(image_data.getvalue(), dtype=np.uint16)
            pixel_array = pixel_array.reshape((height, width))  # Reshape to 2D array
            image = Image.fromarray(pixel_array)


            # Create and return the response
            coordinates_list = []
            status = comms.ResponseStatus(success = comms.Status.SUCCESS)
            response = comms.DetectionResponse(status =status, coordinates_list = coordinates_list)

            #find bounding boxes, this is a placeholder
            bounding_boxes = self.detect_bounding_boxes(image)
            for box in bounding_boxes:
                coordinates = comms.Coordinates(x1=box['x1'], y1=box['y1'], x2=box['x2'], y2=box['y2'])
                response.coordinates_list.append(coordinates)
            ###################

        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            return comms.DetectionResponse(
                comms.ResponseStatus(
                    success = comms.Status.FAILURE,
                    err_message = "Something went wrong"
                ),
                coordinates_list = [])
      
        return response

    def detect_bounding_boxes(self, image):
        # Placeholder method for object detection
        # This is a stub for bounding box detection. Replace with your actual logic.
        
        # Returning mock bounding boxes for demonstration
        return [{'x1': 10, 'y1': 20, 'x2': 50, 'y2': 60},
                {'x1': 100, 'y1': 120, 'x2': 30, 'y2': 40}]



def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    comms_grpc.add_DetectionAndDescriptionServicer_to_server(DetectionAndDescriptionServicer(), server)
    server.add_insecure_port('[::]:50051')
    print("Server is starting on port 50051...")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
