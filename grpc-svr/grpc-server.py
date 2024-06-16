import os
import grpc
from concurrent import futures
import file_service_pb2
import file_service_pb2_grpc
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FileServiceServicer(file_service_pb2_grpc.FileServiceServicer):
    def ListFiles(self, request, context):
        directory_path = request.directory_path
        logging.info(f'Received ListFiles request for directory: {directory_path}')
        try:
            files = os.listdir(directory_path)
            logging.info(f'Found {len(files)} files in directory: {directory_path}')
            return file_service_pb2.FileList(files=files)
        except Exception as e:
            logging.error(f'Error listing files in directory {directory_path}: {e}')
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return file_service_pb2.FileList(files=[])

    def GetFile(self, request, context):
        file_path = request.file_path
        logging.info(f'Received GetFile request for file: {file_path}')
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            logging.info(f'Successfully read file: {file_path}')
            return file_service_pb2.FileContent(content=content)
        except Exception as e:
            logging.error(f'Error reading file {file_path}: {e}')
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return file_service_pb2.FileContent(content=b'')

def serve():
    logging.info('Starting gRPC server...')
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    file_service_pb2_grpc.add_FileServiceServicer_to_server(FileServiceServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logging.info('gRPC server started on port 50051')
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
