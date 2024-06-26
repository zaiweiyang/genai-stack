# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import file_service_pb2 as file__service__pb2


class FileServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ListFiles = channel.unary_unary(
                '/file_service.FileService/ListFiles',
                request_serializer=file__service__pb2.DirectoryRequest.SerializeToString,
                response_deserializer=file__service__pb2.FileList.FromString,
                )
        self.GetFile = channel.unary_unary(
                '/file_service.FileService/GetFile',
                request_serializer=file__service__pb2.FileRequest.SerializeToString,
                response_deserializer=file__service__pb2.FileContent.FromString,
                )


class FileServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def ListFiles(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetFile(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_FileServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ListFiles': grpc.unary_unary_rpc_method_handler(
                    servicer.ListFiles,
                    request_deserializer=file__service__pb2.DirectoryRequest.FromString,
                    response_serializer=file__service__pb2.FileList.SerializeToString,
            ),
            'GetFile': grpc.unary_unary_rpc_method_handler(
                    servicer.GetFile,
                    request_deserializer=file__service__pb2.FileRequest.FromString,
                    response_serializer=file__service__pb2.FileContent.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'file_service.FileService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class FileService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def ListFiles(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/file_service.FileService/ListFiles',
            file__service__pb2.DirectoryRequest.SerializeToString,
            file__service__pb2.FileList.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetFile(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/file_service.FileService/GetFile',
            file__service__pb2.FileRequest.SerializeToString,
            file__service__pb2.FileContent.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
