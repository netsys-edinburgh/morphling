package com.example.confidant.grpcAPI;

import com.example.confidant.globalStates.Common;
import com.example.confidant.rpc.api.GreeterGrpc;
import com.example.confidant.rpc.api.HelloReply;
import com.example.confidant.rpc.api.HelloRequest;

import io.grpc.stub.StreamObserver;

public class HelloworldAPI extends GreeterGrpc.GreeterImplBase {

    @Override
    public void sayHello(HelloRequest request, StreamObserver<HelloReply> responseObserver) {
        HelloReply helloReply = HelloReply.newBuilder()
                .setMessage("Hello " + request.getName() + ", Welcome to the RPC world!")
                .build();
        responseObserver.onNext(helloReply);
        responseObserver.onCompleted();

        Common.printLog("Reply from client: " + helloReply.getMessage());
    }

}
