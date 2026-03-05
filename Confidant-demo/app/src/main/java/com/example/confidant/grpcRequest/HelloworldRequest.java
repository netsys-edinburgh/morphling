package com.example.confidant.grpcRequest;

import com.example.confidant.rpc.api.GreeterGrpc;
import com.example.confidant.rpc.api.HelloReply;
import com.example.confidant.rpc.api.HelloRequest;

import java.io.PrintWriter;
import java.io.StringWriter;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;

public class HelloworldRequest {
    public static String sayHello(String host, int port) {
        try {
            ManagedChannel channel = ManagedChannelBuilder.forAddress(host, port)
                    .usePlaintext()
                    .build();
            GreeterGrpc.GreeterBlockingStub stub = GreeterGrpc.newBlockingStub(channel);
            HelloRequest request = HelloRequest.newBuilder().setName("Hello Confidant").build();
            HelloReply reply = stub.sayHello(request);
            return reply.getMessage();
        } catch (Exception e) {
            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            e.printStackTrace(pw);
            pw.flush();
            return String.format("Failed... : %n%s", sw.toString());
        }
    }
}
