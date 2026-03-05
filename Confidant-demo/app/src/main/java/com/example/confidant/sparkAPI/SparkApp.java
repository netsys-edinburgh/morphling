package com.example.confidant.sparkAPI;

import android.util.Log;

import com.example.confidant.globalStates.Common;
import com.example.confidant.sparkAPI.central.CentralFaultTolerance;
import com.example.confidant.sparkAPI.central.CentralIndex;
import com.example.confidant.sparkAPI.worker.WorkerFaultTolerance;
import com.example.confidant.sparkAPI.worker.WorkerIndex;
import com.example.confidant.sparkAPI.worker.WorkerProactiveFaultTolerance;

import spark.Spark;


public class SparkApp {

    public static final String tag = "TrainSpark";
    public static void main(int role) {
        String roleName = role == 0 ? "Centrl Node" : "Worker Node";
        int port = Common.getHttpPort(); // the default port is set to 50000
        Common.printLog(String.format("Initializing the Spark Service as a %s on port %d", roleName, port));
        Spark.port(port);

        if (role == 0) {
            // Central Node
            Log.i(tag, "Defining central routes");
            CentralIndex.defineRoutes();
            CentralFaultTolerance.defineRoutes();
        } else {
            // Worker node
            Log.i(tag, "Defining worker routes");
            WorkerIndex.defineRoutes();
            WorkerFaultTolerance.defineRoutes();
            WorkerProactiveFaultTolerance.defineRoutes();
        }
    }
}
