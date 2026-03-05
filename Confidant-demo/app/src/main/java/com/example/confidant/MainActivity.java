package com.example.confidant;

import static com.example.confidant.grpcAPI.GrpcServer.createGrpcServer;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Environment;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.example.confidant.databinding.ActivityMainBinding;
import com.example.confidant.faultTolerance.ProactiveFTHandler;
import com.example.confidant.globalStates.Common;
import com.example.confidant.request.RequestHelper;
import com.example.confidant.sparkAPI.SparkApp;
import com.example.confidant.utils.CustomView;
import com.example.confidant.utils.General;
import com.example.confidant.utils.Offline;
import com.example.confidant.utils.TrainCentral;
import com.example.confidant.utils.TrainSingle;
import com.example.confidant.unitTest.TestEntry;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;


public class MainActivity extends AppCompatActivity {
    // load the native library
    static {
        System.loadLibrary("confidant");
    }

    private ActivityMainBinding binding;
    private ImageView imageView;
    private RecyclerView logView;
    private CustomView.LogAdapter logAdapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        // Set the base path that stores the data
        Common.setBasePath(this.getFilesDir().toString());

        // Init the user interface
        initView();

        // Check whether network is connected
        String ipAddress = General.getDeviceIPAddress();
        if (ipAddress == null) {
            Common.printLog("No ip is found!");
        } else {
            Common.printLog("IP Address: " + ipAddress);
        }

        // Print the trained model on UI
        General.printTrainedModel();
        General.syncGlobalStates();

        // Set the buttons in UI
        initButton();

        // Acquire the permission to read and write external storage
        int permission = ActivityCompat.checkSelfPermission(this, android.Manifest.permission.WRITE_EXTERNAL_STORAGE);
        if (permission != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{android.Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE}, 222);
            finish();
        }
    }

    /*
     * Init the UI
     */
    private void initView() {
        logView = findViewById(R.id.logRecycler);

        LinearLayoutManager layoutManager = new LinearLayoutManager(this);
        logView.setLayoutManager(layoutManager);

        layoutManager.setOrientation(LinearLayoutManager.HORIZONTAL);
        layoutManager.setOrientation(LinearLayoutManager.VERTICAL);

        List<CustomView.LogItem> logs = new ArrayList<>();
        logAdapter = new CustomView.LogAdapter(logs);
        Common.setLogAdapter(MainActivity.this, logAdapter);
        Common.setLogView(logView); // For the log to scroll to the bottom
        logView.setAdapter(logAdapter);
    }

    /*
        Init the buttons in the UI
     */
    private void initButton() {
        Button centralBtn = findViewById(R.id.central_btn);
        Button workerBtn = findViewById(R.id.worker_btn);
        Button singleBtn = findViewById(R.id.single_btn);
        Button testBtn = findViewById(R.id.test_btn);
        Button exitBtn = findViewById(R.id.proactive_exit_btn);

        centralBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // no proactive exit button for central node
                centralBtn.setVisibility(View.GONE);
                workerBtn.setVisibility(View.GONE);
                singleBtn.setVisibility(View.GONE);
                testBtn.setVisibility(View.GONE);

                Common.printLog("Init the training as the central node ...");
                initCentral();
            }
        });

        workerBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                exitBtn.setVisibility(View.VISIBLE);
                centralBtn.setVisibility(View.GONE);
                workerBtn.setVisibility(View.GONE);
                singleBtn.setVisibility(View.GONE);
                testBtn.setVisibility(View.GONE);

                Common.printLog("Init the training as the worker node ...");
                initWorker();
            }
        });

        singleBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                centralBtn.setVisibility(View.GONE);
                workerBtn.setVisibility(View.GONE);
                singleBtn.setVisibility(View.GONE);
                testBtn.setVisibility(View.GONE);

                Common.printLog("Init single training ...");

                initConfig();
                setDatasetAndModelWeights();

                singleTrain();
            }
        });

        testBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                centralBtn.setVisibility(View.GONE);
                workerBtn.setVisibility(View.GONE);
                singleBtn.setVisibility(View.GONE);
                testBtn.setVisibility(View.GONE);

                initConfig();
                setDatasetAndModelWeights();

                Common.printLog("Init test ...");

                RequestHelper.initRequestHelper(MainActivity.this);

                Thread thread = new Thread(new Runnable() {
                    @Override
                    public void run() {
                        TestEntry.testEntry(MainActivity.this);
                    }
                }, "UnitTest");
                thread.start();
            }
        });

        exitBtn.setOnClickListener(view -> showExitConfirmationDialog());
    }

    /*
        Display a confirmation dialog to check whether the user wants to trigger the proactive exit
     */
    private void showExitConfirmationDialog() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Proactive Exit Confirmation");
        builder.setMessage("Do you want to exit the training and notify the central node?");

        builder.setPositiveButton("Exit", (dialogInterface, i) -> {
            Button exitBtn = findViewById(R.id.proactive_exit_btn);
            exitBtn.setVisibility(View.GONE);
            Common.printLog("Start Proactive exit ...");
            ProactiveFTHandler.notifyExit();
        });

        builder.setNegativeButton("Cancel", (dialogInterface, i) -> {
            dialogInterface.dismiss();
        });

        builder.show();
    }

    private void initCentral() {
        initConfig();
        setDatasetAndModelWeights();

        initSpark(0);

        Thread thread = new Thread(new Runnable() {
            @Override
            public void run() {
                createGrpcServer(0, Common.getGrpcPort());
            }
        }, "Central.createGrpcServer");
        thread.start();

        // Init the network request helper
        RequestHelper.initRequestHelper(MainActivity.this);

        collaborativeTrain();
    }

    private void initWorker() {
        initConfig();

        initSpark(1);

        Thread thread = new Thread(new Runnable() {
            @Override
            public void run() {
                createGrpcServer(1, Common.getGrpcPort());
            }
        }, "Worker.createGrpcServer");
        thread.start();

        setDatasetAndModelWeights();

        RequestHelper.initRequestHelper(MainActivity.this);
    }

    /*
        Init the Spark that listen on the port, 0 denotes the central node, 1 denotes the worker node
     */
    private void initSpark(int role) {
        SparkApp.main(role);
    }

    /**
     * Load the config file and initialize the config
     */
    private void initConfig() {
        Common.printLog("Loading the config file ... ");
        General.loadConfig(MainActivity.this);
    }

    private Map<String, Object> getConfig() {
        Common.printLog("Geting the config file ... ");

        return General.getConfigInfo(MainActivity.this);
    }

    private void collaborativeTrain() {
        Thread thread = new Thread(new Runnable() {
            @Override
            public void run() {
                int workerNum = Offline.electWorkers();
                if (workerNum == 1) {
                    Common.printLog("No worker nodes are found, training stop");
                    return ;
                }

                Offline.distributeWorkers();
                Offline.offlineProfiling(getConfig());
                Offline.distributeBasicInfo();

                TrainCentral.startTrain();
            }
        }, "collaborativeTraining");
        thread.start();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        for (int result : grantResults) {
            if (result != PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Write Permission!", Toast.LENGTH_SHORT).show();
                this.finish();
            }
        }
    }

    private void setDatasetAndModelWeights(){
        // Acquire the permission to read and write external storage
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[] { Manifest.permission.WRITE_EXTERNAL_STORAGE }, 1);
        }

        // check sd card state
        String state = Environment.getExternalStorageState();
        if (Environment.MEDIA_MOUNTED.equals(state)){
            Common.printLog("sd card mounted");
        }

        // get sdcard root dir /sdcard/emulator/0/
        String rootDir = Environment.getExternalStorageDirectory().getAbsolutePath();
        // get app dir
        String appDir = this.getFilesDir().getAbsolutePath();

        // set dataset path
        Common.printLog("---------------------------------");
        String datasetPath = "confidant" + File.separator + "trainingDatas" + File.separator +
                Common.getModelName() + File.separator;
        String rootDatasetPath = rootDir + File.separator + datasetPath;
        String appDatasetPath = appDir + File.separator + datasetPath;
        Common.setDatasetBasePath(appDatasetPath);
        Common.printLog("rootDatasetPath: " + rootDatasetPath);
        Common.printLog("appDatasetPath: " + appDatasetPath);
        try {
            copyDatasAndWeightsToApp(rootDatasetPath, appDatasetPath);
            Common.printLog("Copy datasets success...");
        }
        catch (Exception e){
            e.printStackTrace();
            Common.printLog("Copy datasets failed!");
            Common.printLog(e.getMessage());
        }
        Common.printLog("---------------------------------");

        // set weights path
        Common.printLog("---------------------------------");
        String weightsPath = "confidant" + File.separator + "modelWeights" + File.separator +
                Common.getModelName() + File.separator;
        String rootWeightsPath = rootDir + File.separator + weightsPath;
        String appWeightsPath = appDir + File.separator + weightsPath;
        Common.setWeightsPath(appWeightsPath);
        Common.printLog("rootWeightsPath: " + rootWeightsPath);
        Common.printLog("appWeightsPath: " + appWeightsPath);
        try {
            copyDatasAndWeightsToApp(rootWeightsPath, appWeightsPath);
            Common.printLog("Copy weights success...");
        }
        catch (Exception e){
            e.printStackTrace();
            Common.printLog("Copy weights failed!");
            Common.printLog(e.getMessage());
        }
        Common.printLog("---------------------------------");
    }

    void copyDatasAndWeightsToApp(String sourcePath, String targetPath) {
        File sourceDirectory = new File(sourcePath);
        File targetDirectory = new File(targetPath);

        if (sourceDirectory.exists() && sourceDirectory.isDirectory()) {
            if (!targetDirectory.exists()) {
                targetDirectory.mkdirs();
            }
            copyFilesAndFolders(sourceDirectory, targetDirectory);
        } else {
            System.out.println("Source path is not a valid directory.");
        }
    }

    private void copyFilesAndFolders(File source, File target) {
        Common.printLog("On file: " + source.getName());
        if (source.isDirectory()) {
            File[] files = source.listFiles();
            if (files != null) {
                for (File file : files) {
                    File newSource = new File(source, file.getName());
                    File newTarget = new File(target, file.getName());
                    if (file.isDirectory()) {
                        Common.printLog("copy folder: " + file.getName());
                        newTarget.mkdirs();
                        copyFilesAndFolders(newSource, newTarget);
                    } else {
                        Common.printLog("copy file: " + file.getName());
                        if (!newTarget.exists()) {
                            // if file does not exist, copy it
                            copyFile(newSource, newTarget);
                        } else {
                            Common.printLog("File already exists: " + newTarget.getName());
                        }
                    }
                }
            } else {
                Common.printLog("file " + source.getName() + " is null");
            }
        }
    }

    private void copyFile(File source, File target) {
        try {
            FileInputStream inputStream = new FileInputStream(source);
            FileOutputStream outputStream = new FileOutputStream(target);
            byte[] buffer = new byte[1024];
            int bytesRead;
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
            }
            outputStream.flush();
            outputStream.close();
            inputStream.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void singleTrain() {
        Thread thread = new Thread(new Runnable() {
            @Override
            public void run() {
                TrainSingle.startTrain();
            }
        }, "singleTraining");
        thread.start();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        Common.shutdownAllChannels();
    }
}
