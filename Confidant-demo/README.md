# Confidant Demo

## Quick Start

This quick start guide shows how to set up and run the **Confidant Demo** on Android devices with minimal configuration.

### Requirements

- Android Studio (Koala 2024.1.2 recommended)
- One or more Android mobile devices
- All devices connected to the same network

### Steps

1. **Install Android Studio**  
   Download and install Android Studio (Koala 2024.1.2 was used during development).

2. **Open the Project**  
   Launch Android Studio and open the `Confidant-demo` project.

3. **Prepare Mobile Devices**  
   Connect the Android device(s) to your computer and enable the USB debugging mode.

4. **Copy Dataset and Pretrained Weights Data**  
   Extract the `confidant.tar` file and copy the extracted `confidant` folder to the **root directory** of the device’s internal storage.

5. **Configure the App**  
   - Edit the following file:
   ```app/src/main/java/com/example/confidant/globalStates/Common.java```
   - Update the following fields:
     - Lines 48–51: Model to be fine-tuned
     - Line 79: IP address of the central node
     - Line 84: IP addresses of all worker nodes

6. **Install the App**  
   Build and install Confidant on each mobile device. Make sure all required permissions are granted.

7. **Launch Worker Nodes**  
   On each worker device:
   - Open the Confidant app
   - Click **Worker Node** to switch to worker mode

8. **Launch Central Node**  
   On the designated central device:
   - Open the Confidant app
   - Click **Central Node**
   - The app will automatically search for worker nodes

9. **Start Fine-tuning**  
   Once all nodes are connected, the fine-tuning process will start automatically.

### Notes

- Ensure all devices are connected within the same LAN and are reachable via the configured IP addresses.
- If worker nodes are not discovered, check network connectivity and app permissions.