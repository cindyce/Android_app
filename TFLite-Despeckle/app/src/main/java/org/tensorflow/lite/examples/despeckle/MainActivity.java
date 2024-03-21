/*
 * Copyright 2020 The TensorFlow Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.despeckle;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.os.SystemClock;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.provider.MediaStore;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;
import androidx.annotation.WorkerThread;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.github.chrisbanes.photoview.PhotoView;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/** generate despeckle images * */
public class MainActivity extends AppCompatActivity {
  private static final int PICK_PHOTO = 100;
  private static final int MY_PERMISSIONS_REQUEST_CALL_PHONE2 = 101;

  static {
    System.loadLibrary("Despeckle");
  }

  private static final String TAG = "ImageDespeckle";
  private static final String MODEL_NAME = "model128_tensorflow.tflite";
  private static final int IN_HEIGHT = 128;
  private static final int IN_WIDTH = 128;
  private static final String LR_IMG_5 = "b_369.bmp";
  private static final String LR_IMG_6 = "b_404.bmp";
  private static final String LR_IMG_7 = "b_900.bmp";
  private long processingTimeMs;

  private MappedByteBuffer model;
  private long despeckleNativeHandle = 0;
  private Bitmap selectedLRBitmap = null;
  private Bitmap srBitmap = null;
  private boolean useGPU = false;

  private ImageView ImageView1;
  private ImageView ImageView2;
  private ImageView ImageView3;
  private PhotoView dsPhotoView;
  private TextView selectedImageTextView;
  private TextView progressTextView;
  private TextView logTextView;
  private Switch gpuSwitch;
  private UIHandler UIhandler;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    requirePremision();

    final Button despeckleButton = findViewById(R.id.upsample_button);
    ImageView1 = findViewById(R.id.image_1);
    ImageView2 = findViewById(R.id.image_2);
    ImageView3 = findViewById(R.id.image_3);
    progressTextView = findViewById(R.id.progress_tv);
    logTextView = findViewById(R.id.log_view);

    dsPhotoView = (PhotoView) findViewById(R.id.ds_view);
    PhotoView selectedPhotoView = (PhotoView) findViewById(R.id.selected_view);

    selectedImageTextView = findViewById(R.id.chosen_image_tv);
    gpuSwitch = findViewById(R.id.switch_use_gpu);

    ImageView[] lowResImageViews = {ImageView1, ImageView2, ImageView3};

    AssetManager assetManager = getAssets();
    try {
      InputStream inputStream1 = assetManager.open(LR_IMG_5);
      Bitmap bitmap1 = BitmapFactory.decodeStream(inputStream1);
      ImageView1.setImageBitmap(bitmap1);

      InputStream inputStream2 = assetManager.open(LR_IMG_6);
      Bitmap bitmap2 = BitmapFactory.decodeStream(inputStream2);
      ImageView2.setImageBitmap(bitmap2);

      InputStream inputStream3 = assetManager.open(LR_IMG_7);
      Bitmap bitmap3 = BitmapFactory.decodeStream(inputStream3);
      ImageView3.setImageBitmap(bitmap3);
    } catch (IOException e) {
      Log.e(TAG, "Failed to open an low resolution image");
    }

    for (ImageView iv : lowResImageViews) {
      setImageViewListener(iv);
    }

    findViewById(R.id.save_button).setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View v) {
        if(srBitmap != null){

          MediaStore.Images.Media.insertImage(getContentResolver(), srBitmap, "Despeckle", "TFLite SR demo");
          Toast.makeText(
                          getApplicationContext(),
                          "Saved!",
                          Toast.LENGTH_LONG)
                  .show();
        }

      }
    });

    despeckleButton.setOnClickListener(
            new View.OnClickListener() {
              @Override
              public void onClick(View view) {
                if (selectedLRBitmap == null) {
                  Toast.makeText(
                                  getApplicationContext(),
                                  "Please choose one low resolution image",
                                  Toast.LENGTH_LONG)
                          .show();
                  return;
                }

                dsPhotoView.setImageDrawable(null);
                selectedPhotoView.setImageBitmap(selectedLRBitmap);
                progressTextView.setText("loading...");

                if (despeckleNativeHandle == 0) {
                  despeckleNativeHandle = initTFLiteInterpreter(gpuSwitch.isChecked());
                } else if (useGPU != gpuSwitch.isChecked()) {
                  // We need to reinitialize interpreter when execution hardware is changed
                  deinit();
                  despeckleNativeHandle = initTFLiteInterpreter(gpuSwitch.isChecked());
                }
                useGPU = gpuSwitch.isChecked();
                if (despeckleNativeHandle == 0) {
                  showToast("TFLite interpreter failed to create!");
                  return;
                }

                UIhandler = new UIHandler();
                new Thread(new Runnable() {
                  @Override
                  public void run() {
                    doDespeckle();
                  }
                }).start();

              }
            });


    //从相册选择图片
    findViewById(R.id.open_button).setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View v) {
        //动态申请获取访问 读写磁盘的权限
        if (ContextCompat.checkSelfPermission(MainActivity.this,
                Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
          ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, 101);
        } else {
          //打开相册
          Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
          //Intent.ACTION_GET_CONTENT = "android.intent.action.GET_CONTENT"
          intent.setType("image/*");
          startActivityForResult(intent, PICK_PHOTO); // 打开相册
        }
      }
    });

  }

  private class UIHandler extends Handler {
    @Override
    public void handleMessage(Message msg) {
      // TODO Auto-generated method stub
      super.handleMessage(msg);
      Bundle bundle = msg.getData();
      String progress = bundle.getString("progress");

      if(progress!=null){
        progressTextView.setText(progress);
        logTextView.setText(progress);
      }
      dsPhotoView.setImageBitmap(srBitmap);
    }
  }

  private void requirePremision() {
    if (ContextCompat.checkSelfPermission(this,
            Manifest.permission.WRITE_EXTERNAL_STORAGE)
            != PackageManager.PERMISSION_GRANTED)
    {
      ActivityCompat.requestPermissions(this,
              new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
              MY_PERMISSIONS_REQUEST_CALL_PHONE2);
    }else {
      //权限已经被授予，在这里直接写要执行的相应方法即可
    }
  }

  @Override
  public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults)
  {
    if (requestCode == MY_PERMISSIONS_REQUEST_CALL_PHONE2)
    {
      if (grantResults[0] == PackageManager.PERMISSION_GRANTED)
      {

      } else
      {
        // Permission Denied
        Toast.makeText(MainActivity.this, "Permission Denied", Toast.LENGTH_SHORT).show();
      }
    }
    super.onRequestPermissionsResult(requestCode, permissions, grantResults);
  }


  @Override
  protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
    switch (requestCode) {
      case PICK_PHOTO:
        if (resultCode == RESULT_OK && null != data) { // 判断手机系统版本号
          Uri uri = data.getData();
          try {
            selectedLRBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
            selectedImageTextView.setText(
                    "You opened low resolution image:  ("
                            + data.toString()
                            + ")");
          } catch (IOException e) {
            e.printStackTrace();
            showToast("Pick photo failed!");
          }
        }
        break;
      default:
        break;
    }
    super.onActivityResult(requestCode, resultCode, data);
  }


  @Override
  public void onDestroy() {
    super.onDestroy();
    deinit();
  }

  private void setImageViewListener(ImageView iv) {
    iv.setOnTouchListener(
            new View.OnTouchListener() {
              @Override
              public boolean onTouch(View v, MotionEvent event) {
                if (v.equals(ImageView1)) {
                  selectedLRBitmap = ((BitmapDrawable) ImageView1.getDrawable()).getBitmap();
                  selectedImageTextView.setText(
                          "You are using low resolution image: 1");
                } else if (v.equals(ImageView2)) {
                  selectedLRBitmap = ((BitmapDrawable) ImageView2.getDrawable()).getBitmap();
                  selectedImageTextView.setText(
                          "You are using low resolution image: 2");
                } else if (v.equals(ImageView3)) {
                  selectedLRBitmap = ((BitmapDrawable) ImageView3.getDrawable()).getBitmap();
                  selectedImageTextView.setText(
                          "You are using low resolution image: 3");
                }
                return false;
              }
            });
  }

  @WorkerThread
  public synchronized void doDespeckle() {
    final long startTime = SystemClock.uptimeMillis();
    int progress = 0;
    int w = selectedLRBitmap.getWidth();
    int h = selectedLRBitmap.getHeight();

    int max_a = (int)Math.ceil ((float)w/IN_WIDTH);
    int max_b = (int)Math.ceil((float)h/IN_HEIGHT);

    int max_w = max_a * IN_WIDTH;
    int max_h = max_b * IN_HEIGHT;

    srBitmap = Bitmap.createBitmap(w,h, Bitmap.Config.ARGB_8888); //指定的寬高可變
    Bitmap inputBitmap = Bitmap.createBitmap(selectedLRBitmap,0,0,w,h);

    for(int a = 0; a<max_a;a++){
      Message msg = new Message();
      if(a!=0){
        Bundle bundle = new Bundle();
        bundle.putString("progress", "progress: " +  a + "/" + max_a
                + ", need " +  (SystemClock.uptimeMillis() - startTime)/a*(max_a -a) + "ms");
        msg.setData(bundle);
      }
      MainActivity.this.UIhandler.sendMessage(msg);

//      int in_width = (max_a-a==1)?w-a*IN_WIDTH:IN_WIDTH;
      int x = (max_a-a==1)?(w-IN_WIDTH):IN_WIDTH*a;

      for(int b=0;b<max_b;b++){
//        int in_height = (max_b-b==1)?h-b*IN_HEIGHT:IN_HEIGHT;
        int y = (max_b-b==1)?h-IN_HEIGHT:IN_HEIGHT*b;
        int[] lowResRGB = new int[IN_WIDTH*IN_HEIGHT];

        inputBitmap.getPixels(
                lowResRGB, 0, IN_WIDTH,x , y, IN_WIDTH, IN_HEIGHT);

//        Log.d("this is the patch", "lowResRGB: " + Arrays.toString(lowResRGB));
//        Log.d("this is image height", "h: " + String.valueOf(h));
//        Log.d("this is image width", "h: " + String.valueOf(w));

        srBitmap.setPixels(
                despeckleFromJNI(despeckleNativeHandle, lowResRGB)
                ,0,IN_WIDTH
                ,x,y,IN_WIDTH,IN_HEIGHT);

        if(progress<0){
          processingTimeMs = -1;
          return ;
        }
        progress ++;
      }
    }
    processingTimeMs = SystemClock.uptimeMillis() - startTime;
    Message msg = new Message();
    Bundle bundle = new Bundle();
    bundle.putString("progress", "Inference time: " + processingTimeMs + "ms");
    msg.setData(bundle);
    MainActivity.this.UIhandler.sendMessage(msg);

  }


  @WorkerThread
  public synchronized int[] dodespeckle(int[] lowResRGB, int w, int h) {
    return despeckleFromJNI(despeckleNativeHandle, lowResRGB);
  }

  private MappedByteBuffer loadModelFile() throws IOException {
    try (AssetFileDescriptor fileDescriptor =
                 AssetsUtil.getAssetFileDescriptorOrCached(getApplicationContext(), MODEL_NAME);
         FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
      FileChannel fileChannel = inputStream.getChannel();
      long startOffset = fileDescriptor.getStartOffset();
      long declaredLength = fileDescriptor.getDeclaredLength();
      return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
  }

  private void showToast(String str) {
    Toast.makeText(getApplicationContext(), str, Toast.LENGTH_LONG).show();
  }

  private long initTFLiteInterpreter(boolean useGPU) {
    try {
      model = loadModelFile();
    } catch (IOException e) {
      Log.e(TAG, "Fail to load model", e);
    }
    return initWithByteBufferFromJNI(model, useGPU);
  }

  private void deinit() {
    deinitFromJNI(despeckleNativeHandle);
  }

  private native int[] despeckleFromJNI(long despeckleNativeHandle, int[] lowResRGB);

  private native long initWithByteBufferFromJNI(MappedByteBuffer modelBuffer, boolean useGPU);

  private native void deinitFromJNI(long despeckleNativeHandle);
}
