package ai.skymind.skil.examples.yolo2.modelserver.inference;

import ai.skymind.skil.examples.yolo2.modelserver.inference.yolo2.*;

import org.deeplearning4j.nn.layers.objdetect.DetectedObject;

import org.datavec.image.data.Image;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ColorConversionTransform;
import org.datavec.image.data.Image;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ColorConversionTransform;
import org.datavec.image.data.ImageWritable;
import org.datavec.api.util.ClassPathResource;
import org.datavec.image.transform.ImageTransformProcess;
import org.datavec.spark.transform.model.Base64NDArrayBody;
import org.datavec.spark.transform.model.BatchImageRecord;
import org.datavec.spark.transform.model.SingleImageRecord;
import org.datavec.image.data.Image;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ColorConversionTransform;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Broadcast;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.serde.base64.Nd4jBase64;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Broadcast;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;


import com.mashape.unirest.http.JsonNode;
import com.mashape.unirest.http.ObjectMapper;
import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.exceptions.UnirestException;

import org.json.JSONObject;
import org.json.JSONArray;


import java.text.MessageFormat;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
//import java.util.logging.Level;
//import java.util.logging.Logger;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.*;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.io.InputStream;
import java.net.URL;
import java.util.List;

import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.converter.HttpMessageConverter;
import org.springframework.http.converter.json.MappingJackson2HttpMessageConverter;
import org.springframework.web.client.RestTemplate;

import com.beust.jcommander.Parameter;
import com.beust.jcommander.JCommander;

import org.apache.commons.io.IOUtils;


import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.layout.StackPane;
import javafx.scene.paint.Color;
import javafx.scene.paint.Paint;
import javafx.scene.text.TextAlignment;
import javafx.stage.Stage;

import org.slf4j.LoggerFactory;
import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;

/**

    The network we're targetting on SKIL is:



    This is the official website listed with the yolo900 paper 

    https://pjreddie.com/darknet/yolo/

    The weights are from here and are listed under YOLOv2 608x608


    This repo converts it from darknet to TF and has instructions on how to get the single pb file aka the frozen graph 

    https://github.com/thtrieu/darkflow




    For the client side, we are targetting something similar to:

    https://github.com/experiencor/basic-yolo-keras/blob/master/frontend.py#L289

    Similar to:

    https://github.com/deeplearning4j/deeplearning4j/blob/3c260c3a607b91d1cbbde495b76c7d1827ba0851/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/objdetect/Yolo2OutputLayer.java#L631


*/
public class YOLO2_TF_Client extends Application {

    //public static final int nClasses = 80;
    public static final int gridWidth = 19;
    public static final int gridHeight = 19;
    public static final double[][] priorBoxes = {{0.57273, 0.677385}, {1.87446, 2.06253}, {3.33843, 5.47434}, {7.88282, 3.52778}, {9.77052, 9.16828}};
/*
    private static final String[] CLASSES = { "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
            "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
            "train", "tvmonitor", "truck", "traffic light" };
*/
    private static final String[] COLORS = { 
            "#6793be", "#990000", "#fececf", "#ffbcc9", "#ffb9c7", "#fdc6d1",
            "#fdc9d3", "#6793be", "#73a4d4", "#9abde0", "#9abde0", "#8fff8f", "#ffcfd8", "#808080", "#808080",
            "#ffba00", "#6699ff", "#009933", "#1c1c1c", "#08375f", "#116ebf", "#e61d35", "#106bff", "#8f8fff",
            "#8fff8f", "#dbdbff", "#dbffdb", "#dbffff", "#ffdbdb", "#ffc2c2", "#ffa8a8", "#ff8f8f", "#e85e68",
            "#123456", "#5cd38c", "#1d1f5f", "#4e4b04", "#495a5b", "#489d73", "#9d4872", "#d49ea6", "#ff0080",
            "#6793be", "#990000", "#fececf", "#ffbcc9", "#ffb9c7", "#fdc6d1",
            "#fdc9d3", "#6793be", "#73a4d4", "#9abde0", "#9abde0", "#8fff8f", "#ffcfd8", "#808080", "#808080",
            "#ffba00", "#6699ff", "#009933", "#1c1c1c", "#08375f", "#116ebf", "#e61d35", "#106bff", "#8f8fff",
            "#8fff8f", "#dbdbff", "#dbffdb", "#dbffff", "#ffdbdb", "#ffc2c2", "#ffa8a8", "#ff8f8f", "#e85e68",
            "#123456", "#5cd38c", "#1d1f5f", "#4e4b04", "#495a5b", "#489d73", "#9d4872", "#d49ea6", "#ff0080" 
        };

    @Parameter(names="--endpoint", description="Endpoint for classification", required=true)
    private String skilInferenceEndpoint = ""; // EXAMPLE: "http://localhost:9008/endpoints/mnist/model/mnistmodel/default/";

    // this is public cause the canvas object needs it on creation --> hack
    @Parameter(names="--input", description="Image input file url", required=true)
    public String input_image = "";

    static InputStream inputFileStream = null;
    static INDArray networkGlobalOutput = null;

    static List<DetectedObject> predictedObjects = null;
    Mat imgMat = null;
    
    static String source_image = ""; // "https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg";
    int imageWidth = 0, imageHeight = 0, imageChannels = 3;

    public YOLO2_TF_Client() { }

    public YOLO2_TF_Client( String sourceImage ) {

        this.source_image = sourceImage;

    }


/*
    Images

        Dog
                "https://raw.githubusercontent.com/deeplearning4j/deeplearning4j/master/deeplearning4j-zoo/src/main/resources/goldenretriever.jpg"
                224x224 - 3 channels

        Dog + Bike
            https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg
            635x476

*/
    public void run() throws Exception, IOException {

Set<String> loggers = new HashSet<>(Arrays.asList("org.apache.http", "groovyx.net.http"));
    
    for(String log:loggers) { 
    Logger logger = (Logger)LoggerFactory.getLogger(log);
    logger.setLevel(Level.INFO);
    logger.setAdditive(false);
    }

        //List<String> labels = IOUtils.readLines(new URL("https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names").openStream());

        // make model server network calls for { auth, inference }
        skilClientGetImageInference( imageWidth, imageHeight );

        // kick off the javaFX rendering code --> .start( Stage ) below, blocked on the skil model server round trips for { auth, inference }
        launch();    

    }    

    private void skilClientGetImageInference( int width, int height ) throws Exception, IOException  {
//List<String> labels, 
        InputStream imgStream = new URL( source_image ).openStream();
        inputFileStream = imgStream;

        imgMat = imdecode(new Mat(IOUtils.toByteArray( imgStream )), CV_LOAD_IMAGE_COLOR);
        imageHeight = imgMat.rows();
        imageWidth = imgMat.cols();
        System.out.println( "Input Image: " + source_image );
        System.out.println( "Input width: " + imageWidth );
        System.out.println( "Input height: " + imageHeight );

        NativeImageLoader imageLoader = new NativeImageLoader(608, 608, 3, new ColorConversionTransform(COLOR_BGR2RGB));
        INDArray imgNDArrayTmp = imageLoader.asMatrix( imgMat );
        INDArray inputFeatures = imgNDArrayTmp.permute(0, 2, 3, 1).muli(1.0 / 255.0).dup('c');
        System.out.println( "Finished image conversion" );

        String imgBase64 = Nd4jBase64.base64String( inputFeatures );
        Authorization auth = new Authorization();
        String auth_token = auth.getAuthToken( "admin", "admin" );
        System.out.println( "auth token: " + auth_token );

        System.out.println( "\n\n\n\nNow Sending the Classification Payload......\n\n\n" );

        try {

            JSONObject returnJSONObject = 
                    Unirest.post( skilInferenceEndpoint + "predict" )
                            .header("accept", "application/json")
                            .header("Content-Type", "application/json")
                            .header( "Authorization", "Bearer " + auth_token)
                            .body(new JSONObject() //Using this because the field functions couldn't get translated to an acceptable json
                                    .put( "id", "some_id" )
                                    .put("prediction", new JSONObject().put("array", imgBase64))
                                    .toString())
                            .asJson()
                            .getBody().getObject(); //.toString(); 

            try {

                returnJSONObject.getJSONObject("prediction").getString("array");

            } catch (org.json.JSONException je) { 

                System.out.println( "\n\nException\n\nReturn: " + returnJSONObject );
                return;

            }

            String predict_return_array = returnJSONObject.getJSONObject("prediction").getString("array");
            System.out.println( "classification return length: " + predict_return_array.length() );
            INDArray networkOutput = Nd4jBase64.fromBase64( predict_return_array );

            networkGlobalOutput = networkOutput; // hack to pass info to main thread

            System.out.println( "done with SKIL calls ... " );

        } catch (UnirestException e) {
            e.printStackTrace();
        }

    }

    public static void main(String[] args) throws Exception {

        // the JavaFX code initializes early, needed the image size from the image src path (yes, hack)        
        YOLO2_TF_Client m = new YOLO2_TF_Client( args[ 1 ] );

        JCommander.newBuilder()
          .addObject(m)
          .build()
          .parse(args);

        m.run();
    }

    @Override
    public void start(Stage stage) throws Exception {
        
        // only because this is a separate thread and we need the info
        InputStream imgStream = new URL( source_image ).openStream();
        inputFileStream = imgStream;
        imgMat = imdecode(new Mat(IOUtils.toByteArray( imgStream )), CV_LOAD_IMAGE_COLOR);
        imageHeight = imgMat.rows();
        imageWidth = imgMat.cols();

        Canvas canvas = new Canvas(imageWidth, imageHeight);
        GraphicsContext ctx = canvas.getGraphicsContext2D();
        InputStream is = new URL( source_image ).openStream();

        ctx.drawImage(new javafx.scene.image.Image(is, this.imageWidth, this.imageHeight, false, false), 0, 0);
        stage.setScene(new Scene(new StackPane(canvas), this.imageWidth, this.imageHeight));
        renderJavaFXStyle( ctx );
        stage.setTitle("YOLO2");
        stage.show();

    }

    /**
        TODO: we're using 2 different sources of labels here, fix this
        --- let's move to centralized colors and labels for the demo
    */
    private void renderJavaFXStyle(GraphicsContext ctx) throws Exception {

        INDArray boundingBoxPriors = Nd4j.create(priorBoxes);
        List<String> labels = IOUtils.readLines(new URL("https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names").openStream());
        Map<String, Paint> colors = new HashMap<>();
        /*
        for (int i = 0; i < CLASSES.length; i++) {
            colors.put(CLASSES[i], Color.web(COLORS[i]));
        }
*/
        for (int i = 0; i < labels.size(); i++) {
            colors.put( labels.get( i ), Color.web( COLORS[i] ) );
        }


        ctx.setLineWidth(3);
        ctx.setTextAlign(TextAlignment.LEFT);

        long start = System.nanoTime();
        for (int i = 0; i < 1; i++) {
            INDArray permuted = networkGlobalOutput.permute(0, 3, 1, 2);
            INDArray activated = YoloUtils.activate(boundingBoxPriors, permuted);
            List<DetectedObject> predictedObjects = YoloUtils.getPredictedObjects(boundingBoxPriors, activated, 0.6, 0.4);
            
            System.out.println( "width: " + imageWidth );
            System.out.println( "height: " + imageHeight );

            for (DetectedObject o : predictedObjects) {
                String label = labels.get(o.getPredictedClass());
                long x = Math.round(imageWidth  * o.getCenterX() / gridWidth);
                long y = Math.round(imageHeight * o.getCenterY() / gridHeight);
                long w = Math.round(imageWidth  * o.getWidth()   / gridWidth);
                long h = Math.round(imageHeight * o.getHeight()  / gridHeight);

                System.out.println("\"" + label + "\" at [" + x + "," + y + ";" + w + "," + h + "], conf = " + o.getConfidence());

                double[] xy1 = o.getTopLeftXY();
                double[] xy2 = o.getBottomRightXY();
                int x1 = (int) Math.round(imageWidth  * xy1[0] / gridWidth);
                int y1 = (int) Math.round(imageHeight * xy1[1] / gridHeight);
                int x2 = (int) Math.round(imageWidth  * xy2[0] / gridWidth);
                int y2 = (int) Math.round(imageHeight * xy2[1] / gridHeight);

                int rectW = x2 - x1;
                int rectH = y2 - y1;
                System.out.printf("%s - %d, %d, %d, %d \n", label, x1, x2, y1, y2);
                System.out.println( "color: " + colors.get(label) );
                ctx.setStroke(colors.get(label));
                ctx.strokeRect(x1, y1, rectW, rectH);
                
                int labelWidth = label.length() * 10;
                int labelHeight = 14;
                
                ctx.setFill( colors.get(label) );
                ctx.strokeRect(x1, y1-labelHeight, labelWidth, labelHeight);
                ctx.fillRect(x1, y1 - labelHeight, labelWidth, labelHeight);
                ctx.setFill( Color.WHITE );
                ctx.fillText(label, x1 + 3, y1 - 3 );

            }

        }

        long end = System.nanoTime();
        System.out.println((end - start) / 1000000 + " ms");

    }

    /**
        Simple helper class to encapsulate some of the raw REST code for making basic authentication calls
    */
    private class Authorization {

        private String host;
        private String port;

        public Authorization() {
            this.host = "localhost";
            this.port = "9008";
        }

        public Authorization(String host, String port) {
            this.host = host;
            this.port = port;
        }

        public String getAuthToken(String userId, String password) {
            String authToken = null;

            try {
                authToken =
                        Unirest.post(MessageFormat.format("http://{0}:{1}/login", host, port))
                                .header("accept", "application/json")
                                .header("Content-Type", "application/json")
                                .body(new JSONObject() //Using this because the field functions couldn't get translated to an acceptable json
                                        .put("userId", userId)
                                        .put("password", password)
                                        .toString())
                                .asJson()
                                .getBody().getObject().getString("token");
            } catch (UnirestException e) {
                e.printStackTrace();
            }

            return authToken;
        }
    }


}
