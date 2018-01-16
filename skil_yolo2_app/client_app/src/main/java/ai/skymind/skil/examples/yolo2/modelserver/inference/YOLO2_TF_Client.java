package ai.skymind.skil.examples.yolo2.modelserver.inference;

/*
import io.skymind.auth.AuthClient;
import io.skymind.skil.predict.client.PredictServiceClient;
import io.skymind.skil.service.model.ClassificationResult;
import io.skymind.skil.service.model.JsonArrayResponse;
import io.skymind.skil.service.model.MultiClassClassificationResult;
import io.skymind.skil.service.model.Prediction;
*/


import org.datavec.image.data.ImageWritable;

import org.datavec.api.util.ClassPathResource;
import org.datavec.image.transform.ImageTransformProcess;
import org.datavec.spark.transform.model.Base64NDArrayBody;
import org.datavec.spark.transform.model.BatchImageRecord;
import org.datavec.spark.transform.model.SingleImageRecord;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.serde.base64.Nd4jBase64;

//import ai.skymind.skil.examples.mnist.modelserver.inference.model.TransformedImage;


import com.mashape.unirest.http.JsonNode;
import com.mashape.unirest.http.ObjectMapper;
import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.exceptions.UnirestException;
import org.json.JSONObject;
import java.io.IOException;

import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.exceptions.UnirestException;
import org.json.JSONObject;

import java.text.MessageFormat;


// --

//import org.deeplearning4j.nn.layers.objdetect.DetectedObject;

import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;



import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.converter.HttpMessageConverter;
import org.springframework.http.converter.json.MappingJackson2HttpMessageConverter;
import org.springframework.web.client.RestTemplate;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.JCommander;



import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;


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
public class YOLO2_TF_Client {


    @Parameter(names="--endpoint", description="Endpoint for classification", required=true)
    private String skilInferenceEndpoint = ""; // EXAMPLE: "http://localhost:9008/endpoints/mnist/model/mnistmodel/default/";

    @Parameter(names="--input", description="image input file", required=true)
    private String inputImageFile = "";

/*
    public static void main(String... args) throws Exception {
        AuthClient authClient = new AuthClient("http://localhost:9008");
        authClient.login("admin", "admin");
        String authToken = authClient.getAuthToken();

        String basePath = "http://host:9008/endpoints/tf_models/model/yolo2/default";
        //String basePath = "http://localhost:9008/endpoints/foo/model/bar/default";
        //String basePath = "http://localhost:9602";
        PredictServiceClient client = new PredictServiceClient(basePath);
        client.setAuthToken(authToken);

        INDArray black = Nd4j.zeros(1, 608, 608, 3);
        //INDArray eye = Nd4j.eye(28).reshape(1, 28 * 28);
        Prediction input = new Prediction(black, "black");

        long start = System.nanoTime();
        for (int i = 0; i < 1; i++) {
            JsonArrayResponse result = client.jsonArrayPredict(input);
            List<DetectedObject> predictedObjects = getPredictedObjects(result.getArray(), 0.6);

            for (DetectedObject o : predictedObjects) {
                System.out.println(o.toString());
            }
        }
        long end = System.nanoTime();

        System.out.println((end - start) / 1000000 + " ms");
    }
    */


    public void run() throws Exception, IOException {

        ImageTransformProcess imgTransformProcess = new ImageTransformProcess.Builder().seed(12345)
            .build();
        
        File imageFile = null;
        INDArray finalRecord = null;

        if ("blank".equals(inputImageFile)) {

            finalRecord = Nd4j.zeros( 1, 608, 608, 3 );

            System.out.println( "Generating blank test image ..." );

        } else {

            imageFile = new File( inputImageFile );

            if (!imageFile.exists() || !imageFile.isFile()) {
                System.err.format("unable to access file %s\n", inputImageFile);
                System.exit(2);
            } else {

                System.out.println( "Inference for: " + inputImageFile );

            }

            ImageWritable img = imgTransformProcess.transformFileUriToInput( imageFile.toURI() );
            finalRecord = imgTransformProcess.executeArray( img ).reshape(1, 28 * 28);

        }

        String imgBase64 = Nd4jBase64.base64String(finalRecord);

        System.out.println( imgBase64 );  

        System.out.println( "Finished image conversion" );

        skilClientGetImageInference( imgBase64 );

    }    



    private void skilClientGetImageInference( String imgBase64 ) {

        Authorization auth = new Authorization();
        String auth_token = auth.getAuthToken( "admin", "admin" );

        System.out.println( "auth token: " + auth_token );


        System.out.println( "\n\n\n\nNow Sending the Classification Payload......\n\n\n" );

        try {

            String returnVal =
                    Unirest.post( skilInferenceEndpoint + "predict" ) 
                            .header("accept", "application/json")
                            .header("Content-Type", "application/json")
                            .header( "Authorization", "Bearer " + auth_token)
                            .body(new JSONObject() //Using this because the field functions couldn't get translated to an acceptable json
                                    .put( "id", "some_id" )
                                    .put("prediction", new JSONObject().put("array", imgBase64))
                                    .toString())
                            .asJson()
                            .getBody().getObject().toString(); 


            System.out.println( "classification return: " + returnVal );

        } catch (UnirestException e) {
            e.printStackTrace();
        }

    }



/*
    public static List<DetectedObject> getPredictedObjects(INDArray networkOutput, double threshold){
        if(networkOutput.rank() != 4){
            throw new IllegalStateException("Invalid network output activations array: should be rank 4. Got array "
                    + "with shape " + Arrays.toString(networkOutput.shape()));
        }
        if(threshold < 0.0 || threshold > 1.0){
            throw new IllegalStateException("Invalid threshold: must be in range [0,1]. Got: " + threshold);
        }

        //Activations format: [mb, 5b+c, h, w]
        int mb = networkOutput.size(0);
        int h = networkOutput.size(1);
        int w = networkOutput.size(2);
        int b = 17;
        int c = (networkOutput.size(3)/b)-5;  //input.size(1) == b * (5 + C) -> C = (input.size(1)/b) - 5

        //Reshape from [minibatch, B*(5+C), H, W] to [minibatch, B, 5+C, H, W] to [minibatch, B, 5, H, W]
        INDArray output5 = networkOutput.dup('c').reshape(mb, b, 5+c, h, w);
        INDArray predictedConfidence = output5.get(all(), all(), point(4), all(), all());    //Shape: [mb, B, H, W]
        INDArray softmax = output5.get(all(), all(), interval(5, 5+c), all(), all());

        List<DetectedObject> out = new ArrayList<>();
        for( int i=0; i<mb; i++ ){
            for( int x=0; x<w; x++ ){
                for( int y=0; y<h; y++ ){
                    for( int box=0; box<b; box++ ){
                        double conf = predictedConfidence.getDouble(i, box, y, x);
                        if(conf < threshold){
                            continue;
                        }

                        double px = output5.getDouble(i, box, 0, y, x); //Originally: in 0 to 1 in grid cell
                        double py = output5.getDouble(i, box, 1, y, x); //Originally: in 0 to 1 in grid cell
                        double pw = output5.getDouble(i, box, 2, y, x); //In grid units (for example, 0 to 13)
                        double ph = output5.getDouble(i, box, 3, y, x); //In grid units (for example, 0 to 13)

                        //Convert the "position in grid cell" to "position in image (in grid cell units)"
                        px += x;
                        py += y;


                        INDArray sm;
                        try (MemoryWorkspace wsO = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                            sm = softmax.get(point(i), point(box), all(), point(y), point(x)).dup();
                        }

                        out.add(new DetectedObject(i, px, py, pw, ph, sm, conf));
                    }
                }
            }
        }

        return out;
    }
*/




    public static void main(String[] args) throws Exception {
        YOLO2_TF_Client m = new YOLO2_TF_Client();

        JCommander.newBuilder()
          .addObject(m)
          .build()
          .parse(args);

        m.run();
    }



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
