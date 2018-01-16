package ai.skymind.skil.examples.yolo2.modelserver.inference;

/*
import io.skymind.auth.AuthClient;
import io.skymind.skil.predict.client.PredictServiceClient;
import io.skymind.skil.service.model.ClassificationResult;
import io.skymind.skil.service.model.JsonArrayResponse;
import io.skymind.skil.service.model.MultiClassClassificationResult;
import io.skymind.skil.service.model.Prediction;
*/

import ai.skymind.skil.examples.yolo2.modelserver.inference.DetectedObject;

import org.datavec.image.data.ImageWritable;

import org.datavec.api.util.ClassPathResource;
import org.datavec.image.transform.ImageTransformProcess;
import org.datavec.spark.transform.model.Base64NDArrayBody;
import org.datavec.spark.transform.model.BatchImageRecord;
import org.datavec.spark.transform.model.SingleImageRecord;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.serde.base64.Nd4jBase64;

import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
//import org.nd4j.linalg.factory.Broadcast;

//import ai.skymind.skil.examples.mnist.modelserver.inference.model.TransformedImage;


import com.mashape.unirest.http.JsonNode;
//import com.mashape.unirest.http.JsonArray;
import com.mashape.unirest.http.ObjectMapper;
import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.exceptions.UnirestException;
import org.json.JSONObject;
import org.json.JSONArray;
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

import org.nd4j.linalg.ops.transforms.Transforms;


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

        //System.out.println( imgBase64 );  

        System.out.println( "Finished image conversion" );

        skilClientGetImageInference( imgBase64 );

    }    



    private void skilClientGetImageInference( String imgBase64 ) throws Exception, IOException  {

        Authorization auth = new Authorization();
        String auth_token = auth.getAuthToken( "admin", "admin" );

        System.out.println( "auth token: " + auth_token );


        System.out.println( "\n\n\n\nNow Sending the Classification Payload......\n\n\n" );

        try {

//            String returnVal =

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


            //System.out.println( "return data: " + returnJSONObject.toString() + "\n\n" );

            // extract fields from the object
//            String predict_return_array = returnJSONObject.getString("array");
            

            String predict_return_array = returnJSONObject.getJSONObject("prediction").getString("array");


            System.out.println( "classification return length: " + predict_return_array.length() );

            //System.out.println( "return data: " + returnVal + "\n\n" );

            INDArray networkOutput = Nd4jBase64.fromBase64( predict_return_array );




            // TODO: before we can extract detected objects, we need to apply activation functions




            List<DetectedObject> list_objects = getPredictedObjects( networkOutput, 0.7 );

            System.out.println( "Objects found: " + list_objects.size() );

//System.out.println( "return data: " + predict_return_array + "\n\n" );
/*
            JSONArray jsonarray = returnJSONObject.getJSONArray();

//String[] names = new String[jsonArray.length()];    
//String[] formattedNames = new String[jsonArray.length()];  

            for(int i=0;i<jsonArray.length();i++)
            {

                System.out.println( "Objects found: " + jsonArray.getJSONObject(i) );

            }
*/

        } catch (UnirestException e) {
            e.printStackTrace();
        }

    }



    /**
     * Output (loss) layer for YOLOv2 object detection model, based on the papers:
     * YOLO9000: Better, Faster, Stronger - Redmon & Farhadi (2016) - https://arxiv.org/abs/1612.08242<br>
     * and<br>
     * You Only Look Once: Unified, Real-Time Object Detection - Redmon et al. (2016) -
     * http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf<br>
     * <br>
     * This loss function implementation is based on the YOLOv2 version of the paper. However, note that it doesn't
     * currently support simultaneous training on both detection and classification datasets as described in the
     * YOlO9000 paper.<br>
     * <br>
     * Label format: [minibatch, 4+C, H, W]<br>
     * Order for labels depth: [x1,y1,x2,y2,(class labels)]<br>
     * x1 = box top left position<br>
     * y1 = as above, y axis<br>
     * x2 = box bottom right position<br>
     * y2 = as above y axis<br>
     * Note: labels are represented as a multiple of grid size - for a 13x13 grid, (0,0) is top left, (13,13) is bottom right<br>
     * <br>
     * Input format: [minibatch, B*(5+C), H, W]    ->      Reshape to [minibatch, B, 5+C, H, W]<br>
     * B = number of bounding boxes (determined by config)<br>
     * C = number of classes<br>
     * H = output/label height<br>
     * W = output/label width<br>
     * <br>
     * Note that mask arrays are not required - this implementation infers the presence or absence of objects in each grid
     * cell from the class labels (which should be 1-hot if an object is present, or all 0s otherwise).
     *
     * @author Alex Black
     */
    private INDArray activate(INDArray input, boolean training, int numberOfBoundingBoxes) {
        //Essentially: just apply activation functions...


        // ---- get the base variables -------------
        int mb = input.size(0);         // minibatch?
        int h = input.size(2);          // output/label height
        int w = input.size(3);          // output/label width
        int b = numberOfBoundingBoxes; 
        int c = (input.size(1)/b)-5;  //input.size(1) == b * (5 + C) -> C = (input.size(1)/b) - 5 // number of classes

        // ---- start computing intermediate stuff -------
        INDArray output = Nd4j.create(input.shape(), 'c');
        INDArray output5 = output.reshape('c', mb, b, 5+c, h, w);
        INDArray output4 = output;  //output.get(all(), interval(0,5*b), all(), all());
        INDArray input4 = input.dup('c');    //input.get(all(), interval(0,5*b), all(), all()).dup('c');
        INDArray input5 = input4.reshape('c', mb, b, 5+c, h, w);

        //X/Y center in grid: sigmoid
        INDArray predictedXYCenterGrid = input5.get(all(), all(), interval(0,2), all(), all());
        Transforms.sigmoid(predictedXYCenterGrid, false);

        //width/height: prior * exp(input)
        INDArray predictedWHPreExp = input5.get(all(), all(), interval(2,4), all(), all());
        INDArray predictedWH = Transforms.exp(predictedWHPreExp, false);

// TODO: commented out due to issues w missing Broadcast entry
//        Broadcast.mul(predictedWH, layerConf().getBoundingBoxes(), predictedWH, 1, 2);  //Box priors: [b, 2]; predictedWH: [mb, b, 2, h, w]

        //Confidence - sigmoid
        INDArray predictedConf = input5.get(all(), all(), point(4), all(), all());   //Shape: [mb, B, H, W]
        Transforms.sigmoid(predictedConf, false);

        output4.assign(input4);

        //Softmax
        //TODO OPTIMIZE?
        INDArray inputClassesPreSoftmax = input5.get(all(), all(), interval(5, 5+c), all(), all());   //Shape: [minibatch, C, H, W]
        INDArray classPredictionsPreSoftmax2d = inputClassesPreSoftmax.permute(0,1,3,4,2) //[minibatch, b, c, h, w] To [mb, b, h, w, c]
                .dup('c').reshape('c', new int[]{mb*b*h*w, c});
        Transforms.softmax(classPredictionsPreSoftmax2d, false);
        INDArray postSoftmax5d = classPredictionsPreSoftmax2d.reshape('c', mb, b, h, w, c ).permute(0, 1, 4, 2, 3);

        INDArray outputClasses = output5.get(all(), all(), interval(5, 5+c), all(), all());   //Shape: [minibatch, C, H, W]
        outputClasses.assign(postSoftmax5d);

        return output;
    }

    /*

        code to parse individual bounding boxes from network output

    */
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
