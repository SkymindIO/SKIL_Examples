package ai.skymind.skil.examples.mnist.modelserver.inference;

import org.datavec.image.data.ImageWritable;

import org.datavec.api.util.ClassPathResource;
import org.datavec.image.transform.ImageTransformProcess;
import org.datavec.spark.transform.model.Base64NDArrayBody;
import org.datavec.spark.transform.model.BatchImageRecord;
import org.datavec.spark.transform.model.SingleImageRecord;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.serde.base64.Nd4jBase64;

import ai.skymind.skil.examples.mnist.modelserver.inference.model.TransformedImage;


import com.mashape.unirest.http.JsonNode;
import com.mashape.unirest.http.ObjectMapper;
import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.exceptions.UnirestException;
import org.json.JSONObject;
import java.io.IOException;

import ai.skymind.skil.examples.mnist.modelserver.inference.model.Inference;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.converter.HttpMessageConverter;
import org.springframework.http.converter.json.MappingJackson2HttpMessageConverter;
import org.springframework.web.client.RestTemplate;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.JCommander;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import ai.skymind.skil.examples.mnist.modelserver.auth.Authorization;

public class MNISTModelServerInferenceExample {


    @Parameter(names="--endpoint", description="Endpoint for classification", required=true)
    private String skilInferenceEndpoint = ""; // EXAMPLE: "http://localhost:9008/endpoints/mnist/model/mnistmodel/default/";

    @Parameter(names="--input", description="image input file", required=true)
    private String inputImageFile = "";

    public void run() throws Exception, IOException {

        ImageTransformProcess imgTransformProcess = new ImageTransformProcess.Builder().seed(12345)
            .build();
        
        File imageFile = null;
        INDArray finalRecord = null;

        if ("blank".equals(inputImageFile)) {

            finalRecord = Nd4j.zeros( 1, 28 * 28 );

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

        try {

            String returnVal =
                    Unirest.post( skilInferenceEndpoint + "multiclassify" ) 
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

    public static void main(String[] args) throws Exception {
        MNISTModelServerInferenceExample m = new MNISTModelServerInferenceExample();

        JCommander.newBuilder()
          .addObject(m)
          .build()
          .parse(args);

        m.run();
    }
}
