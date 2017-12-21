package ai.skymind.skil.examples.skil_client;

import io.skymind.auth.rest.LoginResponse;
import io.skymind.deployment.client.SKILDeploymentClient;

import java.text.MessageFormat;

public class Main {
    public static void main(String[] args) {

        String host = "localhost";
        String port = "9008";

        String userId = "admin";
        String password = "admin";

        String path = MessageFormat.format("http://{0}:{1}", host, port);

        //Set these variables to the file locations in the server before running this example.
        String modelFileLocation = "file:///tmp/model.zip",
                reimportedModelFileLocation = "file:///tmp/reimported_model.zip";
        String transformFileLocation = "file:///tmp/transform.json",
                reimportedTransformFileLocation = "file:///tmp/reimported_transform.json";
        String knnFileLocation = "file:///tmp/knn.bin",
                reimportedKnnFileLocation = "file:///tmp/reimported_knn.bin";

        SKILDeploymentClient skilDeploymentClient = new SKILDeploymentClient(path);
        /*---------------------------------------------------------------------------------*/
        /*--------------------------------AUTH ENDPOINTS-----------------------------------*/
        /*---------------------------------------------------------------------------------*/

        /* Getting authorized in the deployment client */
        // The following code will authenticate you as a user in the SKIL Server
        LoginResponse loginResponse = skilDeploymentClient.login(userId, password);
        // To get the token
        print(MessageFormat.format("Authorization Token: {0}", loginResponse.getToken()));

        /*---------------------------------------------------------------------------------*/
        /*--------------------------------DEPLOYMENT ENDPOINTS-----------------------------*/
        /*---------------------------------------------------------------------------------*/


        /*---------------------------------------------------------------------------------*/
        /*--------------------------------MODEL ENDPOINTS-----------------------------*/
        /*---------------------------------------------------------------------------------*/


        /*---------------------------------------------------------------------------------*/
        /*--------------------------------TRANSFORM ENDPOINTS-----------------------------*/
        /*---------------------------------------------------------------------------------*/


        /*---------------------------------------------------------------------------------*/
        /*--------------------------------KNN ENDPOINTS-----------------------------*/
        /*---------------------------------------------------------------------------------*/

    }

    private static void print(Object x) {
        System.out.println(x);
    }
}