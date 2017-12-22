package ai.skymind.skil.examples.deployment_client;

import io.skymind.auth.rest.LoginResponse;
import io.skymind.deployment.client.SKILDeploymentClient;
import io.skymind.deployment.model.ModelEntity;
import io.skymind.deployment.rest.CreateDeploymentRequest;
import io.skymind.deployment.rest.DeploymentResponse;
import io.skymind.deployment.rest.ImportModelRequest;
import io.skymind.deployment.rest.UpdateModelStateRequest;
import io.skymind.skil.client.errors.ClientException;

import java.text.MessageFormat;
import java.util.List;

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
        // The following code will authenticate you as a user in the SKIL Server (Not required for all requests)
        LoginResponse loginResponse = skilDeploymentClient.login(userId, password);

        /*---------------------------------------------------------------------------------*/
        /*--------------------------------DEPLOYMENT ENDPOINTS-----------------------------*/
        /*---------------------------------------------------------------------------------*/
        // Adding a new deployment
        String newDeploymentName = "New-Deployment";
        DeploymentResponse addedDeploymentResponse = null;
        try {
            addedDeploymentResponse = skilDeploymentClient.addDeployment(new CreateDeploymentRequest(newDeploymentName));
        } catch (ClientException e) {
            e.printStackTrace();
        }

        // Getting the list of deployments in the SKIL server
        List<DeploymentResponse> deployments = skilDeploymentClient.getDeployments();

        // Getting a deployment by ID
        long deploymentId = Long.parseLong(deployments.get(0).getId());
        DeploymentResponse deployment = skilDeploymentClient.getDeployment(deploymentId);

        /*---------------------------------------------------------------------------------*/
        /*--------------------------------MODEL ENDPOINTS----------------------------------*/
        /*---------------------------------------------------------------------------------*/

        // Adding a new model to a deployment
        String newModelName = "New-Model";
        String singleEndpointUrl = "new_model_endpoint";
        ModelEntity addedModelEntity = null;
        try {
            addedModelEntity = skilDeploymentClient.addModel(deploymentId,
                    ImportModelRequest.builder()
                            .name(newModelName)
                            .fileLocation(modelFileLocation)
                            .scale(1)
                            .uri(singleEndpointUrl)
                            .build());
        } catch (ClientException e) {
            e.printStackTrace();
        }

        // Getting all the models in a deployment
        List<ModelEntity> modelEntities = skilDeploymentClient.getModels(deploymentId);

        // Reimporting a model to a model ID in a deployment
        String reimportedModelName = "Reimported-Model";
        ModelEntity reimportedModelEntity = null;
        if (addedModelEntity != null) {
            reimportedModelEntity = skilDeploymentClient.reimportModel(
                    deploymentId,
                    (Long) addedModelEntity.getId(),
                    ImportModelRequest.builder()
                            .name(reimportedModelName)
                            .build());
        }

        // Setting model state
        ModelEntity stateChangedModel = null;
        if (addedModelEntity != null) {
            stateChangedModel = skilDeploymentClient.setModelState(
                    deploymentId,
                    (Long) addedModelEntity.getId(),
                    UpdateModelStateRequest.builder()
                            .state(ModelEntity.SetState.STOP)
                            .build()
            );
        }

        String deletedModelDetails = null;
        if (addedModelEntity != null) {
            deletedModelDetails = skilDeploymentClient.deleteModel(
                    deploymentId,
                    (Long) addedModelEntity.getId()
            );
        }

        /*---------------------------------------------------------------------------------*/
        /*--------------------------------TRANSFORM ENDPOINTS------------------------------*/
        /*---------------------------------------------------------------------------------*/


        /*---------------------------------------------------------------------------------*/
        /*------------------------------------KNN ENDPOINTS--------------------------------*/
        /*---------------------------------------------------------------------------------*/


        /*---------------------------------------------------------------------------------*/
        /*--------------------------------Printing Responses-------------------------------*/
        /*---------------------------------------------------------------------------------*/

        print("----------------------------------------------------------------------------");
        print("------------------------------Authorization---------------------------------");
        print("----------------------------------------------------------------------------");

        // To get the token
        print(MessageFormat.format("Authorization Token: {0}", loginResponse.getToken()));

        print("----------------------------------------------------------------------------");
        print("------------------------------Deployments-----------------------------------");
        print("----------------------------------------------------------------------------");

        print("\n\n----------------------------Listing all deployments----------------------------------");
        print(MessageFormat.format("Number of Deployments: {0}", deployments.size()));

        if (addedDeploymentResponse != null) {
            print("\n\n----------------------------Added Deployment----------------------------------");
            print(MessageFormat.format("Deployment ID: {0} | Name: {1} | Slug: {2} | Status: {3}" +
                            " | Other Details:\n{4}", addedDeploymentResponse.getId(),
                    addedDeploymentResponse.getName(),
                    addedDeploymentResponse.getDeploymentSlug(),
                    addedDeploymentResponse.getStatus(),
                    addedDeploymentResponse.getBody()));
        }

        print("\n\n----------------------------Deployment by ID----------------------------------");
        print(MessageFormat.format("Deployment ID: {0} | Name: {1} | Slug: {2} | Status: {3}" +
                        " | Other Details:\n{4}", deploymentId,
                deployment.getName(),
                deployment.getDeploymentSlug(),
                deployment.getStatus(),
                deployment.getBody()));

        print("----------------------------------------------------------------------------");
        print("------------------------------Models-----------------------------------");
        print("----------------------------------------------------------------------------");
    }

    private static void print(Object x) {
        System.out.println(x);
    }
}