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
        String singleModelEndpointUrl = "new_model_endpoint";
        ModelEntity addedModelEntity = null;
        try {
            addedModelEntity = skilDeploymentClient.addModel(deploymentId,
                    ImportModelRequest.builder()
                            .name(newModelName)
                            .fileLocation(modelFileLocation)
                            .scale(1)
                            .uri(singleModelEndpointUrl)
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
                            .fileLocation(reimportedModelFileLocation)
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

        // Deleting the added model
        String deletedModelResponse = null;
        if (addedModelEntity != null) {
            deletedModelResponse = skilDeploymentClient.deleteModel(
                    deploymentId,
                    (Long) addedModelEntity.getId()
            );
        }

        /*---------------------------------------------------------------------------------*/
        /*--------------------------------TRANSFORM ENDPOINTS------------------------------*/
        /*---------------------------------------------------------------------------------*/

        // Adding a new transform to a deployment
        String newTransformName = "New-Transform";
        String singleTransformEndpointUrl = "new_transform_endpoint";
        ModelEntity addedTransformEntity = null;
        try {
            addedTransformEntity = skilDeploymentClient.addModel(deploymentId,
                    ImportModelRequest.builder()
                            .name(newTransformName)
                            .fileLocation(transformFileLocation)
                            .scale(1)
                            .subType(ModelEntity.ModelType.TRANSFORM.name())
                            .uri(singleTransformEndpointUrl)
                            .build());
        } catch (ClientException e) {
            e.printStackTrace();
        }

        // Reimporting a transform to a model ID in a deployment
        String reimportedTransformName = "Reimported-Transform";
        ModelEntity reimportedTransformEntity = null;
        if (addedTransformEntity != null) {
            reimportedTransformEntity = skilDeploymentClient.reimportModel(
                    deploymentId,
                    (Long) addedTransformEntity.getId(),
                    ImportModelRequest.builder()
                            .name(reimportedTransformName)
                            .fileLocation(reimportedTransformFileLocation)
                            .build());
        }

        // Setting transform state
        ModelEntity stateChangedTransform = null;
        if (addedTransformEntity != null) {
            stateChangedTransform = skilDeploymentClient.setModelState(
                    deploymentId,
                    (Long) addedTransformEntity.getId(),
                    UpdateModelStateRequest.builder()
                            .state(ModelEntity.SetState.STOP)
                            .build()
            );
        }

        // Deleting the added transform
        String deletedTransformResponse = null;
        if (addedTransformEntity != null) {
            deletedTransformResponse = skilDeploymentClient.deleteModel(
                    deploymentId,
                    (Long) addedTransformEntity.getId()
            );
        }

        /*---------------------------------------------------------------------------------*/
        /*------------------------------------KNN ENDPOINTS--------------------------------*/
        /*---------------------------------------------------------------------------------*/

        // Adding a new knn to a deployment
        String newKnnName = "New-Knn";
        String singleKnnEndpointUrl = "new_knn_endpoint";
        ModelEntity addedKnnEntity = null;
        try {
            addedKnnEntity = skilDeploymentClient.addModel(deploymentId,
                    ImportModelRequest.builder()
                            .name(newKnnName)
                            .fileLocation(knnFileLocation)
                            .scale(1)
                            .subType(ModelEntity.ModelType.KNN.name())
                            .uri(singleKnnEndpointUrl)
                            .build());
        } catch (ClientException e) {
            e.printStackTrace();
        }

        // Reimporting a knn to a model ID in a deployment
        String reimportedKnnName = "Reimported-Knn";
        ModelEntity reimportedKnnEntity = null;
        if (addedKnnEntity != null) {
            reimportedKnnEntity = skilDeploymentClient.reimportModel(
                    deploymentId,
                    (Long) addedKnnEntity.getId(),
                    ImportModelRequest.builder()
                            .name(reimportedKnnName)
                            .fileLocation(reimportedKnnFileLocation)
                            .build());
        }

        // Setting knn state
        ModelEntity stateChangedKnn = null;
        if (addedKnnEntity != null) {
            stateChangedKnn = skilDeploymentClient.setModelState(
                    deploymentId,
                    (Long) addedKnnEntity.getId(),
                    UpdateModelStateRequest.builder()
                            .state(ModelEntity.SetState.STOP)
                            .build()
            );
        }

        // Deleting the added knn
        String deletedKnnResponse = null;
        if (addedKnnEntity != null) {
            deletedKnnResponse = skilDeploymentClient.deleteModel(
                    deploymentId,
                    (Long) addedKnnEntity.getId()
            );
        }

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
                        " | Other Details:\n{4}",
                deploymentId,
                deployment.getName(),
                deployment.getDeploymentSlug(),
                deployment.getStatus(),
                deployment.getBody()));

        print("----------------------------------------------------------------------------");
        print("------------------------------Models----------------------------------------");
        print("----------------------------------------------------------------------------");

        if(addedModelEntity != null) {
            print("\n\n----------------------------Adding a model----------------------------------");
            print(MessageFormat.format("Model ID: {0} | Name: {1} | Scale: {2} | State: {3} " +
                            "| File location:\n{4}",
                    addedModelEntity.getId(),
                    addedModelEntity.getName(),
                    addedModelEntity.getScale(),
                    addedModelEntity.getState(),
                    addedModelEntity.getFileLocation()));

            print("\n\n----------------------------Reimporting a model-----------------------------");
            print(MessageFormat.format("Model ID: {0} | Name: {1} | Scale: {2} | State: {3} " +
                            "| File location:\n{4}",
                    reimportedModelEntity.getId(),
                    reimportedModelEntity.getName(),
                    reimportedModelEntity.getScale(),
                    reimportedModelEntity.getState(),
                    reimportedModelEntity.getFileLocation()));

            print("\n\n----------------------------Changing model state----------------------------");
            print(MessageFormat.format("Model ID: {0} | Name: {1} | Scale: {2} | State: {3} " +
                            "| File location:\n{4}",
                    stateChangedModel.getId(),
                    stateChangedModel.getName(),
                    stateChangedModel.getScale(),
                    stateChangedModel.getState(),
                    stateChangedModel.getFileLocation()));

            print("\n\n----------------------------Deleting a model--------------------------------");
            print("Response: " + deletedModelResponse);
        }

        print("\n\n----------------------------Listing all Models----------------------------------");
        print(MessageFormat.format("Number of models in deployment id '{0}' -> {1}", deploymentId, modelEntities.size()));

        print("----------------------------------------------------------------------------");
        print("------------------------------Transforms------------------------------------");
        print("----------------------------------------------------------------------------");

        if(addedTransformEntity != null) {
            print("\n\n----------------------------Adding a transform----------------------------------");
            print(MessageFormat.format("Model ID: {0} | Name: {1} | Scale: {2} | State: {3} " +
                            "| File location:\n{4}",
                    addedTransformEntity.getId(),
                    addedTransformEntity.getName(),
                    addedTransformEntity.getScale(),
                    addedTransformEntity.getState(),
                    addedTransformEntity.getFileLocation()));

            print("\n\n----------------------------Reimporting a transform-----------------------------");
            print(MessageFormat.format("Model ID: {0} | Name: {1} | Scale: {2} | State: {3} " +
                            "| File location:\n{4}",
                    reimportedTransformEntity.getId(),
                    reimportedTransformEntity.getName(),
                    reimportedTransformEntity.getScale(),
                    reimportedTransformEntity.getState(),
                    reimportedTransformEntity.getFileLocation()));

            print("\n\n----------------------------Changing transform state----------------------------");
            print(MessageFormat.format("Model ID: {0} | Name: {1} | Scale: {2} | State: {3} " +
                            "| File location:\n{4}",
                    stateChangedTransform.getId(),
                    stateChangedTransform.getName(),
                    stateChangedTransform.getScale(),
                    stateChangedTransform.getState(),
                    stateChangedTransform.getFileLocation()));

            print("\n\n----------------------------Deleting a transform--------------------------------");
            print("Response: " + deletedTransformResponse);
        }

        print("----------------------------------------------------------------------------");
        print("------------------------------KNNs------------------------------------------");
        print("----------------------------------------------------------------------------");

        if(addedKnnEntity != null) {
            print("\n\n----------------------------Adding a knn----------------------------------");
            print(MessageFormat.format("Model ID: {0} | Name: {1} | Scale: {2} | State: {3} " +
                            "| File location:\n{4}",
                    addedKnnEntity.getId(),
                    addedKnnEntity.getName(),
                    addedKnnEntity.getScale(),
                    addedKnnEntity.getState(),
                    addedKnnEntity.getFileLocation()));

            print("\n\n----------------------------Reimporting a knn-----------------------------");
            print(MessageFormat.format("Model ID: {0} | Name: {1} | Scale: {2} | State: {3} " +
                            "| File location:\n{4}",
                    reimportedKnnEntity.getId(),
                    reimportedKnnEntity.getName(),
                    reimportedKnnEntity.getScale(),
                    reimportedKnnEntity.getState(),
                    reimportedKnnEntity.getFileLocation()));

            print("\n\n----------------------------Changing knn state----------------------------");
            print(MessageFormat.format("Model ID: {0} | Name: {1} | Scale: {2} | State: {3} " +
                            "| File location:\n{4}",
                    stateChangedKnn.getId(),
                    stateChangedKnn.getName(),
                    stateChangedKnn.getScale(),
                    stateChangedKnn.getState(),
                    stateChangedKnn.getFileLocation()));

            print("\n\n----------------------------Deleting a knn--------------------------------");
            print("Response: " + deletedKnnResponse);
        }
    }

    private static void print(Object x) {
        System.out.println(x);
    }
}