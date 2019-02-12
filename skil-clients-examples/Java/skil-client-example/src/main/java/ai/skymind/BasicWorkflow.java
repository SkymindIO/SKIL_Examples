package ai.skymind;

import ai.skymind.auth.ApiKeyAuth;
import ai.skymind.skil.DefaultApi;
import ai.skymind.skil.model.*;

import java.io.File;
import java.util.Arrays;
import java.util.Objects;

/**
 * Hello world!
 *
 */
public class BasicWorkflow 
{
    public static void main( String[] args ) {

        ApiClient apiClient = new ApiClient();
        apiClient.setBasePath("http://localhost:9008"); // Replace this with the host and port of your SKIL server, if required.

        DefaultApi apiInstance = new DefaultApi(apiClient);

        String deploymentId = null;
        String modelId = null;

        try { // Login Successful
            LoginResponse loginResponse = apiInstance.login(
                    new LoginRequest()
                        .userId("admin")
                        .password("Password@123")
            );

            System.out.println(loginResponse);

            // Configure API key authorization: api_key
            ApiKeyAuth api_key = (ApiKeyAuth) apiClient.getAuthentication("api_key");
            api_key.setApiKeyPrefix("Bearer");
            api_key.setApiKey(loginResponse.getToken());

            DeploymentResponse deploymentResponse = apiInstance.deploymentCreate(
                    new CreateDeploymentRequest().name("new_deployment"));
            deploymentId = deploymentResponse.getId();

            System.out.println("\n\nDeployment Created with ID: " + deploymentResponse.getId());

            String modelFilePath = BasicWorkflow.class.getClassLoader().getResource("keras_cifar10_trained_model.h5").getPath();
            FileUploadList fileUploadList = apiInstance.upload(new File(modelFilePath));

            String serverFilePath = "file://" + fileUploadList.getFileUploadResponseList().get(0).getPath();
            System.out.println("File saved in server at path: " + serverFilePath);

            String modelName = "new_model";

            ModelEntity modelEntity = apiInstance.deployModel(deploymentId,
                new ImportModelRequest()
                    .name(modelName)
                    .uri(Arrays.asList(
                        deploymentResponse.getDeploymentSlug() + "/model/" + modelName + "/default",
                        deploymentResponse.getDeploymentSlug() + "/model/" + modelName + "/v1"))
                    .modelType("model")
                    .scale(1)
                    .fileLocation(serverFilePath)
                );
            modelId = modelEntity.getId().toString();

            System.out.println("Model deployed: ");
            System.out.println(modelEntity);

            System.out.print("Waiting for model server to start.");
            do {
                modelEntity = apiInstance.modelStateChange(
                        deploymentId,
                        modelId,
                        new SetState().state(SetState.StateEnum.START));

                Thread.sleep(5000);
                Thread.sleep(5000);
                System.out.print(".");
            } while (modelEntity.getState() != ModelEntity.StateEnum.STARTED);
            System.out.println("\nModel server started successfully!");

            Thread.sleep(5000);

            String testImageFilePath = BasicWorkflow.class.getClassLoader().getResource("frog.jpg").getPath();
            File testImageFile = new File(testImageFilePath);

            System.out.println(
                    apiInstance.predictimage(
                            deploymentResponse.getDeploymentSlug(), "default", modelName, testImageFile
                    )
            );
        } catch (Exception e) {
            e.printStackTrace();
        } finally { // Clean up resources
            try {
                System.out.println("\nCleaning up resources...");
                if (deploymentId != null && modelId != null) {
                    ModelEntity modelEntity;
                    System.out.print("Waiting for model server to stop.");
                    do {
                        modelEntity = apiInstance.modelStateChange(
                                deploymentId,
                                modelId,
                                new SetState().state(SetState.StateEnum.STOP));

                        Thread.sleep(5000);
                        System.out.print(".");
                    } while (modelEntity.getState() != ModelEntity.StateEnum.STOPPED);
                    System.out.println("\nModel server stopped successfully!");

                    System.out.println("\nDeleting model!");
                    System.out.println(apiInstance.deleteModel(deploymentId, modelId));
                }

                if(deploymentId != null) {
                    System.out.println("\nDeleting deployment");
                    System.out.println(apiInstance.deploymentDelete(deploymentId));
                }
            } catch (Exception e) {
                System.err.println("Error while cleaning up resources.");
                e.printStackTrace();
            }
        }
    }
}
