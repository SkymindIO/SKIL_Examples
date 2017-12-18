package ai.skymind.skil.examples;

import ai.skymind.skil.examples.endpoints.*;
import org.json.JSONArray;
import org.json.JSONObject;

import java.text.MessageFormat;

public class Main {
    public static void main(String[] args) {
        //Set these variables to the file locations in the server before running this example.
        String modelFileLocation = null,
                reimportedModelFileLocation = null;
        String transformFileLocation = null,
                reimportedTransformFileLocation = null;
        String knnFileLocation = null,
                reimportedKnnFileLocation = null;

        /*---------------------------------------------------------------------------------*/
        /*--------------------------------AUTH ENDPOINTS-----------------------------------*/
        /*---------------------------------------------------------------------------------*/
        Authorization authorization = new Authorization();

        String authToken = authorization.getAuthToken("admin", "admin");
        System.out.println(MessageFormat.format("Auth Token: {0}", authToken));

        /*---------------------------------------------------------------------------------*/
        /*--------------------------------DEPLOYMENT ENDPOINTS-----------------------------*/
        /*---------------------------------------------------------------------------------*/
        Deployment deployment = new Deployment();

        //There can be only 2 deployments in SKIL CE
        JSONObject addedDeployment = deployment.addDeployment("New deployment");
        System.out.println(addedDeployment.toString(4));

        JSONArray deployments = deployment.getAllDeployments();
        System.out.println(deployments.toString(4));

        for (int i = 0; i < deployments.length(); i++) {
            int id = deployments.getJSONObject(i).getInt("id");
            JSONObject deploymentById = deployment.getDeploymentById(id);
            System.out.println(deploymentById.toString(4));
        }

        for (int i = 0; i < deployments.length(); i++) {
            int id = deployments.getJSONObject(i).getInt("id");
            JSONArray knnsForDeployment = deployment.getModelsForDeployment(id);
            System.out.println(knnsForDeployment.toString(4));
        }

        /*---------------------------------------------------------------------------------*/
        /*--------------------------------MODEL ENDPOINTS-----------------------------*/
        /*---------------------------------------------------------------------------------*/
        Model model = new Model(deployments.getJSONObject(0).getInt("id"));

        JSONObject addedModel = model.addModel("new_model", modelFileLocation, 1, "new_model_endpoint");
        System.out.println(addedModel.toString(4));

        int modelID = addedModel.getInt("id");
        JSONObject reimportedModel = model.reimportModel(modelID,"reimported_model", reimportedModelFileLocation);
        System.out.println(reimportedModel.toString(4));

        JSONObject setModelState = model.setModelState(modelID, "stop");
        System.out.println(setModelState.toString(4));

        JSONObject deletedModel = model.deleteModel(modelID);
        System.out.println(deletedModel.toString(4));

        /*---------------------------------------------------------------------------------*/
        /*--------------------------------TRANSFORM ENDPOINTS-----------------------------*/
        /*---------------------------------------------------------------------------------*/
        Transform transform = new Transform(deployments.getJSONObject(0).getInt("id"));

        JSONObject addedTransform = transform.addTransform("new_transform", transformFileLocation, 1, "new_transform_endpoint");
        System.out.println(addedTransform.toString(4));

        int transformID = addedTransform.getInt("id");
        JSONObject reimportedTransform = transform.reimportTransform(transformID,"reimported_transform", reimportedTransformFileLocation);
        System.out.println(reimportedTransform.toString(4));

        JSONObject setTransformState = transform.setTransformState(transformID, "stop");
        System.out.println(setTransformState.toString(4));

        JSONObject deletedTransform = transform.deleteTransform(transformID);
        System.out.println(deletedTransform.toString(4));

        /*---------------------------------------------------------------------------------*/
        /*--------------------------------KNN ENDPOINTS-----------------------------*/
        /*---------------------------------------------------------------------------------*/
        KNN knn = new KNN(deployments.getJSONObject(0).getInt("id"));

        JSONObject addedKnn = knn.addKNN("new_knn", knnFileLocation, 1, "new_knn_endpoint");
        System.out.println(addedKnn.toString(4));

        int knnID = addedKnn.getInt("id");
        JSONObject reimportedKnn = knn.reimportKNN(knnID,"reimported_knn", reimportedKnnFileLocation);
        System.out.println(reimportedKnn.toString(4));

        JSONObject setKnnState = knn.setKNNState(knnID, "stop");
        System.out.println(setKnnState.toString(4));

        JSONObject deletedKnn = knn.deleteKNN(knnID);
        System.out.println(deletedKnn.toString(4));
    }
}