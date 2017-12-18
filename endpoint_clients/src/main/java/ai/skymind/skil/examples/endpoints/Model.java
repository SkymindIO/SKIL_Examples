package ai.skymind.skil.examples.endpoints;

import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.exceptions.UnirestException;
import org.json.JSONArray;
import org.json.JSONObject;

import java.text.MessageFormat;
import java.util.ArrayList;
import java.util.List;

public class Model {

    private String host;
    private String port;
    private int deploymentID;

    public Model(int deploymentID) {
        this.host = "localhost";
        this.port = "9008";
        this.deploymentID = deploymentID;
    }

    public Model(String host, String port, int deploymentID) {
        this.host = host;
        this.port = port;
        this.deploymentID = deploymentID;
    }

    public JSONObject addModel(String name, String fileLocation, int scale, String uri) {
        JSONObject model = new JSONObject();

        try {
            List<String> uriList = new ArrayList<String>();
            uriList.add(uri);

            model =
                    Unirest.post(MessageFormat.format("http://{0}:{1}/deployment/{2}/model", host, port, deploymentID))
                            .header("accept", "application/json")
                            .header("Content-Type", "application/json")
                            .body(new JSONObject()
                                    .put("name", name)
                                    .put("modelType", "model")
                                    .put("fileLocation", fileLocation)
                                    .put("scale", scale)
                                    .put("uri", uriList)
                                    .toString())
                            .asJson()
                            .getBody().getObject();
        } catch (UnirestException e) {
            e.printStackTrace();
        }

        return model;
    }

    public JSONObject reimportModel(int modelID, String name, String fileLocation) {
        JSONObject model = new JSONObject();

        try {
            model =
                    Unirest.post(MessageFormat.format("http://{0}:{1}/deployment/{2}/model/{3}", host, port, deploymentID, modelID))
                            .header("accept", "application/json")
                            .header("Content-Type", "application/json")
                            .body(new JSONObject()
                                    .put("name", name)
                                    .put("modelType", "model")
                                    .put("fileLocation", fileLocation)
                                    .toString())
                            .asJson()
                            .getBody().getObject();
        } catch (UnirestException e) {
            e.printStackTrace();
        }

        return model;
    }

    public JSONObject deleteModel(int modelID) {
        JSONObject model = new JSONObject();

        try {
            model =
                    Unirest.delete(MessageFormat.format("http://{0}:{1}/deployment/{2}/model/{3}", host, port, deploymentID, modelID))
                            .header("accept", "application/json")
                            .header("Content-Type", "application/json")
                            .asJson()
                            .getBody().getObject();
        } catch (UnirestException e) {
            e.printStackTrace();
        }

        return model;
    }

    public JSONObject setModelState(int modelID, String state) {
        JSONObject model = new JSONObject();

        try {
            model =
                    Unirest.delete(MessageFormat.format("http://{0}:{1}/deployment/{2}/model/{3}/state", host, port, deploymentID, modelID))
                            .header("accept", "application/json")
                            .header("Content-Type", "application/json")
                            .body(new JSONObject()
                                    .put("name", state)
                                    .toString())
                            .asJson()
                            .getBody().getObject();
        } catch (UnirestException e) {
            e.printStackTrace();
        }

        return model;
    }
}
