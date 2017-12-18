package ai.skymind.skil.examples.endpoints;

import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.exceptions.UnirestException;
import org.json.JSONArray;
import org.json.JSONObject;

import java.text.MessageFormat;
import java.util.ArrayList;
import java.util.List;

public class KNN {

    private String host;
    private String port;
    private int deploymentID;

    public KNN(int deploymentID) {
        this.host = "localhost";
        this.port = "9008";
        this.deploymentID = deploymentID;
    }

    public KNN(String host, String port, int deploymentID) {
        this.host = host;
        this.port = port;
        this.deploymentID = deploymentID;
    }

    public JSONObject addKNN(String name, String fileLocation, int scale, String uri) {
        JSONObject knn = new JSONObject();

        try {
            List<String> uriList = new ArrayList<String>();
            uriList.add(uri);

            knn =
                    Unirest.post(MessageFormat.format("http://{0}:{1}/deployment/{2}/model", host, port, deploymentID))
                            .header("accept", "application/json")
                            .header("Content-Type", "application/json")
                            .body(new JSONObject()
                                    .put("name", name)
                                    .put("modelType", "knn")
                                    .put("fileLocation", fileLocation)
                                    .put("scale", scale)
                                    .put("uri", uriList)
                                    .toString())
                            .asJson()
                            .getBody().getObject();
        } catch (UnirestException e) {
            e.printStackTrace();
        }

        return knn;
    }

    public JSONObject reimportKNN(int knnID, String name, String fileLocation) {
        JSONObject knn = new JSONObject();

        try {
            knn =
                    Unirest.post(MessageFormat.format("http://{0}:{1}/deployment/{2}/model/{3}", host, port, deploymentID, knnID))
                            .header("accept", "application/json")
                            .header("Content-Type", "application/json")
                            .body(new JSONObject()
                                    .put("name", name)
                                    .put("modelType", "knn")
                                    .put("fileLocation", fileLocation)
                                    .toString())
                            .asJson()
                            .getBody().getObject();
        } catch (UnirestException e) {
            e.printStackTrace();
        }

        return knn;
    }

    public JSONObject deleteKNN(int knnID) {
        JSONObject knn = new JSONObject();

        try {
            knn =
                    Unirest.delete(MessageFormat.format("http://{0}:{1}/deployment/{2}/model/{3}", host, port, deploymentID, knnID))
                            .header("accept", "application/json")
                            .header("Content-Type", "application/json")
                            .asJson()
                            .getBody().getObject();
        } catch (UnirestException e) {
            e.printStackTrace();
        }

        return knn;
    }

    public JSONObject setKNNState(int knnID, String state) {
        JSONObject knn = new JSONObject();

        try {
            knn =
                    Unirest.delete(MessageFormat.format("http://{0}:{1}/deployment/{2}/model/{3}/state", host, port, deploymentID, knnID))
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

        return knn;
    }
}
