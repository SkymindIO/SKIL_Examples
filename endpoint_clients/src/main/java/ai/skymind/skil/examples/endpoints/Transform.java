package ai.skymind.skil.examples.endpoints;

import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.exceptions.UnirestException;
import org.json.JSONArray;
import org.json.JSONObject;

import java.text.MessageFormat;
import java.util.ArrayList;
import java.util.List;

public class Transform {

    private String host;
    private String port;
    private int deploymentID;

    public Transform(int deploymentID) {
        this.host = "localhost";
        this.port = "9008";
        this.deploymentID = deploymentID;
    }

    public Transform(String host, String port, int deploymentID) {
        this.host = host;
        this.port = port;
        this.deploymentID = deploymentID;
    }

    public JSONObject addTransform(String name, String fileLocation, int scale, String uri) {
        JSONObject transform = new JSONObject();

        try {
            List<String> uriList = new ArrayList<String>();
            uriList.add(uri);

            transform =
                    Unirest.post(MessageFormat.format("http://{0}:{1}/deployment/{2}/model", host, port, deploymentID))
                            .header("accept", "application/json")
                            .header("Content-Type", "application/json")
                            .body(new JSONObject()
                                    .put("name", name)
                                    .put("modelType", "transform")
                                    .put("fileLocation", fileLocation)
                                    .put("scale", scale)
                                    .put("uri", uriList)
                                    .toString())
                            .asJson()
                            .getBody().getObject();
        } catch (UnirestException e) {
            e.printStackTrace();
        }

        return transform;
    }

    public JSONObject reimportTransform(int transformID, String name, String fileLocation) {
        JSONObject transform = new JSONObject();

        try {
            transform =
                    Unirest.post(MessageFormat.format("http://{0}:{1}/deployment/{2}/model/{3}", host, port, deploymentID, transformID))
                            .header("accept", "application/json")
                            .header("Content-Type", "application/json")
                            .body(new JSONObject()
                                    .put("name", name)
                                    .put("modelType", "transform")
                                    .put("fileLocation", fileLocation)
                                    .toString())
                            .asJson()
                            .getBody().getObject();
        } catch (UnirestException e) {
            e.printStackTrace();
        }

        return transform;
    }

    public JSONObject deleteTransform(int transformID) {
        JSONObject transform = new JSONObject();

        try {
            transform =
                    Unirest.delete(MessageFormat.format("http://{0}:{1}/deployment/{2}/model/{3}", host, port, deploymentID, transformID))
                            .header("accept", "application/json")
                            .header("Content-Type", "application/json")
                            .asJson()
                            .getBody().getObject();
        } catch (UnirestException e) {
            e.printStackTrace();
        }

        return transform;
    }

    public JSONObject setTransformState(int transformID, String state) {
        JSONObject transform = new JSONObject();

        try {
            transform =
                    Unirest.delete(MessageFormat.format("http://{0}:{1}/deployment/{2}/model/{3}/state", host, port, deploymentID, transformID))
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

        return transform;
    }
}
