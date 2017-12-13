package ai.skymind.skil.examples.endpoints;

import ai.skymind.skil.examples.request_objects.ModelDetails;
import org.json.JSONObject;

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

    public JSONObject addModel(ModelDetails modelDetails) {
        return null;
    }
}
