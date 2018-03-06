package ai.skymind.skil.examples.clinical_lstm.modelserver.auth;


import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.exceptions.UnirestException;
import org.json.JSONObject;

import java.text.MessageFormat;


public class Authorization {

    private String host;
    private String port;

    public Authorization() {
        this.host = "localhost";
        this.port = "9008";
    }

    public Authorization(String host, String port) {
        this.host = host;
        this.port = port;
    }

    public String getAuthToken(String userId, String password) {
        String authToken = null;

        try {
            authToken =
                    Unirest.post(MessageFormat.format("http://{0}:{1}/login", host, port))
                            .header("accept", "application/json")
                            .header("Content-Type", "application/json")
                            .body(new JSONObject()
                                    .put("userId", userId)
                                    .put("password", password)
                                    .toString())
                            .asJson()
                            .getBody().getObject().getString("token");
        } catch (UnirestException e) {
            e.printStackTrace();
        }

        return authToken;
    }
}