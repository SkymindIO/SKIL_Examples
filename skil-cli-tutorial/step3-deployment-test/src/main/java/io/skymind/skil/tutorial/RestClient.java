package io.skymind.skil.tutorial;


import com.sun.jersey.api.client.Client;
import com.sun.jersey.api.client.ClientResponse;
import com.sun.jersey.api.client.WebResource;
import org.codehaus.jettison.json.JSONObject;

import java.util.UUID;

public class RestClient {
    private String dataVecAddress = "localhost:9200";
    private String inferenceAddress = "localhost:9300";

    public static void main(String... args) {
        new RestClient().entryPoint(args);
    }

    public void entryPoint(String... args) {
        switch (args.length) {
            case 1:
                inferenceAddress = args[0];
                break;
            case 2:
                dataVecAddress = args[0];
                inferenceAddress = args[1];
                break;
            default:
                // Do nothing, leave as is
                break;
        }

        singleRecordExample();
        System.out.println("\n\n\n");
        batchRecordExample();
    }


    public void singleRecordExample() {
        try {
            String methodName = Thread.currentThread().getStackTrace()[1].getMethodName();
            System.out.println("****************************");
            System.out.println(String.format("** %22s **", methodName));
            System.out.println("****************************");

            String dataVecAddress = String.format("http://%s/transformincrementalarray", this.dataVecAddress);
            String inferenceAddress = String.format("http://%s/classify", this.inferenceAddress);

            System.out.format("Using dataVec address: %s\n", dataVecAddress);
            System.out.format("Using inference address: %s\n", inferenceAddress);

            System.out.format("Sending transform POST to %s ...\n", dataVecAddress);
            Client client = Client.create();

            WebResource dataVecEndpoint = client.resource(dataVecAddress);

            String input = "{\"values\":[\"7.2\",\"2.9\",\"5.8\",\"2.2\",\"99\"]}";

            System.out.println("Sending: " + input);

            ClientResponse response = dataVecEndpoint.type("application/json").post(ClientResponse.class, input);

            System.out.println("Response Status Code: " + response.getStatus());
            System.out.println("Response body:");
            JSONObject output = new JSONObject(response.getEntity(String.class));

            // If you use the /transformincremental endpoint you'll get:
            // {"values":["1.6383554656192094","-0.3551707113411996","1.1569427049526904","1.3120870592762433"]}

            // Since we want the base64 representation we use the transformedincrementalarray and get:
            // {"ndarray":"AAdKQVZBQ1BQAAAACAADSU5UAAAAAgAAAAEAAAAEAAAAAQAAAAEAAAAAAAAAAQAAAGMAB0pBVkFD\r\nUFAAAAAEAAVGTE9BVD/RtaK+tdjvP5QWsz+n8ng=\r\n"}
            System.out.println(output.toString());

            Thread.sleep(5 * 1000);

            System.out.format("Sending result as POST to inference server at %s ...\n", inferenceAddress);

            String array = output.getString("ndarray").replace("\r\n", "\\r\\n");
            String id = UUID.randomUUID().toString();

            // Creates a request like {"id": "848dadb2-86b7-46ae-b166-f31dd7a86a7c", "prediction": { "array": "AAdKQVZBQ1BQAAAACAADSU5UAAAAAgAAAAEAAAAEAAAAAQAAAAEAAAAAAAAAAQAAAGMAB0pBVkFD\r\nUFAAAAAEAAVGTE9BVD/RtaK+tdjvP5QWsz+n8ng=\r\n" } }
            String req = String.format("{\"id\": \"%s\", \"prediction\": { \"array\": \"%s\" } }", id, array);

            WebResource inferenceEndpoint = client.resource(inferenceAddress);

            ClientResponse prediction = inferenceEndpoint.type("application/json").post(ClientResponse.class, req);

            System.out.println("Classification Status Code: " + prediction.getStatus());
            System.out.println("Response Body:");

            // If you use the /classifyarray endpoint you'll get:
            // {"ndarray":"AAdKQVZBQ1BQAAAACAADSU5UAAAAAgAAAAEAAAADAAAAAQAAAAEAAAAAAAAAAQAAAGYAB0pBVkFD\r\nUFAAAAADAAVGTE9BVCWcUFAvmqMEP4AAAA==\r\n"}

            // Should print out {"results":[2],"probabilities":[1.0]}
            System.out.println(prediction.getEntity(String.class));

            System.out.println("Done");
            Thread.sleep(2 * 1000);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void batchRecordExample() {
        try {
            String methodName = Thread.currentThread().getStackTrace()[1].getMethodName();
            System.out.println("****************************");
            System.out.println(String.format("** %22s **", methodName));
            System.out.println("****************************");

            String dataVecAddress = String.format("http://%s/transformarray", this.dataVecAddress);
            String inferenceAddress = String.format("http://%s/classify", this.inferenceAddress);

            System.out.format("Using dataVec address: %s\n", dataVecAddress);
            System.out.format("Using inference address: %s\n", inferenceAddress);

            System.out.format("Sending transform POST to %s ...\n", dataVecAddress);
            Client client = Client.create();

            WebResource dataVecEndpoint = client.resource(dataVecAddress);

            String batchInput = "{\"records\":[{\"values\":[\"7.2\",\"2.9\",\"5.8\",\"2.2\",\"99\"]}," +
                    "{\"values\":[\"3.6\",\"2.7\",\"5.8\",\"2.2\",\"99\"]}," +
                    "{\"values\":[\"3.9\",\"1.7\",\"1.8\",\"1.8\",\"99\"]}" +
                    "]}";

            System.out.println("Sending: " + batchInput);

            ClientResponse response = dataVecEndpoint.type("application/json").post(ClientResponse.class, batchInput);

            System.out.println("Response Status Code: " + response.getStatus());
            System.out.println("Response body:");
            JSONObject output = new JSONObject(response.getEntity(String.class));

            // If you use the /transform endpoint you'll get:
            // {"records":[{"values":["1.6383554656192094","-0.3551707113411996","1.1569427049526904","1.3120870592762433"]},
            //              {"values":["-2.709123411208172","-0.8164313754206767","1.1569427049526904","1.3120870592762433"]},
            //              {"values":["-2.34683350480589","-3.1227346958180653","-1.1100906816299831","0.787951083533403"]}]}

            // Since we want the base64 representation we use the transformarray and get:
            // {"ndarray":"AAdKQVZBQ1BQAAAACAADSU5UAAAAAgAAAAMAAAAEAAAABAAAAAEAAAAAAAAAAQAAAGMAB0pBVkFD\r\nUFAAAAAMAAVGTE9BVD\/RtaK+tdjvP5QWsz+n8njALWJHv1EBpj+UFrM\/p\/J4wBYyhcBH2uO\/jhd0\r\nP0m3Kg==\r\n"}
            System.out.println(output.toString());

            Thread.sleep(5 * 1000);

            System.out.format("Sending result as POST to inference server at %s ...\n", inferenceAddress);

            String array = output.getString("ndarray").replace("\r\n", "\\r\\n");
            String id = UUID.randomUUID().toString();

            // Creates a request like {"id": "848dadb2-86b7-46ae-b166-f31dd7a86a7c", "prediction": { "array": "AAdKQVZBQ1BQAAAACAADSU5UAAAAAgAAAAEAAAAEAAAAAQAAAAEAAAAAAAAAAQAAAGMAB0pBVkFD\r\nUFAAAAAEAAVGTE9BVD/RtaK+tdjvP5QWsz+n8ng=\r\n" } }
            String req = String.format("{\"id\": \"%s\", \"prediction\": { \"array\": \"%s\" } }", id, array);

            WebResource inferenceEndpoint = client.resource(inferenceAddress);

            ClientResponse prediction = inferenceEndpoint.type("application/json").post(ClientResponse.class, req);

            System.out.println("Classification Status Code: " + prediction.getStatus());
            System.out.println("Response Body:");

            // If you use the /classifyarray endpoint you'll get:
            // {"ndarray":"AAdKQVZBQ1BQAAAACAADSU5UAAAAAgAAAAMAAAADAAAAAQAAAAMAAAAAAAAAAQAAAGYAB0pBVkFD\r\nUFAAAAAJAAVGTE9BVCWcUFA0ntxIOg6Wri+aovA4YZCrP38uWD+AAAA/f/x0Oy4CJQ==\r\n"}

            // Should print out {"results":[2,2,1],"probabilities":[1.0,0.999945878982544,0.9968008995056152]}
            System.out.println(prediction.getEntity(String.class));

            System.out.println("Done");
            Thread.sleep(2 * 1000);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
